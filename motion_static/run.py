"""One training entry point for image and nested fields. Run from the repository root."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
import re
import time

import numpy as np
import torch

from .losses import compute_training_losses
from .models.hash_nf import HashNeuralField
from .models.coordinates import DEFAULT_BOUNDS
from .models.nested_mlp import NestedMLP
from .models.nested_hash_nf import NestedHashNF
from .models.nested_low_rank_hash_nf import NestedLowRankHashNF
from .models.low_rank_intensity import LowRankIntensity
from .models.oracle_branches import PHANTOM_REFERENCE_TIME, TrueMotion, TrueReferenceIntensity
from .models.time_grid import TimeGridField
from .phantoms import moby
from .phantoms.data import DEFAULT_PHANTOM, DEFAULT_SAMPLING, coordinate_grid, make_pair, sample
from .splits import block_split
from .evaluation.evaluate import evaluate, evaluate_image_only
from .evaluation.block_evaluate import evaluate_blocks
from .training_monitor import TrainingMonitor
from .run_records import (
    Tee,
    initialize_run_metadata,
    save_checkpoint,
    snapshot_source,
    update_summary,
    utc_now,
    write_json,
)


def build_model(config):
    # Older toy checkpoints omit bounds; all new runs save them explicitly.
    bounds = config.get("coordinate_bounds", DEFAULT_BOUNDS)
    reference_time = config.get("reference_time")
    if config["model"] == "hash_nf":
        return HashNeuralField(coordinate_bounds=bounds)
    if config["model"] == "nested_mlp":
        return NestedMLP(
            config.get("intensity_width", 128), config.get("motion_width", 32),
            coordinate_bounds=bounds, reference_time=reference_time,
        )
    if config["model"] == "nested_hash_nf":
        return NestedHashNF(
            width=config["head_width"], coordinate_bounds=bounds,
            reference_time=reference_time, **config["encoding"],
        )
    if config["model"] == "low_rank_hash_nf":
        # No motion module: spatial bases are evaluated at observation coordinates r.
        return LowRankIntensity(
            rank=config["rank"],
            width=config["head_width"],
            temporal_width=config["temporal_width"],
            coordinate_bounds=bounds,
            temporal_encoding=config.get("temporal_encoding"),
            **config["encoding"],
        )
    if config["model"] == "nested_low_rank_hash_nf":
        model = NestedLowRankHashNF(
            rank=config["rank"],
            width=config["head_width"],
            temporal_width=config["temporal_width"],
            coordinate_bounds=bounds,
            reference_time=reference_time,
            motion_encoding=config.get("motion_encoding"),
            temporal_encoding=config.get("temporal_encoding"),
            **config["encoding"],
        )
        return apply_oracle_branch(model, config)
    raise ValueError(f"Unknown model: {config['model']}")


def apply_oracle_branch(model, config):
    """Diagnostic only: replace one learned branch by the phantom's true field.

    Both branches are built first, so the learned branch's initialization and
    all later training batches match the run without an oracle branch.
    """
    branch = config.get("oracle_branch")
    if branch == "motion":
        model.motion = TrueMotion(reference_time=config["reference_time"])
    elif branch == "intensity":
        model.intensity = TrueReferenceIntensity(size=config["size"])
    elif branch is not None:
        raise ValueError(f"Unknown oracle branch: {branch}")
    return model


def build_optimizer(model, config):
    """Give hash tables and MLPs their configured learning rates.

    Keep group names and order stable: checkpoints and loss.csv use them.
    The low-rank temporal MLP is separate from its spatial hash and MLP.
    """
    model_name = config["model"]
    if model_name in ("hash_nf", "low_rank_hash_nf"):
        fields = [("image", model)]
    elif model_name in ("nested_hash_nf", "nested_low_rank_hash_nf"):
        # An oracle branch is fixed ground truth with nothing to optimize.
        oracle = config.get("oracle_branch")
        fields = []
        if oracle != "intensity":
            fields.append(("intensity", model.intensity))
        if oracle != "motion":
            fields.append(("motion", model.motion.field))
    else:
        return torch.optim.Adam([
            {"params": model.parameters(), "lr": config["mlp_lr"], "name": "mlp"},
        ])

    groups = []
    for branch_name, field in fields:
        groups.append({
            "params": field.encoding.parameters(),
            "lr": config["hash_lr"],
            "name": f"{branch_name}_hash",
        })
        groups.append({
            "params": field.mlp.parameters(),
            "lr": config["mlp_lr"],
            "name": f"{branch_name}_mlp",
        })
    learned_low_rank = config.get("oracle_branch") != "intensity"
    if model_name in ("low_rank_hash_nf", "nested_low_rank_hash_nf") and learned_low_rank:
        if model_name == "low_rank_hash_nf":
            temporal = model.temporal
            prefix = "image_time"
        else:
            temporal = model.intensity.temporal
            prefix = "intensity_time"
        # The time MLP may use its own rate; it defaults to the shared MLP rate.
        temporal_mlp_lr = config.get("temporal_mlp_lr", config["mlp_lr"])
        if isinstance(temporal, TimeGridField):
            # A time grid table is a feature table like a hash table.
            groups.append({
                "params": temporal.encoding.parameters(),
                "lr": config["hash_lr"],
                "name": f"{prefix}_grid",
            })
            temporal_parameters = temporal.mlp.parameters()
        else:
            temporal_parameters = temporal.parameters()
        groups.append({
            "params": temporal_parameters,
            "lr": temporal_mlp_lr,
            "name": f"{prefix}_mlp",
        })
    return torch.optim.Adam(groups)


def train(model, coordinates, intensities, config, *, out=None, validation=None):
    """One step = one random batch and one joint optimizer update.

    Loss definitions live in losses.compute_training_losses. Validation is used
    only by the monitor, after updates, without gradients. Passing out=None skips
    file output. Return the logged (step, image MSE, temporal L2, total) history
    and elapsed seconds; the model is updated in place.
    """
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["steps"])
    training_coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    observed_intensities = torch.as_tensor(intensities.reshape(-1), dtype=torch.float32)
    checkpoint_steps = set(config.get("checkpoint_steps", []))
    log_interval = config.get("log_interval", 500)
    history = []
    start = time.perf_counter()
    model.train()

    # Record the initial model on fixed samples, using a separate sampling RNG.
    monitor = None
    if out is not None:
        monitor = TrainingMonitor(out, coordinates, intensities, validation, config, optimizer)
        monitor.record(model, optimizer, 0, time.perf_counter() - start)

    for step in range(1, config["steps"] + 1):
        # 1. Sample observed space-time points, with replacement.
        indices = torch.randint(len(observed_intensities), (config["batch_size"],))
        batch_coordinates = training_coordinates[indices]
        batch_targets = observed_intensities[indices]

        # 2. Compute named loss terms and the objective before the update.
        losses = compute_training_losses(model, batch_coordinates, batch_targets, config)
        total_loss = losses["total_loss"]
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")

        # 3. Update all model branches jointly, then set the next step's rates.
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # 4. Log pre-update batch losses and post-update fixed-sample MSE.
        should_log = (
            step == 1
            or step % log_interval == 0
            or step == config["steps"]
            or step in checkpoint_steps
        )
        if should_log:
            scalar_losses = {name: value.item() for name, value in losses.items()}
            history.append((step, scalar_losses["batch_mse"],
                            scalar_losses["temporal_l2"], scalar_losses["total_loss"]))
            monitoring_text = ""
            if monitor is not None:
                row = monitor.record(
                    model, optimizer, step, time.perf_counter() - start,
                    batch_losses=scalar_losses,
                )
                monitoring_text = (
                    f", fixed train {row['fixed_training_mse']:.6g}, "
                    f"held-out {row.get('fixed_validation_mse', float('nan')):.6g}"
                )
            print(
                f"{config['phantom']}/{config['model']} step {step}: "
                f"MSE {scalar_losses['batch_mse']:.6g}, "
                f"temporal L2 {scalar_losses['temporal_l2']:.6g}, "
                f"total {scalar_losses['total_loss']:.6g}{monitoring_text}",
                flush=True,
            )

        if out is not None and step in checkpoint_steps:
            save_checkpoint(out, model, optimizer, scheduler, config, step)

    return history, time.perf_counter() - start


def parse_configuration(argv=None):
    """Resolve template + CLI overrides, validate, and choose a new output path.

    This function creates no files or models; --dry-run uses the same settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON template copy; defaults to experiments/template.json")
    parser.add_argument("--phantom", choices=("simple", "complex"), help="Explicitly override the template phantom")
    parser.add_argument("--model", choices=("hash_nf", "nested_mlp", "nested_hash_nf",
                                           "low_rank_hash_nf", "nested_low_rank_hash_nf"))
    parser.add_argument("--split", choices=("midpoints", "blocks"))
    parser.add_argument("--out", type=Path, help="New run directory; default is results/<UTC timestamp>_<name>")
    parser.add_argument("--name", help="Short run label for the automatic directory name")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--frames", type=int, help="Total training frames, including endpoints")
    parser.add_argument("--cycles", type=int, help="Complete periods; extend time without speeding up dynamics")
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", help="Optional intermediate checkpoints; default: none")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rank", type=int, help="Total space-time rank for either low-rank model")
    parser.add_argument("--temporal-weight", type=float)
    parser.add_argument("--temporal-step", type=float)
    parser.add_argument("--gifs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without creating files or training")
    args = parser.parse_args(argv)

    # Load the template, then apply explicit command-line overrides.
    root = Path(__file__).resolve().parent
    path = args.config or root / "experiments/template.json"
    if not path.exists():
        parser.error(f"Configuration does not exist: {path}")
    config = json.loads(path.read_text())
    if args.phantom is not None and args.phantom != config.get("phantom"):
        config.update(phantom=args.phantom, **DEFAULT_SAMPLING[args.phantom])
    config.setdefault("phantom", DEFAULT_PHANTOM)
    if args.model is not None:
        config["model"] = args.model
    measured = config["phantom"] == moby.NAME
    if measured:
        # The data file and window fix the grid, frame count and coordinate bounds.
        if args.frames is not None or args.cycles is not None:
            parser.error("dce_moby takes its window from config['data'], not --frames/--cycles")
        try:
            config["data"], window = moby.resolve_settings(config["data"])
        except (KeyError, OSError, ValueError) as error:
            parser.error(f"dce_moby data: {error}")
        config.update(window)
    elif config["phantom"] not in DEFAULT_SAMPLING:
        parser.error("Unknown phantom")
    else:
        for key, value in DEFAULT_SAMPLING[config["phantom"]].items():
            config.setdefault(key, value)
    for key in ("steps", "seed", "frames", "cycles", "checkpoint_steps", "rank",
                "temporal_weight", "temporal_step"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)

    # Validate the model, sampling budget and logging options.
    config["evaluation_split"] = args.split or config.get("evaluation_split", "midpoints")
    if config["evaluation_split"] not in ("midpoints", "blocks"):
        parser.error("evaluation_split must be midpoints or blocks")
    low_rank = config["model"] in ("low_rank_hash_nf", "nested_low_rank_hash_nf")
    nested = config["model"] in ("nested_mlp", "nested_hash_nf", "nested_low_rank_hash_nf")
    if args.rank is not None and not low_rank:
        parser.error("--rank requires low_rank_hash_nf or nested_low_rank_hash_nf")
    if low_rank and config["rank"] < 1:
        parser.error("rank must be positive")
    for key in ("steps", "batch_size", "threads", "size", "frames", "cycles"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            parser.error(f"{key} must be a positive integer")
    if config["frames"] < 2 or config["size"] < 16:
        parser.error("frames must be at least 2 and size must be at least 16")
    config.setdefault("checkpoint_steps", [])
    if any(step < 1 or step > config["steps"] for step in config["checkpoint_steps"]):
        parser.error("Checkpoint steps must be within the training budget")
    for key, default in (("monitor_seed", 2026), ("monitor_samples", 65536), ("log_interval", 500)):
        config.setdefault(key, default)
    if config["monitor_samples"] < 1 or config["log_interval"] < 1:
        parser.error("monitor_samples and log_interval must be positive")

    # Large visual/array exports are opt-in; numeric losses are always recorded.
    options = {"gifs": False, "predictions": False, **config.get("output", {})}
    if args.gifs is not None:
        options["gifs"] = args.gifs
    if args.save_predictions is not None:
        options["predictions"] = args.save_predictions
    if set(options) != {"gifs", "predictions"} or any(type(value) is not bool for value in options.values()):
        parser.error("output accepts boolean gifs and predictions only")
    config["output"] = options

    # Physical coordinates span complete periods, with the first frame as reference.
    if measured:
        bounds = config["coordinate_bounds"]  # mm and seconds, from the data file
        if config["evaluation_split"] != "midpoints":
            parser.error("dce_moby supports the midpoints split only")
        if options["gifs"]:
            parser.error("GIFs are not implemented for dce_moby; use output.predictions")
    else:
        bounds = [[-.5, .5], [-.5, .5], [-.5, config["cycles"] - .5]]
        if "coordinate_bounds" in config and config["coordinate_bounds"] != bounds:
            parser.error("coordinate_bounds do not match the configured phantom domain")
    config["coordinate_bounds"] = bounds
    if nested and config.get("reference_time") is None:
        config["reference_time"] = bounds[2][0]
    if config["model"] == "low_rank_hash_nf":
        config.pop("motion_encoding", None)
        config.pop("reference_time", None)

    # The optional existing temporal penalty uses physical time differences.
    weight = config.get("temporal_weight", 0.)
    if not np.isfinite(weight) or weight < 0:
        parser.error("temporal_weight must be finite and nonnegative")
    if weight > 0 and not nested:
        parser.error("Temporal intensity regularization requires a nested model")
    if args.temporal_step is not None and weight == 0:
        parser.error("--temporal-step requires positive temporal_weight")
    if weight > 0:
        delta = config.get("temporal_step", float(np.diff(np.linspace(*bounds[2], config["frames"])).min()))
        if not np.isfinite(delta) or not 0 < delta <= config["cycles"]:
            parser.error("temporal_step must be positive and no larger than the time span")
        config.update(temporal_weight=weight, temporal_step=delta,
                      temporal_sampling="detached warped training coordinates; same q at both times",
                      temporal_penalty="mean squared finite-difference time derivative")

    # Optional time-network settings exist only for a learned low-rank intensity.
    time_grid = config.get("temporal_encoding")
    temporal_mlp_lr = config.get("temporal_mlp_lr")
    if time_grid is not None or temporal_mlp_lr is not None:
        if not low_rank or config.get("oracle_branch") == "intensity":
            parser.error("temporal_encoding and temporal_mlp_lr require a learned low-rank intensity")
    if time_grid is not None:
        keys = {"levels", "features", "min_resolution", "max_resolution"}
        if (not isinstance(time_grid, dict) or set(time_grid) != keys
                or any(type(time_grid[key]) is not int or time_grid[key] < 1 for key in keys)
                or time_grid["max_resolution"] < time_grid["min_resolution"]):
            parser.error("temporal_encoding needs positive integer levels, features, min_resolution <= max_resolution")
    if temporal_mlp_lr is not None and not (np.isfinite(temporal_mlp_lr) and temporal_mlp_lr > 0):
        parser.error("temporal_mlp_lr must be finite and positive")

    # Oracle diagnostics replace one branch by the complex phantom's true field.
    oracle = config.get("oracle_branch")
    if oracle is not None:
        if oracle not in ("motion", "intensity"):
            parser.error("oracle_branch must be motion or intensity")
        if (config["model"] != "nested_low_rank_hash_nf" or config["phantom"] != "complex"
                or config["evaluation_split"] != "midpoints"):
            parser.error("oracle_branch requires nested_low_rank_hash_nf, the complex phantom and midpoints")
        if config["reference_time"] != PHANTOM_REFERENCE_TIME:
            parser.error("oracle_branch requires the phantom's motion reference time -0.5")
        if weight > 0:
            parser.error("oracle_branch does not support the temporal penalty")

    # Blocks are a separate protocol, with their own grid divisibility constraint.
    if config["evaluation_split"] == "blocks":
        config.setdefault("block_split", dict(block_shape=[8, 8, 2], test_fraction=.2, seed=0))
        shape = (config["size"], config["size"], config["frames"])
        if any(n % b for n, b in zip(shape, config["block_split"]["block_shape"])):
            parser.error("Block holdout requires each grid size divisible by its block size; choose explicit compatible settings")

    # Never overwrite a run, including an incomplete one.
    name = args.name or f"{config['model']}_seed{config['seed']}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        parser.error("name must contain only letters, digits, underscore, hyphen or period")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = args.out or root / "results" / f"{stamp}_{name}"
    if out.exists():
        parser.error("Output already exists; choose a new --out")
    return args, config, out


def prepare_data(config):
    """Use saved settings for training or later inference, with no output files."""
    if config["phantom"] == moby.NAME:
        return moby.make_pair(config["data"]), None
    if config.get("evaluation_split", "midpoints") == "midpoints":
        pair = make_pair(config["phantom"], config["size"], config["frames"], config["cycles"])
        return pair, None
    times = np.linspace(-.5, config["cycles"] - .5, config["frames"])
    training = sample(config["phantom"], config["size"], times, (-.5, config["cycles"] - .5))
    split = block_split(training["truth"].shape, **config["block_split"])
    return {"training": training}, split


def execute(config, out):
    """Run one explicitly selected configuration and save a flat record."""
    root = Path(__file__).resolve().parent
    initialize_run_metadata(config)
    out.mkdir(parents=True, exist_ok=False)
    with (
        (out / "run.log").open("w") as log,
        redirect_stdout(Tee(sys.stdout, log)),
        redirect_stderr(Tee(sys.stderr, log)),
    ):
        try:
            # Save provenance before data/model construction so failures are traceable.
            config["code_provenance"] = snapshot_source(root, out)
            write_json(out / "config.json", config)
            print(f"Phantom: {config['phantom']} | size: {config['size']}x{config['size']} | "
                  f"cycles: {config['cycles']} | training frames: {config['frames']} | time: {config['time_bounds']}", flush=True)
            torch.set_num_threads(config["threads"])
            torch.manual_seed(config["seed"])

            # Build observed and held-out pools; held-out samples go only to the monitor.
            data, split = prepare_data(config)
            training = data["training"]
            coordinates = coordinate_grid(training["x"], training["y"], training["t"])
            intensities = training["truth"].reshape(-1)
            if split is None:
                held_out = data["midpoint"]
                validation = (
                    coordinate_grid(held_out["x"], held_out["y"], held_out["t"]),
                    held_out["truth"].reshape(-1),
                )
            else:
                config.update(split=split["settings"], held_out_labels_used_for_training=False)
                observed = split["train_mask"].reshape(-1)
                hidden = split["test_mask"].reshape(-1)
                validation = (coordinates[hidden], intensities[hidden])
                coordinates = coordinates[observed]
                intensities = intensities[observed]
                # Preserve the exact evaluation mask; monitoring indices refer to these selected pools.
                np.savez_compressed(
                    out / "split.npz",
                    train_mask=split["train_mask"], test_mask=split["test_mask"],
                    test_blocks=split["test_blocks"],
                    x=training["x"], y=training["y"], t=training["t"],
                )

            # Optimize, then retain the final weights for later inference.
            model = build_model(config)
            write_json(out / "config.json", config)
            _, seconds = train(model, coordinates, intensities, config, out=out, validation=validation)
            torch.save(dict(config=config, step=config["steps"], state_dict=model.state_dict()), out / "model.pt")

            # Evaluate the complete configured grid, independently of export options.
            model.eval()
            if config["phantom"] == moby.NAME:
                # No true motion or reference intensity exists for measured data.
                evaluate_image_only(model, data, config, out, seconds)
            elif split is None:
                evaluate(model, data, config, out, seconds)
            else:
                evaluate_blocks(model, training, split, config, out, seconds)
            config.update(status="complete", finished_at=utc_now())
            write_json(out / "config.json", config)
            update_summary(out.parent)
            print(f"Saved results: {out}", flush=True)
        except BaseException as error:
            # Keep partial files and the original error; never label a failed run complete.
            config.update(status="failed", finished_at=utc_now(), error=f"{type(error).__name__}: {error}")
            write_json(out / "config.json", config)
            traceback.print_exc()
            try:
                update_summary(out.parent)
            except Exception:
                traceback.print_exc()
            raise


def main(argv=None):
    args, config, out = parse_configuration(argv)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return config
    execute(config, out)


if __name__ == "__main__":
    main()
