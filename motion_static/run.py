"""One training entry point for image and nested fields. Run from the repository root."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
import platform
import re
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from .losses import intensity_temporal_l2
from .models.hash_nf import HashNeuralField
from .models.coordinates import DEFAULT_BOUNDS
from .models.nested_mlp import NestedMLP
from .models.nested_hash_nf import NestedHashNF
from .models.nested_low_rank_hash_nf import NestedLowRankHashNF
from .phantoms.data import DEFAULT_PHANTOM, DEFAULT_SAMPLING, coordinate_grid, make_pair, sample
from .splits import block_split
from .evaluation.evaluate import evaluate
from .evaluation.block_evaluate import evaluate_blocks
from .training_monitor import TrainingMonitor
from .run_records import Tee, update_summary, utc_now, write_json


def build_model(config):
    # Older toy checkpoints omit bounds; all new runs save them explicitly.
    bounds = config.get("coordinate_bounds", DEFAULT_BOUNDS)
    reference_time = config.get("reference_time")
    if config["model"] == "hash_nf":
        return HashNeuralField(coordinate_bounds=bounds)
    if config["model"] == "nested_mlp":
        return NestedMLP(config.get("intensity_width", 128), config.get("motion_width", 32),
                         coordinate_bounds=bounds, reference_time=reference_time)
    if config["model"] == "nested_hash_nf":
        return NestedHashNF(width=config["head_width"], coordinate_bounds=bounds,
                            reference_time=reference_time, **config["encoding"])
    if config["model"] == "nested_low_rank_hash_nf":
        return NestedLowRankHashNF(
            rank=config["rank"], width=config["head_width"], temporal_width=config["temporal_width"],
            coordinate_bounds=bounds, reference_time=reference_time,
            motion_encoding=config.get("motion_encoding"), **config["encoding"])
    raise ValueError(f"Unknown model: {config['model']}")


def train(model, coordinates, intensities, config, *, out=None, validation=None):
    """Optimize observed images only; optional validation is read-only monitoring."""
    if config["model"] in ("hash_nf", "nested_hash_nf", "nested_low_rank_hash_nf"):
        fields = [model] if config["model"] == "hash_nf" else [model.intensity, model.motion.field]
        groups = []
        for index, field in enumerate(fields):
            name = "image" if config["model"] == "hash_nf" else ("intensity", "motion")[index]
            groups.extend([dict(params=field.encoding.parameters(), lr=config["hash_lr"], name=f"{name}_hash"),
                           dict(params=field.mlp.parameters(), lr=config["mlp_lr"], name=f"{name}_mlp")])
        if config["model"] == "nested_low_rank_hash_nf":
            groups.append(dict(params=model.intensity.temporal.parameters(), lr=config["mlp_lr"], name="intensity_time_mlp"))
    else:
        groups = [dict(params=model.parameters(), lr=config["mlp_lr"], name="mlp")]
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["steps"])
    x = torch.as_tensor(coordinates, dtype=torch.float32)
    y = torch.as_tensor(intensities.reshape(-1), dtype=torch.float32)
    history = []
    temporal_weight = config.get("temporal_weight", 0.)
    start = time.perf_counter()
    model.train()
    monitor = None if out is None else TrainingMonitor(out, coordinates, intensities, validation, config, optimizer)
    if monitor is not None:
        monitor.record(model, optimizer, 0, time.perf_counter() - start)
    checkpoint_steps = set(config.get("checkpoint_steps", []))
    for step in range(1, config["steps"] + 1):
        indices = torch.randint(len(y), (config["batch_size"],))
        batch = x[indices]
        prediction = model(batch)
        data_loss = (prediction - y[indices]).square().mean()
        temporal_loss = data_loss.new_zeros(())
        if temporal_weight > 0:
            temporal_loss = intensity_temporal_l2(
                model, batch, config["temporal_step"], config["coordinate_bounds"][2][1])
        loss = data_loss + temporal_weight * temporal_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % config.get("log_interval", 500) == 0 or step == config["steps"] or step in checkpoint_steps:
            losses = (data_loss.item(), temporal_loss.item(), loss.item())
            history.append((step, *losses))
            extra = ""
            if monitor is not None:
                row = monitor.record(model, optimizer, step, time.perf_counter() - start, losses)
                extra = (f", fixed train {row['fixed_training_mse']:.6g}, "
                         f"held-out {row.get('fixed_validation_mse', float('nan')):.6g}")
            print(f"{config['phantom']}/{config['model']} step {step}: "
                  f"MSE {data_loss.item():.6g}, temporal L2 {temporal_loss.item():.6g}, "
                  f"total {loss.item():.6g}{extra}", flush=True)
        if out is not None and step in checkpoint_steps:
            checkpoint_dir = Path(out) / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(dict(config=config, step=step, state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            torch_rng_state=torch.get_rng_state()), checkpoint_dir / f"step_{step:06d}.pt")
    return history, time.perf_counter() - start


def snapshot_source(root, out):
    """Save the actual source used, including uncommitted code, without committing."""
    hashes = {}
    with ZipFile(out / "source.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(root)
            if relative.parts[0] in ("archive", "results"):
                continue
            name = str(Path(root.name) / relative)
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=False)
    return dict(git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
                source_archive="source.zip", source_sha256=hashes)


def parse_configuration(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON template copy; defaults to experiments/template.json")
    parser.add_argument("--phantom", choices=("simple", "complex"), help="Explicitly override the template phantom")
    parser.add_argument("--model", choices=("hash_nf", "nested_mlp", "nested_hash_nf", "nested_low_rank_hash_nf"))
    parser.add_argument("--split", choices=("midpoints", "blocks"))
    parser.add_argument("--out", type=Path, help="New run directory; default is results/<UTC timestamp>_<name>")
    parser.add_argument("--name", help="Short run label for the automatic directory name")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--frames", type=int, help="Total training frames, including endpoints")
    parser.add_argument("--cycles", type=int, help="Complete periods; extend time without speeding up dynamics")
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", help="Optional intermediate checkpoints; default: none")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rank", type=int, help="Total intensity rank, for nested_low_rank_hash_nf only")
    parser.add_argument("--temporal-weight", type=float)
    parser.add_argument("--temporal-step", type=float)
    parser.add_argument("--gifs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without creating files or training")
    args = parser.parse_args(argv)
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
    if config["phantom"] not in DEFAULT_SAMPLING:
        parser.error("Unknown phantom")
    for key, value in DEFAULT_SAMPLING[config["phantom"]].items():
        config.setdefault(key, value)
    for key in ("steps", "seed", "frames", "cycles", "checkpoint_steps", "rank",
                "temporal_weight", "temporal_step"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    config["evaluation_split"] = args.split or config.get("evaluation_split", "midpoints")
    if config["evaluation_split"] not in ("midpoints", "blocks"):
        parser.error("evaluation_split must be midpoints or blocks")
    if args.rank is not None and config["model"] != "nested_low_rank_hash_nf":
        parser.error("--rank requires nested_low_rank_hash_nf")
    if config["model"] == "nested_low_rank_hash_nf" and config["rank"] < 1:
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
    options = {"gifs": False, "predictions": False, **config.get("output", {})}
    if args.gifs is not None:
        options["gifs"] = args.gifs
    if args.save_predictions is not None:
        options["predictions"] = args.save_predictions
    if set(options) != {"gifs", "predictions"} or any(type(value) is not bool for value in options.values()):
        parser.error("output accepts boolean gifs and predictions only")
    config["output"] = options
    bounds = [[-.5, .5], [-.5, .5], [-.5, config["cycles"] - .5]]
    if "coordinate_bounds" in config and config["coordinate_bounds"] != bounds:
        parser.error("coordinate_bounds do not match the configured phantom domain")
    config["coordinate_bounds"] = bounds
    if config["model"] != "hash_nf" and config.get("reference_time") is None:
        config["reference_time"] = bounds[2][0]
    weight = config.get("temporal_weight", 0.)
    if not np.isfinite(weight) or weight < 0:
        parser.error("temporal_weight must be finite and nonnegative")
    if weight > 0 and config["model"] == "hash_nf":
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
    if config["evaluation_split"] == "blocks":
        config.setdefault("block_split", dict(block_shape=[8, 8, 2], test_fraction=.2, seed=0))
        shape = (config["size"], config["size"], config["frames"])
        if any(n % b for n, b in zip(shape, config["block_split"]["block_shape"])):
            parser.error("Block holdout requires each grid size divisible by its block size; choose explicit compatible settings")
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
    config.pop("error", None)
    config.pop("finished_at", None)
    config.update(status="running", started_at=utc_now(),
                  loss="MSE(predicted image, observed image)", device="cpu", scheduler="cosine to zero",
                  torch_version=str(torch.__version__), numpy_version=np.__version__,
                  python_version=platform.python_version(), platform=platform.platform(),
                  true_motion_used_for_training=False, reference_intensity_labels_used_for_training=False,
                  midpoint_labels_used_for_training=False,
                  joint_period=1., time_bounds=config["coordinate_bounds"][2],
                  intervals_per_cycle=(config["frames"] - 1) / config["cycles"],
                  monitoring="Fixed uniform samples without replacement; image MSE; no validation gradients",
                  monitor_lr_convention="Learning rates after scheduler step, for the next optimizer update")
    if config.get("temporal_weight", 0) > 0:
        config["loss"] += " + temporal_weight * intensity_temporal_l2"
    out.mkdir(parents=True, exist_ok=False)
    with (out / "run.log").open("w") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            config["code_provenance"] = snapshot_source(root, out)
            write_json(out / "config.json", config)
            print(f"Phantom: {config['phantom']} | size: {config['size']}x{config['size']} | "
                  f"cycles: {config['cycles']} | training frames: {config['frames']} | time: {config['time_bounds']}", flush=True)
            torch.set_num_threads(config["threads"])
            torch.manual_seed(config["seed"])
            data, split = prepare_data(config)
            training = data["training"]
            coordinates = coordinate_grid(training["x"], training["y"], training["t"])
            intensities = training["truth"].reshape(-1)
            if split is None:
                held_out = data["midpoint"]
                validation = (coordinate_grid(held_out["x"], held_out["y"], held_out["t"]), held_out["truth"].reshape(-1))
            else:
                config.update(split=split["settings"], held_out_labels_used_for_training=False)
                observed, hidden = split["train_mask"].reshape(-1), split["test_mask"].reshape(-1)
                validation = (coordinates[hidden], intensities[hidden])
                coordinates, intensities = coordinates[observed], intensities[observed]
                # Preserve the exact evaluation mask; monitoring indices refer to these selected pools.
                np.savez_compressed(out / "split.npz", train_mask=split["train_mask"], test_mask=split["test_mask"],
                                    test_blocks=split["test_blocks"], x=training["x"], y=training["y"], t=training["t"])
            model = build_model(config)
            write_json(out / "config.json", config)
            _, seconds = train(model, coordinates, intensities, config, out=out, validation=validation)
            torch.save(dict(config=config, step=config["steps"], state_dict=model.state_dict()), out / "model.pt")
            model.eval()
            if split is None:
                evaluate(model, data, config, out, seconds)
            else:
                evaluate_blocks(model, training, split, config, out, seconds)
            config.update(status="complete", finished_at=utc_now())
            write_json(out / "config.json", config)
            update_summary(out.parent)
            print(f"Saved results: {out}", flush=True)
        except BaseException as error:
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
