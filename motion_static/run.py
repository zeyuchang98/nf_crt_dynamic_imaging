"""One training entry point for image and nested fields. Run from the repository root."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
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


def build_model(config):
    # Older toy checkpoints omit bounds; all new runs save them explicitly.
    bounds = config.get("coordinate_bounds", DEFAULT_BOUNDS)
    reference_time = config.get("reference_time")
    if config["model"] == "hash_nf":
        return HashNeuralField(coordinate_bounds=bounds)
    if config["model"] == "nested_mlp":
        return NestedMLP(config["intensity_width"], config["motion_width"],
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phantom", choices=("simple", "complex"), default=DEFAULT_PHANTOM,
                        help="Phantom to fit (default: complex)")
    parser.add_argument("--model", choices=("hash_nf", "nested_mlp", "nested_hash_nf", "nested_low_rank_hash_nf"), required=True)
    parser.add_argument("--split", choices=("midpoints", "blocks"), default="midpoints")
    parser.add_argument("--out", type=Path, help="Use a new folder to preserve completed experiments")
    parser.add_argument("--config", type=Path, help="Explicit experiment JSON; phantom and model must match")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--frames", type=int, help="Total training frames over all cycles, including endpoints")
    parser.add_argument("--cycles", type=int, help="Complete periods; extend time without speeding up dynamics")
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", help="Save intermediate model/optimizer states")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rank", type=int, help="Total intensity rank, for nested_low_rank_hash_nf only")
    parser.add_argument("--temporal-weight", type=float, default=0., help="Weight of intensity time-derivative L2; 0 disables it")
    parser.add_argument("--temporal-step", type=float, help="Time difference in input units; defaults to one training interval")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    config_path = args.config or root / "experiments" / args.phantom / f"{args.model}.json"
    if not config_path.exists():
        parser.error(f"No experiment configuration for {args.phantom}/{args.model}")
    config = json.loads(config_path.read_text())
    if config["phantom"] != args.phantom or config["model"] != args.model:
        parser.error("Configuration phantom/model must match the command")
    for key, value in DEFAULT_SAMPLING[args.phantom].items():
        config.setdefault(key, value)
    if args.rank is not None:
        if args.model != "nested_low_rank_hash_nf" or args.rank < 1:
            parser.error("--rank requires nested_low_rank_hash_nf and a positive integer")
        config["rank"] = args.rank
    if args.split == "blocks":
        split_config = json.loads((root / "experiments" / "splits" / "blocks.json").read_text())
    for key in ("steps", "seed", "frames", "cycles", "checkpoint_steps"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if min(config[k] for k in ("steps", "batch_size", "threads")) < 1:
        parser.error("steps, batch_size, threads must be positive")
    if (config["frames"] < 2 or not isinstance(config["cycles"], int)
            or isinstance(config["cycles"], bool) or config["cycles"] < 1):
        parser.error("frames must be at least 2 and cycles must be positive")
    if any(step < 1 or step > config["steps"] for step in config.get("checkpoint_steps", [])):
        parser.error("Checkpoint steps must be within the training budget")
    for key, default in (("monitor_seed", 2026), ("monitor_samples", 65536), ("log_interval", 500)):
        config.setdefault(key, default)
    if config["monitor_samples"] < 1 or config["log_interval"] < 1:
        parser.error("monitor_samples and log_interval must be positive")
    if not np.isfinite(args.temporal_weight) or args.temporal_weight < 0:
        parser.error("--temporal-weight must be finite and nonnegative")
    if args.temporal_weight > 0 and args.model == "hash_nf":
        parser.error("Temporal intensity regularization requires a nested model")
    if args.temporal_step is not None and args.temporal_weight == 0:
        parser.error("--temporal-step requires a positive --temporal-weight")
    result_root = root / "results" / args.phantom
    if args.split == "blocks":
        result_root = result_root / "block_split"
    default_out = (result_root / "low_rank" / f"rank_{config['rank']}"
                   if args.model == "nested_low_rank_hash_nf" else result_root / args.model)
    out = args.out or default_out
    if out.exists():
        parser.error("Output already exists; choose a new --out")
    print(f"Phantom: {config['phantom']} | size: {config['size']}x{config['size']} | "
          f"cycles: {config['cycles']} | training frames: {config['frames']} | "
          f"time: [-0.5, {config['cycles'] - 0.5}]", flush=True)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    if args.split == "midpoints":
        data = make_pair(config["phantom"], config["size"], config["frames"], config["cycles"])
        training = data["training"]
        split = None
    else:
        time_bounds = (-.5, config["cycles"] - .5)
        times = np.linspace(*time_bounds, config["frames"])
        training = sample(config["phantom"], config["size"], times, time_bounds)
        split = block_split(training["truth"].shape, **split_config)
        config["split"] = split["settings"]
    coordinates = coordinate_grid(training["x"], training["y"], training["t"])
    # Use the complete coordinate domain, never the extrema of a random batch.
    data_bounds = [[float(training[axis][0]), float(training[axis][-1])] for axis in ("x", "y", "t")]
    config.setdefault("coordinate_bounds", data_bounds)
    if not np.array_equal(config["coordinate_bounds"], data_bounds):
        parser.error("Configured coordinate_bounds do not match the generated phantom domain")
    if args.temporal_weight > 0:
        delta = args.temporal_step if args.temporal_step is not None else float(np.diff(training["t"]).min())
        if not np.isfinite(delta) or not 0 < delta <= np.ptp(training["t"]):
            parser.error("--temporal-step must be positive and no larger than the time span")
        config.update(temporal_weight=args.temporal_weight, temporal_step=delta,
                      temporal_sampling="detached warped training coordinates; same q at both times",
                      temporal_penalty="mean squared finite-difference time derivative")
    if config["model"] != "hash_nf" and config.get("reference_time") is None:
        config["reference_time"] = config["coordinate_bounds"][2][0]
    model = build_model(config)
    out.mkdir(parents=True)
    intensities = training["truth"].reshape(-1)
    if split is None:
        held_out = data["midpoint"]
        validation = (coordinate_grid(held_out["x"], held_out["y"], held_out["t"]), held_out["truth"].reshape(-1))
    if split is not None:
        # Only observed coordinates and intensities are ever passed to train().
        observed = split["train_mask"].reshape(-1)
        hidden = split["test_mask"].reshape(-1)
        validation = (coordinates[hidden], intensities[hidden])
        coordinates, intensities = coordinates[observed], intensities[observed]
        np.savez_compressed(out / "split.npz", train_mask=split["train_mask"], test_mask=split["test_mask"],
                            test_blocks=split["test_blocks"], x=training["x"], y=training["y"], t=training["t"])
    config.update(loss="MSE(predicted image, observed image)", device="cpu",
                  scheduler="cosine to zero", torch_version=str(torch.__version__),
                  true_motion_used_for_training=False, reference_intensity_labels_used_for_training=False,
                  midpoint_labels_used_for_training=False)
    config.update(joint_period=1., time_bounds=config["coordinate_bounds"][2],
                  intervals_per_cycle=(config["frames"] - 1) / config["cycles"],
                  monitoring="Fixed uniform samples without replacement; image MSE; no validation gradients",
                  monitor_lr_convention="Learning rates after scheduler step, for the next optimizer update")
    if args.temporal_weight > 0:
        config["loss"] += " + temporal_weight * intensity_temporal_l2"
    if split is not None:
        config["held_out_labels_used_for_training"] = False
    config["code_provenance"] = snapshot_source(root, out)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    _, seconds = train(model, coordinates, intensities, config, out=out, validation=validation)
    torch.save(dict(config=config, step=config["steps"], state_dict=model.state_dict()), out / "model.pt")
    model.eval()
    if split is None:
        evaluate(model, data, config, out, seconds)
    else:
        evaluate_blocks(model, training, split, config, out, seconds)
    print(f"Saved results: {out}", flush=True)


if __name__ == "__main__":
    main()
