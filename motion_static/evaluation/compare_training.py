"""Compare saved training budgets and checkpoints using existing branch metrics.

python -m motion_static.evaluation.compare_training RUN_8K RUN_32K --out OUTPUT
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import motion_scores, predict
from .metrics import error_metrics, save_metrics
from .plots import pyplot, save_comparison
from ..phantoms.data import coordinate_grid, make_pair, sample
from ..run_records import read_report


def load_model(path):
    # Local import avoids a circular dependency with the training entry point.
    from ..run import build_model
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def scores(model, data):
    """Same full-grid RRMSE and foreground motion/Jacobian definitions as evaluate()."""
    coordinates = coordinate_grid(data["x"], data["y"], data["t"])
    shape = data["truth"].shape
    image = predict(model, coordinates).reshape(shape)
    intensity = predict(model.intensity, coordinates).reshape(shape)
    motion = predict(model.motion, coordinates).reshape(*shape, 2)
    motion_report = motion_scores(data["u_true"], motion, data["foreground_mask"],
                                  float(data["x"][1] - data["x"][0]))
    return dict(image_rrmse=error_metrics(data["truth"], image, data_range=1)["rrmse"],
                intensity_rrmse=error_metrics(data["reference_intensity"], intensity, data_range=1)["rrmse"],
                motion_rmse_pixels=motion_report["foreground"]["endpoint_rmse_pixels"],
                nonpositive_jacobian_fraction=motion_report["foreground_nonpositive_jacobian_fraction"])


def summary_scores(summary):
    result = {}
    for split, name in (("training", "training_grid"), ("midpoint", "unseen_time_midpoints")):
        result[split] = dict(image_rrmse=summary["image"][name]["rrmse"],
                             intensity_rrmse=summary["reference_intensity"][name]["rrmse"],
                             motion_rmse_pixels=summary["motion"][split]["foreground"]["endpoint_rmse_pixels"],
                             nonpositive_jacobian_fraction=summary["motion"][split]["foreground_nonpositive_jacobian_fraction"])
    return result


def validate_configs(configs, *, vary_motion_grid=False, vary_intensity_grid=False):
    """Keep data and training conventions fixed when comparing spatial grids."""
    if vary_motion_grid and vary_intensity_grid:
        raise ValueError("Vary only one branch's spatial grid per comparison")
    config = configs[0]
    for other in configs[1:]:
        for key in ("phantom", "model", "size", "frames", "cycles", "seed", "rank", "coordinate_bounds"):
            if config[key] != other[key]:
                raise ValueError(f"Comparison requires identical {key}")
        if vary_intensity_grid:
            # Only the intensity grid's maximum resolution is varied in this test.
            first_encoding = {k: v for k, v in config["encoding"].items() if k != "max_resolution"}
            other_encoding = {k: v for k, v in other["encoding"].items() if k != "max_resolution"}
            if first_encoding != other_encoding:
                raise ValueError("Intensity grid comparison may only change max_resolution")
        elif config["encoding"] != other["encoding"]:
            raise ValueError("Intensity grid changes require --vary-intensity-grid")
        if vary_motion_grid or vary_intensity_grid:
            for key in ("steps", "batch_size", "head_width", "temporal_width", "mlp_lr", "hash_lr",
                        "reference_time", "loss", "scheduler"):
                if config[key] != other[key]:
                    raise ValueError(f"Spatial grid comparison requires identical {key}")
        if not vary_motion_grid and config.get("motion_encoding") != other.get("motion_encoding"):
            raise ValueError("Motion grid changes require --vary-motion-grid")


def plot_comparison(runs, traces, out, labels=None, note=None):
    plt = pyplot(out)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for run, label in zip(runs, labels or [run.name for run in runs]):
        with (run / "loss.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for ax, key, title in zip(axes, ("fixed_training_mse", "fixed_validation_mse"),
                                  ("Fixed training samples", "Fixed held-out midpoint samples")):
            ax.semilogy([int(row["step"]) for row in rows], [float(row[key]) for row in rows], label=label)
            ax.set(xlabel="Optimizer steps", ylabel="Image MSE", title=title)
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle(note or "Same 3-cycle data and initialization; cosine schedule ends at each run's budget")
    fig.savefig(out / "loss_comparison.png", dpi=160)
    fig.savefig(out / "loss_comparison.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    metrics = (("image_rrmse", "Image RRMSE (%)", 100),
               ("intensity_rrmse", "Reference intensity RRMSE (%)", 100),
               ("motion_rmse_pixels", "Foreground motion RMSE (pixels)", 1),
               ("nonpositive_jacobian_fraction", "Foreground nonpositive Jacobian (%)", 100))
    for ax, (key, title, scale) in zip(axes.flat, metrics):
        for index, (name, entries) in enumerate(traces.items()):
            for split, style in (("training", "-"), ("midpoint", "--")):
                ax.plot([entry["step"] for entry in entries],
                        [scale * entry["scores"][split][key] for entry in entries],
                        color=f"C{index}", linestyle=style, marker="o", label=f"{name} / {split}")
        ax.set(xlabel="Optimizer steps", ylabel=title)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Full-grid branch diagnostics; checkpoint labels are never used for gradients")
    fig.savefig(out / "branch_convergence.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gifs", action="store_true", help="Also render the motion comparison for each run")
    parser.add_argument("--previous", type=Path, help="Old one-cycle model, scored on common unseen first-cycle times")
    parser.add_argument("--labels", nargs="+", help="Unique display labels, one per run")
    grid_options = parser.add_mutually_exclusive_group()
    grid_options.add_argument("--vary-motion-grid", action="store_true",
                              help="Compare motion spatial grids at the same training budget")
    grid_options.add_argument("--vary-intensity-grid", action="store_true",
                              help="Compare intensity maximum resolution with the same motion grid and budget")
    args = parser.parse_args()
    labels = args.labels or [run.name for run in args.runs]
    if len(labels) != len(args.runs) or len(set(labels)) != len(labels):
        parser.error("Provide one unique --labels value per run")
    args.out.mkdir(parents=True, exist_ok=True)
    configs = [json.loads((run / "config.json").read_text()) for run in args.runs]
    config = configs[0]
    validate_configs(configs, vary_motion_grid=args.vary_motion_grid,
                     vary_intensity_grid=args.vary_intensity_grid)
    torch.set_num_threads(config["threads"])
    data = make_pair(config["phantom"], config["size"], config["frames"], config["cycles"])
    traces, hashes = {}, {}
    time_resolutions = None
    motion_resolutions = None
    for run, label in zip(args.runs, labels):
        summary = read_report(run)
        entries = []
        for path in sorted((run / "checkpoints").glob("step_*.pt")):
            model, checkpoint = load_model(path)
            if args.vary_motion_grid:
                resolution = model.motion.field.encoding.resolutions
                temporal = resolution if resolution.ndim == 1 else resolution[:, 2]
                if time_resolutions is None:
                    time_resolutions = temporal
                elif not torch.equal(time_resolutions, temporal):
                    raise ValueError("Motion spatial grid comparison must preserve temporal resolution")
            if args.vary_intensity_grid:
                resolution = model.motion.field.encoding.resolutions
                if motion_resolutions is None:
                    motion_resolutions = resolution
                elif not torch.equal(motion_resolutions, resolution):
                    raise ValueError("Intensity grid comparison must preserve the entire motion grid")
            hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            if checkpoint["step"] == summary["config"]["steps"]:
                values = summary_scores(summary)
            else:
                print(f"Evaluating {run.name} step {checkpoint['step']} on full training/midpoint grids", flush=True)
                values = {split: scores(model, entry) for split, entry in data.items()}
            entries.append(dict(step=checkpoint["step"], scores=values))
        if not entries or entries[-1]["step"] != summary["config"]["steps"]:
            entries.append(dict(step=summary["config"]["steps"], scores=summary_scores(summary)))
        traces[label] = entries
        save_metrics(args.out / "checkpoint_metrics.json", traces)
        if args.gifs:
            from .render import main as render
            render([str(run), "--views", "motion"])
    common = {}
    if args.previous is not None:
        times = data["midpoint"]["t"]
        times = times[times < .5]
        shared = sample(config["phantom"], config["size"], times)
        for run in [args.previous, *args.runs]:
            model, checkpoint = load_model(run / "model.pt")
            trained_times = np.linspace(*checkpoint["config"]["coordinate_bounds"][2], checkpoint["config"]["frames"])
            if np.isclose(times[:, None], trained_times[None, :], rtol=0, atol=1e-10).any():
                raise ValueError("Common evaluation times overlap a model's training times")
            common[str(run.resolve())] = scores(model, shared)
            hashes[str((run / "model.pt").resolve())] = hashlib.sha256((run / "model.pt").read_bytes()).hexdigest()
    grid_branch = "motion" if args.vary_motion_grid else "intensity"
    vary_grid = args.vary_motion_grid or args.vary_intensity_grid
    note = (f"Same data, seed and {config['steps']:,}-step schedule; {grid_branch} spatial resolution differs"
            if vary_grid else None)
    plot_comparison(args.runs, traces, args.out, labels, note)
    report = dict(runs=[str(run.resolve()) for run in args.runs], checkpoints=traces,
                  common_first_cycle_midpoints=common,
                  common_time_count=len(times) if common else 0,
                  comparison_note=("Same training budget and temporal grid. Parameter shapes and RNG consumption "
                                   "differ; the same seed does not imply identical branch initialization or batches."
                                   if vary_grid else
                                   "Long run starts from seed 0, not an 8000-step resume; cosine horizons differ."),
                  labels=labels, comparison_type=f"{grid_branch}_spatial_grid" if vary_grid else "training_budget",
                  source_sha256=hashes,
                  analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_metrics(args.out / "comparison.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
