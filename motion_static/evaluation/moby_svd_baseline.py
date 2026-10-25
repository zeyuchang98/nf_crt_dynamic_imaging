"""Semiseparable baseline SS(r) for a dce_moby window: truncated SVD of the training frames.

The training frames form F (pixels x frames). Its rank-r truncated SVD (no centering)
gives r spatial maps u_i(r) and r temporal curves v_i(t) at the training times:
f(r, t) ~ sum_i u_i(r) v_i(t), with r (N + K) parameters for N pixels and K training
frames. Held-out (validation and test) frames are reconstructed by linear
interpolation of v_i(t) in time; they never enter the SVD. Metrics reuse the image
report of the neural-field runs, on the same alternate validation/test split.

python -m motion_static.evaluation.moby_svd_baseline motion_static/results/RUN --out motion_static/results/NEW
`RUN` is any dce_moby run; only its data settings are used.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluate import ALTERNATE_SPLITS, scalar_report
from .metrics import save_metrics
from .moby_gifs import git_commit
from .plots import pyplot
from ..phantoms import moby
from ..run import prepare_data


def semiseparable_factors(training):
    """Full SVD of the (pixels x training frames) matrix, largest singular values first."""
    frames = training["truth"].reshape(-1, training["truth"].shape[-1]).astype(np.float64)
    u, s, vt = np.linalg.svd(frames, full_matrices=False)
    return u * s, vt.T  # spatial maps scaled by singular values, temporal curves at training times


def reconstruct(spatial, temporal, training_times, times, rank, shape):
    """Rank-r image at `times`; temporal curves are linear in time between training frames."""
    if np.any(times < training_times[0]) or np.any(times > training_times[-1]):
        raise ValueError("Times outside the training frames would need extrapolation")
    curves = np.stack([np.interp(times, training_times, temporal[:, i]) for i in range(rank)], axis=1)
    return (spatial[:, :rank] @ curves.T).reshape(*shape[:2], len(times))


def parameter_count(pixels, training_frames, rank):
    return rank * (pixels + training_frames)


def neural_field_points(runs):
    """Test-frame scores of saved runs: their own alternate-split metrics or the re-scored file."""
    points = []
    for run in runs:
        for name in ("metrics.json", "metrics_alternate_split.json"):
            path = run / name
            if path.exists():
                report = json.loads(path.read_text())
                if "test_midpoints" in report["image"]:
                    test = report["image"]["test_midpoints"]
                    points.append(dict(run=run.name, source=name, parameters=report["parameters"],
                                       rrmse=test["rrmse"], psnr_db=test["psnr_db"], ssim=test["ssim"]))
                    break
        else:
            print(f"skip {run.name}: no test-frame metrics (run evaluation.moby_split_eval first)")
    return points


def plot(out, rows, points, labels):
    """Test RRMSE against parameters: full range (left) and the region near the floor (right)."""
    plt = pyplot(out)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    groups = {}
    for point in points:
        groups.setdefault(labels.get(point["run"], point["run"]), []).append(point)
    markers = ("o", "D", "s", "^", "v", "P")
    parameters = [row["parameters"] for row in rows]
    rrmse = [100 * row["test_rrmse"] for row in rows]
    for ax in axes:
        ax.plot(parameters, rrmse, color="#2a78d6", lw=2, marker="o", ms=3, label="SS(r), r = 1, 2, ...")
        for marker, (label, group) in zip(markers, groups.items()):
            ax.scatter([p["parameters"] for p in group], [100 * p["rrmse"] for p in group], color="#eb6834",
                       marker=marker, s=40, zorder=3, edgecolor="white", linewidth=1, label=f"NF: {label}")
        ax.set(xscale="log", xlabel="Parameters", ylabel="Test-frame image RRMSE (%)")
        ax.grid(alpha=.25)
    for rank in (1, 2, 3, 5, 8, 12, 20, 40):
        row = rows[rank - 1] if rank <= len(rows) else None
        if row is not None:
            for ax in axes:
                ax.annotate(f"r={rank}", (row["parameters"], 100 * row["test_rrmse"]), xytext=(3, 6),
                            textcoords="offset points", fontsize=8, color="#52514e", clip_on=True)
    axes[0].set_title("All ranks")
    axes[0].legend(frameon=False, fontsize=9)
    near = [p for p in points if p["rrmse"] < 0.15] + [row for row in rows if row["test_rrmse"] < 0.15]
    if near:
        low = min(r.get("rrmse", r.get("test_rrmse")) for r in near) * 100
        axes[1].set_ylim(low - 0.4, 15)
        axes[1].set_xlim(min(row["parameters"] for row in rows if row["test_rrmse"] < 0.15) / 1.3, parameters[-1] * 1.3)
    axes[1].set_title("Near the floor")
    fig.suptitle("DCE-MOBY z = 100 window: 37 test frames", fontsize=12)
    fig.savefig(out / "rrmse_vs_parameters.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="dce_moby run whose data settings define the window")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--max-rank", type=int, help="largest rank (default: number of training frames)")
    parser.add_argument("--full-reports", type=int, nargs="*", default=[],
                        help="ranks whose complete image report is also saved")
    parser.add_argument("--nf-runs", type=Path, nargs="*", default=[], help="neural-field runs to mark")
    parser.add_argument("--nf-labels", nargs="*", default=[], help="labels for --nf-runs, same order")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output already exists; choose a new --out")

    config = json.loads((args.run / "config.json").read_text())
    if config.get("phantom") != moby.NAME:
        parser.error("Run is not a dce_moby run")
    config = dict(config, held_out_split="alternate_validation_test")
    data, _ = prepare_data(config)
    training = data["training"]
    shape, training_times = training["truth"].shape, training["t"]
    pixels, frames = shape[0] * shape[1], shape[2]
    max_rank = args.max_rank or frames
    if not 1 <= max_rank <= frames:
        parser.error(f"max rank must be between 1 and {frames}")

    spatial, temporal = semiseparable_factors(training)
    args.out.mkdir(parents=True)
    truth = {split: entry["truth"] for split, entry in data.items()}
    rows = []
    for rank in range(1, max_rank + 1):
        prediction = {split: reconstruct(spatial, temporal, training_times, entry["t"], rank, shape)
                      for split, entry in data.items()}
        image = scalar_report(None, "image", data, truth, prediction, points={}, splits=ALTERNATE_SPLITS)
        row = dict(rank=rank, parameters=parameter_count(pixels, frames, rank))
        for split, label in ALTERNATE_SPLITS:
            for metric in ("rrmse", "psnr_db", "ssim"):
                row[f"{split}_{metric}"] = image[label][metric]
        rows.append(row)
        print(f"r={rank:3d} parameters={row['parameters']:9d} train {100 * row['training_rrmse']:6.2f}% "
              f"validation {100 * row['validation_rrmse']:6.2f}% test {100 * row['test_rrmse']:6.2f}%", flush=True)
        if rank in args.full_reports:
            save_metrics(args.out / f"metrics_r{rank}.json",
                         dict(schema_version=1, method=f"SS({rank})", parameters=row["parameters"], image=image))

    with (args.out / "svd_ranks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = dict(zip([run.name for run in args.nf_runs], args.nf_labels))
    points = neural_field_points(args.nf_runs)
    plot(args.out, rows, points, labels)
    summary = dict(
        method="semiseparable SS(r): truncated SVD of the training frames, no centering",
        held_out_frames="temporal curves linearly interpolated in time; validation and test frames never fitted",
        parameters="r * (pixels + training frames)", pixels=pixels, training_frames=frames,
        held_out_split=config["held_out_split"], data=config["data"], source_run=args.run.name,
        code_commit=git_commit(), singular_values=(np.linalg.norm(spatial, axis=0)).tolist(),
        ranks=rows, neural_field_points=points)
    save_metrics(args.out / "svd_baseline.json", summary)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
