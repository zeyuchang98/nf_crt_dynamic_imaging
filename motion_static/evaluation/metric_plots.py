"""Plot saved evaluation JSON and its source NPZ without retraining.

Run: python -m motion_static.evaluation.metric_plots REPORT.json --out-dir DIRECTORY
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .metrics import evaluate_sequence


def _numbers(values):
    """Unavailable and infinite JSON scores leave gaps, not invented values."""
    return np.array([float(v) if isinstance(v, (float, int)) else np.nan for v in values])


def adjacent_truth_baseline(report):
    """Average neighboring training images and score against true midpoint images."""
    evaluations = report["evaluations"]
    if report.get("quantity") != "image" or not all(
        name in evaluations for name in ("training_grid", "unseen_time_midpoints")
    ):
        return None
    training = evaluations["training_grid"]
    midpoint = evaluations["unseen_time_midpoints"]
    times = np.asarray(training["times"])
    expected_times = (times[:-1] + times[1:]) / 2
    if len(midpoint["times"]) != len(expected_times) or not np.allclose(
        midpoint["times"], expected_times, rtol=0, atol=1e-8
    ):
        raise ValueError("Adjacent-frame averaging requires the actual time midpoints")
    settings = midpoint["settings"]
    with np.load(report["source"], allow_pickle=False) as arrays:
        truth = np.moveaxis(arrays["truth"], training["settings"]["time_axis"], -1).astype(float)
        midpoint_truth = np.moveaxis(arrays["midpoint_truth"], settings["time_axis"], -1)
    prediction = (truth[..., :-1] + truth[..., 1:]) / 2
    roi = settings.get("roi")
    return evaluate_sequence(
        midpoint_truth, prediction, data_range=settings["data_range"],
        times=midpoint["times"], roi=None if roi is None else tuple(slice(*bounds) for bounds in roi),
        ssim_win_size=settings["ssim"]["win_size"])


def plot_evaluation(report, out_dir, *, framewise_only=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out_dir / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 170})
    evaluations = report["evaluations"]
    quantity = report.get("quantity", "Intensity")
    is_displacement = report.get("field_type") == "displacement"
    labels = {"training_grid": "Training times", "unseen_time_midpoints": "Unseen midpoints",
              "average_adjacent_truth_frames": "Average adjacent truth frames",
              "train_pixels": "Training pixels", "test_pixels": "Held-out pixels",
              "temporal_interpolation": "Temporal interpolation (train only)",
              "all_grid": "Full-grid diagnostic"}
    colors = {"training_grid": "#0072B2", "unseen_time_midpoints": "#D55E00",
              "average_adjacent_truth_frames": "#666666", "train_pixels": "#0072B2",
              "test_pixels": "#D55E00", "temporal_interpolation": "#666666", "all_grid": "#0072B2"}
    framewise_evaluations = dict(evaluations)
    baseline = adjacent_truth_baseline(report)
    if baseline is not None:
        framewise_evaluations["average_adjacent_truth_frames"] = baseline
    paths = []

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    columns = [("rrmse", "RRMSE (%) - lower is better", 100),
               ("ssim", "SSIM - higher is better", 1),
               ("psnr_db", "PSNR (dB) - higher is better", 1)]
    for row, scope in enumerate((None, "roi")):
        for col, (metric, ylabel, scale) in enumerate(columns):
            ax = axes[row, col]
            found = False
            for name, entry in framewise_evaluations.items():
                values = entry if scope is None else entry.get(scope)
                if values is None:
                    continue
                series = _numbers(values[f"per_frame_{metric}"]) * scale
                if metric == "rrmse" and "per_frame_rrmse_plot_valid" in values:
                    series[~np.asarray(values["per_frame_rrmse_plot_valid"], dtype=bool)] = np.nan
                found |= np.isfinite(series).any()
                ax.plot(entry["times"], series, color=colors.get(name),
                        label=labels.get(name, name), linewidth=1.5,
                        linestyle="--" if name in ("average_adjacent_truth_frames", "temporal_interpolation") else "-")
            ax.set(xlabel="Time (saved sample coordinates)", ylabel=ylabel,
                   title=("Full displacement field" if is_displacement else "Full image") if scope is None else "Spatial ROI")
            ax.grid(alpha=0.2)
            if metric == "rrmse":
                ax.set_ylim(bottom=0)
            if not found:
                ax.text(0.5, 0.5, "Not available", ha="center", transform=ax.transAxes)
    axes[0, 0].legend(fontsize=9)
    title = f"Framewise evaluation: {quantity}" if is_displacement else "Framewise evaluation: fitting and temporal interpolation"
    if is_displacement:
        title += "\nRRMSE omitted at numerically zero-motion frames; other scores retained"
    fig.suptitle(report.get("framewise_title", title), fontsize=14)
    path = out_dir / "framewise_metrics.png"
    fig.savefig(path)
    paths.append(path)
    plt.close(fig)
    if framewise_only or "test_pixels" in evaluations:
        return paths

    selected = "unseen_time_midpoints" if "unseen_time_midpoints" in evaluations else next(iter(evaluations))
    entry = evaluations[selected]
    curves = [(f"Fixed point: {name} {tuple(curve['spatial_index'])}", curve)
              for name, curve in entry.get("point_curves", {}).items()]
    if "activity_curve" in entry:
        mean_label = "ROI-mean displacement (not endpoint error)" if is_displacement else "Region-mean activity (not a motion score)"
        curves.append((mean_label, entry["activity_curve"]))
    if curves:
        fig, axes = plt.subplots(2, len(curves), figsize=(4.5 * len(curves), 6),
                                 squeeze=False, layout="constrained")
        for col, (title, curve) in enumerate(curves):
            times = entry["times"]
            truth, prediction = np.array(curve["truth"]), np.array(curve["prediction"])
            axes[0, col].plot(times, truth, color="#333333", label="Ground truth", linewidth=2)
            axes[0, col].plot(times, prediction, color=colors.get(selected, "#D55E00"),
                              linestyle="--", label="Prediction", linewidth=1.5)
            rrmse = curve["rrmse"]
            score = f"{100 * rrmse:.3f}%" if isinstance(rrmse, (int, float)) else str(rrmse)
            axes[0, col].set(title=f"{title}\nCurve RRMSE = {score}", ylabel=quantity)
            axes[0, col].legend(fontsize=9)
            axes[1, col].plot(times, prediction - truth, color=colors.get(selected, "#D55E00"))
            axes[1, col].axhline(0, color="gray", linewidth=0.7)
            axes[1, col].set(ylabel="Prediction - truth", xlabel="Time (saved sample coordinates)")
            for ax in axes[:, col]:
                ax.grid(alpha=0.2)
        curve_title = quantity if is_displacement else "intensity"
        fig.suptitle(f"Temporal {curve_title} curves: {labels.get(selected, selected)}", fontsize=14)
        path = out_dir / "temporal_curves.png"
        fig.savefig(path)
        paths.append(path)
        plt.close(fig)

    with np.load(report["source"], allow_pickle=False) as arrays:
        prefix = "midpoint_" if selected == "unseen_time_midpoints" else ""
        truth = np.moveaxis(arrays[prefix + "truth"], entry["settings"]["time_axis"], -1)
        prediction = np.moveaxis(arrays[prefix + "prediction"], entry["settings"]["time_axis"], -1)
    # For 3D+t data, show the central slice of the last spatial dimension.
    slice_note = ""
    if truth.ndim == 4:
        index = truth.shape[2] // 2
        truth, prediction = truth[:, :, index, :], prediction[:, :, index, :]
        slice_note = f"; spatial axis 2 slice {index}"
    worst = int(np.argmax(entry["per_frame_mse"]))
    frames = np.unique(np.append(np.linspace(0, truth.shape[-1] - 1, 4, dtype=int), worst))
    error = np.abs(prediction - truth)
    error_max = max(float(error.max()), 1e-12)
    # Shared truth-derived display scale; raw predictions remain unmodified.
    vmin = float(report.get("display_min", truth.min()))
    vmax = vmin + entry["settings"]["data_range"]
    fig, axes = plt.subplots(3, len(frames), figsize=(3 * len(frames), 8),
                             squeeze=False, layout="constrained")
    for col, k in enumerate(frames):
        for row, array in enumerate((truth, prediction, error)):
            im = axes[row, col].imshow(array[..., k], origin="lower", interpolation="nearest",
                cmap="magma" if row == 2 else report.get("cmap", "gray"), vmin=0 if row == 2 else vmin,
                vmax=error_max if row == 2 else vmax)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 1:
                intensity_im = im
        axes[0, col].set_title(f"t = {entry['times'][k]:+.3f}" + ("\nWorst full-frame MSE" if k == worst else ""))
        roi = entry["settings"].get("roi")
        if roi and col == 0 and not slice_note:
            (y0, y1), (x0, x1) = roi
            axes[0, col].add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                  fill=False, edgecolor="#56B4E9", linewidth=1.2))
            for name, curve in entry.get("point_curves", {}).items():
                y, x = curve["spatial_index"]
                axes[0, col].plot(x, y, "+", color="#E69F00")
                axes[0, col].text(x + 2, y, name, color="#E69F00", fontsize=8)
    for row, label in enumerate(("Ground truth", "Prediction", "Absolute error")):
        axes[row, 0].set_ylabel(label)
    fig.colorbar(intensity_im, ax=axes[:2, :].ravel().tolist(), shrink=0.65, label=quantity)
    fig.colorbar(im, ax=axes[2, :].tolist(), shrink=0.8, label="Absolute error")
    field_label = quantity if is_displacement else "images"
    fig.suptitle(f"{labels.get(selected, selected)}: {field_label} and error{slice_note}\n"
                 "Shared value scale; separate error scale. Cyan box: ROI; crosses: sampled points.", fontsize=13)
    path = out_dir / "prediction_errors.png"
    fig.savefig(path)
    paths.append(path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in plot_evaluation(json.loads(args.report.read_text()), args.out_dir):
        print(path.resolve())


if __name__ == "__main__":
    main()
