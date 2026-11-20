"""Figures of the whole-series reconstructions: frames, signed errors and curves over time.

Reads only saved records (model.pt, metrics.json, loss.csv) and five truth frames from the
source h5; renders on the CPU (a run may be using the GPU) and prints each run's per-frame
RRMSE of these renders next to the saved score as a consistency check. Usage:
    python -m motion_static.evaluation.inverse_figures OUT_PREFIX RUN_DIR [RUN_DIR ...]
writes OUT_PREFIX_frames.png and OUT_PREFIX_curves.png. Run folders are named
inverse_full_<date>_<method>_<model>.
"""
import csv
import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from motion_static.run import build_model
from motion_static.phantoms.moby import pixel_centers

# Fixed entity colours (validated categorical slots 1-4); text stays in neutral ink.
STYLE = {"direct_pou": ("Direct POUnet (TV 3)", "#2a78d6"),
         "proximal_pou": ("Proximal POUnet (sq. grad. 1e4)", "#eb6834"),
         "proximal_nested": ("Proximal nested", "#1baf7a"),
         "direct_nested": ("Direct nested", "#eda100")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TIMES = (-18.0, 18.35, 46.0, 300.0, 600.0)  # pre-contrast, wash-in, activity peak, mid, late (s)
TIME_LABELS = ("pre-contrast", "wash-in", "activity peak", "mid", "late")


def run_key(run):
    """'direct_pou' etc. from inverse_full_<date>_<method>_<model>."""
    return re.sub(r"^inverse_full_\d{8}_", "", run.name)


def load(run):
    saved = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    config = saved["config"]
    model = build_model(config)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    metrics = json.load(open(run / "metrics.json"))["image"]["all_window_frames"]
    return config, model, metrics


@torch.no_grad()
def render(model, x, y, times):
    """(len(times), size, size) frames, (y, x) layout as the truth."""
    if hasattr(model, "frames"):  # POUnet: psi(t) C^T, flat pixels x-fastest
        images = model.frames(torch.as_tensor(times, dtype=torch.float32))
    else:
        yy, xx = np.meshgrid(y, x, indexing="ij")
        xy = torch.as_tensor(np.stack([xx.ravel(), yy.ravel()], 1), dtype=torch.float32)
        images = torch.stack([model(torch.cat([xy, torch.full((len(xy), 1), float(t))], 1)).reshape(-1)
                              for t in times])
    return images.reshape(len(times), len(y), len(x)).numpy()


def style_axis(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main(prefix, runs):
    runs = [Path(r) for r in runs]
    loaded = {run_key(r): load(r) for r in runs}
    config = next(iter(loaded.values()))[0]
    data = config["data"]
    all_times = np.array(next(iter(loaded.values()))[2]["times"])
    frames = [int(np.argmin(np.abs(all_times - t))) for t in TIMES]
    with h5py.File(Path(data["file"]).expanduser(), "r") as f:
        fov, spacing = np.asarray(f.attrs["fov"], float), np.asarray(f.attrs["spacing"], float)
        size = f["p0"].shape[0]
        first = data["first_frame"]
        truth = np.stack([f["p0"][:, :, first + k].T for k in frames]) / np.float32(data["intensity_scale_value"])
        times = f["time"][[first + k for k in frames]].astype(float)
    x, y = pixel_centers(fov[0], spacing[0], size), pixel_centers(fov[1], spacing[1], size)

    images = {key: render(model, x, y, times) for key, (_, model, _) in loaded.items()}
    # Consistency: per-frame RRMSE of these renders against the saved per-frame scores.
    for key, (_, _, metrics) in loaded.items():
        mine = np.sqrt(((images[key] - truth) ** 2).sum((1, 2)) / (truth ** 2).sum((1, 2)))
        saved = np.array(metrics["per_frame_rrmse"])[frames]
        print(key, "per-frame RRMSE", np.round(mine * 100, 2), "saved", np.round(saved * 100, 2),
              "max diff", float(np.abs(mine - saved).max()))

    # Figure 1: truth, reconstructions and signed errors at five times.
    keys = list(images)
    vmax = float(np.percentile(truth, 99.5))
    emax = float(np.percentile(np.abs(np.concatenate([images[k] - truth for k in keys])), 99.5))
    rows = 1 + 2 * len(keys)
    fig, axes = plt.subplots(rows, len(frames), figsize=(2.1 * len(frames) + 1.2, 2.1 * rows + 0.6))
    extent = [x[0], x[-1], y[0], y[-1]]
    for col, k in enumerate(frames):
        axes[0, col].set_title(f"{TIME_LABELS[col]}\nt = {times[col]:.0f} s (frame {k})", fontsize=8, color=INK)
        shown = axes[0, col].imshow(truth[col], cmap="gray", vmin=0, vmax=vmax, origin="lower", extent=extent)
        for i, key in enumerate(keys):
            error = images[key][col] - truth[col]
            rrmse = np.sqrt((error ** 2).sum() / (truth[col] ** 2).sum())
            axes[1 + i, col].imshow(images[key][col], cmap="gray", vmin=0, vmax=vmax, origin="lower", extent=extent)
            axes[1 + i, col].text(0.03, 0.04, f"{rrmse:.1%}", transform=axes[1 + i, col].transAxes,
                                  color="white", fontsize=7.5)
            err = axes[1 + len(keys) + i, col].imshow(error, cmap="RdBu_r", vmin=-emax, vmax=emax, origin="lower",
                                                      extent=extent)
    labels = ["Truth"] + [STYLE[k][0] for k in keys] + [f"Error: {STYLE[k][0]}" for k in keys]
    for row, ax_row in enumerate(axes):
        for ax in ax_row:
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax_row[0].set_ylabel(labels[row], fontsize=7.5, color=INK, rotation=90, labelpad=4, wrap=True)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.95, bottom=0.02, wspace=0.04, hspace=0.06)
    image_bar = fig.add_axes([0.9, 0.55, 0.015, 0.38])
    fig.colorbar(shown, cax=image_bar).set_label("normalized intensity", fontsize=8, color=MUTED)
    error_bar = fig.add_axes([0.9, 0.05, 0.015, 0.38])
    fig.colorbar(err, cax=error_bar).set_label("reconstruction - truth", fontsize=8, color=MUTED)
    for bar in (image_bar, error_bar):
        bar.tick_params(labelsize=7, colors=MUTED)
    fig.suptitle("Whole-series reconstructions (DCE-MOBY z = 100, 14,401 frames); "
                 "number = frame RRMSE", fontsize=9.5, color=INK)
    fig.savefig(f"{prefix}_frames.png", dpi=150)
    plt.close(fig)

    # Figure 2: per-frame RRMSE and the activity curve over time; monitor RRMSE over training.
    fig, (top, middle, bottom) = plt.subplots(3, 1, figsize=(9, 9.5), gridspec_kw=dict(hspace=0.45))
    window = 72  # one sensor revolution (3.6 s): every view seen once
    kernel = np.ones(window) / window
    t_smooth = np.convolve(all_times, kernel, mode="valid")
    for key, (_, _, metrics) in loaded.items():
        name, colour = STYLE[key]
        smooth = np.convolve(np.array(metrics["per_frame_rrmse"]) * 100, kernel, mode="valid")
        top.plot(t_smooth, smooth, color=colour, linewidth=1.6, label=name)
        top.text(t_smooth[-1] + 6, smooth[-1], f"{metrics['rrmse']:.1%}", color=INK, fontsize=7.5, va="center")
        activity = metrics["activity_curve"]
        middle.plot(all_times, activity["prediction"], color=colour, linewidth=1.4, label=name)
    middle.plot(all_times, metrics["activity_curve"]["truth"], color=INK, linewidth=1.4, linestyle="--", label="Truth")
    for ax in (top, middle):
        for t in times:
            ax.axvline(t, color=MUTED, linewidth=0.6, linestyle=":")
        style_axis(ax)
        ax.set_xlim(all_times[0], all_times[-1] + 40)
        ax.set_xlabel("time (s)", fontsize=8, color=MUTED)
    top.set_ylabel("frame RRMSE (%)\n(mean over 72 frames)", fontsize=8, color=MUTED)
    top.set_title("Error over the series (label: all-frame RRMSE; dotted lines: frames shown in the image figure)",
                  fontsize=9, color=INK, loc="left")
    middle.set_ylabel("mean intensity in mask", fontsize=8, color=MUTED)
    middle.set_title("Activity curve (fixed spatial mask)", fontsize=9, color=INK, loc="left")
    for key in loaded:
        name, colour = STYLE[key]
        rows = [r for r in csv.DictReader(open(Path([r for r in runs if run_key(r) == key][0]) / "loss.csv"))
                if r.get("data_fidelity")]
        progress_key = "step" if "step" in rows[0] else "outer"
        progress = np.array([float(r[progress_key]) for r in rows])
        bottom.plot(progress / progress[-1] * 100, [float(r["truth_rrmse"]) * 100 for r in rows], color=colour,
                    linewidth=1.6, marker="o", markersize=3, label=name)
    style_axis(bottom)
    bottom.set_xlabel("training progress (% of budget; 0 = warm start)", fontsize=8, color=MUTED)
    bottom.set_ylabel("truth RRMSE (%)\n(151 monitor frames)", fontsize=8, color=MUTED)
    bottom.set_title("Monitor during training", fontsize=9, color=INK, loc="left")
    handles, names = middle.get_legend_handles_labels()
    fig.subplots_adjust(top=0.93)
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=len(names), fontsize=8,
               frameon=False)
    fig.savefig(f"{prefix}_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{prefix}_frames.png", f"{prefix}_curves.png")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main(sys.argv[1], sys.argv[2:])
