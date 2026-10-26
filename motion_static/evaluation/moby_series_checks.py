"""Data-only checks on a whole DCE-MOBY slice series, before choosing the time model.

No model is trained. Three checks, all on raw p0:

1. Tissue mean curves. For frame k, the partial-volume weighted mean of p0 in each
   indicator tissue uses the indicator of that frame's respiratory phase, so the mask
   follows the motion. The mean over each breath (consecutive blocks of P frames, one
   breathing period) is the slow kinetic trend; what remains is breathing modulation
   that the moving mask does not remove. First-pass timing is read from the trend.
2. Time resolution. Each curve and trend is fitted by least squares with a piecewise
   linear function on N equal cells over the series, the form of one TimeGridEncoding
   level. The error against N shows how fine the intensity time grid must be for the
   kinetics, and at which N it starts to follow breathing.
3. Same-respiratory-phase SVD. Frames of one respiratory phase come from one source
   file (every 200 frames, 10 s apart) and share exactly one geometry, so the uncentered
   SVD of these frames (pixels x frames) measures the rank of the intensity change
   alone, as SS(r) would see it without motion.

Breathing periodicity is also measured: before injection, phase p and phase p + P
(P = 50) should show the same geometry if every breath were identical.

python -m motion_static.evaluation.moby_series_checks ~/Data/dce-moby/moby_z100.h5 --out motion_static/results/NEW
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.linalg import solve_banded

from .metrics import save_metrics
from .moby_gifs import git_commit
from .plots import pyplot
from ..phantoms.moby import file_sha256

TISSUES = ("artery", "liver", "tumor", "tumor_core")
DEFAULT_CELLS = (36, 72, 144, 288, 576, 1152, 2304, 4608)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")


def tissue_curves(p0, phase, weights, block=400):
    """Weighted mean of p0 per frame, with each frame's own phase indicator.

    p0: (x, y, frame) dataset or array; phase: (frame,) indices into the last axis of
    every weights[name] (x, y, phase). Frames are read in blocks; a tissue with no
    weight in any phase is left out. Also returns the plain image mean.
    """
    present = {name: w for name, w in weights.items() if w.sum() > 0}
    totals = {name: w.sum(axis=(0, 1)) for name, w in present.items()}
    curves = {name: np.empty(p0.shape[-1]) for name in (*present, "whole_image")}
    for start in range(0, p0.shape[-1], block):
        values = np.asarray(p0[..., start:start + block], dtype=np.float64)
        frames = phase[start:start + values.shape[-1]]
        for name, w in present.items():
            curves[name][start:start + values.shape[-1]] = (
                np.einsum("xyk,xyk->k", values, w[..., frames]) / totals[name][frames])
        curves["whole_image"][start:start + values.shape[-1]] = values.mean(axis=(0, 1))
    return curves


def breath_trend(times, curve, period):
    """Mean over consecutive breaths, interpolated back to every frame.

    Block b covers frames [b P, (b + 1) P); a final incomplete block is dropped. The
    trend is the block mean placed at the block's mean time.
    """
    blocks = len(curve) // period
    centers = times[:blocks * period].reshape(blocks, period).mean(axis=1)
    means = curve[:blocks * period].reshape(blocks, period).mean(axis=1)
    return np.interp(times, centers, means)


def piecewise_linear_fit(times, values, cells):
    """Least-squares fit by linear interpolation between cells + 1 equal nodes.

    The nodes span [times[0], times[-1]], like a time-grid level over the unit time.
    The normal equations are tridiagonal (hat functions overlap their neighbours only).
    """
    position = (times - times[0]) / (times[-1] - times[0]) * cells
    lower = np.minimum(np.floor(position).astype(int), cells - 1)
    right = position - lower
    left = 1 - right
    nodes = cells + 1
    diagonal = (np.bincount(lower, left * left, nodes) + np.bincount(lower + 1, right * right, nodes))
    off = np.bincount(lower, left * right, nodes)[:-1]
    rhs = np.bincount(lower, left * values, nodes) + np.bincount(lower + 1, right * values, nodes)
    banded = np.zeros((3, nodes))
    banded[0, 1:], banded[1], banded[2, :-1] = off, diagonal, off
    # Nodes with no sample nearby would make the system singular; tie them weakly to 0.
    banded[1] += 1e-12 * max(diagonal.max(), 1)
    node_values = solve_banded((1, 1), banded, rhs)
    return left * node_values[lower] + right * node_values[lower + 1]


def resolution_errors(times, curve, trend, cell_counts):
    """RMS error of the piecewise-linear fit, relative to the trend's range."""
    scale = trend.max() - trend.min()
    rows = []
    for cells in cell_counts:
        row = dict(cells=int(cells), seconds_per_cell=float((times[-1] - times[0]) / cells))
        for label, target in (("curve", curve), ("trend", trend)):
            error = piecewise_linear_fit(times, target, cells) - target
            row[f"{label}_relative_rms"] = float(np.sqrt(np.mean(error ** 2)) / scale)
        rows.append(row)
    return rows


def periodicity(frames, phase_count, period):
    """Relative difference ||f(p) - f(p + P)|| / ||f(p)|| for every phase p.

    `frames` (x, y, phase_count) holds one frame per respiratory phase before
    injection, so only geometry differs. Consecutive phases give the scale of motion.
    """
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(a))
    shifted = [rel(frames[..., p], frames[..., p + period]) for p in range(phase_count - period)]
    consecutive = [rel(frames[..., p], frames[..., p + 1]) for p in range(phase_count - 1)]
    return np.array(shifted), np.array(consecutive)


def first_pass(times, curve, baseline_mask):
    """Baseline, peak and the width of the peak above the baseline (breath-mean trend)."""
    baseline = float(curve[baseline_mask].mean())
    peak = int(np.argmax(curve))
    excess = curve - baseline
    half = excess >= 0.5 * excess[peak]
    # Contiguous run of frames above half the peak excess, around the peak.
    left = peak - np.argmin(half[peak::-1]) + 1 if not half[:peak + 1].all() else 0
    right = peak + np.argmin(half[peak:]) - 1 if not half[peak:].all() else len(curve) - 1
    rise = excess[:peak + 1]
    t10 = times[np.argmax(rise >= 0.1 * excess[peak])]
    t90 = times[np.argmax(rise >= 0.9 * excess[peak])]
    return dict(baseline=baseline, peak_value=float(curve[peak]), peak_time_s=float(times[peak]),
                peak_relative_to_baseline=float(excess[peak] / baseline) if baseline else None,
                rise_10_90_s=float(t90 - t10), half_maximum_width_s=float(times[right] - times[left]),
                final_value=float(curve[-1]))


def svd_summary(frames):
    """Uncentered SVD of (pixels x frames); errors of the best rank-r approximation.

    By Eckart-Young the squared error of rank r is the tail sum of squared singular
    values. `change` errors are relative to the change from the series' first frame.
    """
    matrix = frames.reshape(-1, frames.shape[-1]).astype(np.float64)
    singular = np.linalg.svd(matrix, compute_uv=False)
    tail = np.concatenate([np.cumsum((singular ** 2)[::-1])[::-1][1:], [0.]])
    total = np.sqrt(tail / (singular ** 2).sum())
    change_norm = np.linalg.norm(matrix - matrix[:, :1])
    change = np.sqrt(tail) / change_norm
    return singular, total, change


def first_rank_below(errors, threshold):
    """Smallest rank r >= 1 with error below threshold (errors[r - 1] is rank r)."""
    below = np.flatnonzero(errors < threshold)
    return int(below[0] + 1) if len(below) else None


def plot_curves(out, times, curves, trends, zoom):
    plt = pyplot(out)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    for ax, limits in zip(axes, (None, zoom)):
        for color, (name, curve) in zip(COLORS, curves.items()):
            ax.plot(times, curve, color=color, lw=0.6, alpha=0.5)
            ax.plot(times, trends[name], color=color, lw=2, label=f"{name} (breath mean)")
        ax.axvline(0, color="#52514e", lw=1, ls=":")
        ax.set(xlabel="Time (s, injection at 0)", ylabel="Mean raw p0", yscale="log")
        ax.grid(alpha=.25)
        if limits is not None:
            ax.set_xlim(*limits)
            ax.set_title(f"Zoom {limits[0]:g} to {limits[1]:g} s; thin lines: every frame", fontsize=10)
        else:
            ax.set_title("Whole series; thin lines: every frame (breathing modulation)", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(out / "tissue_curves.png", dpi=160)
    plt.close(fig)


def plot_resolution(out, resolution, breaths):
    plt = pyplot(out)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained", sharey=True)
    # The trend is linear between breath means, so from 2 cells per breath it is fitted by construction.
    titles = ("Breath-mean trend (kinetics; linear between breath means)", "Every frame (with breathing)")
    for ax, label, title in zip(axes, ("trend", "curve"), titles):
        for color, (name, rows) in zip(COLORS, resolution.items()):
            ax.plot([r["cells"] for r in rows], [100 * r[f"{label}_relative_rms"] for r in rows],
                    color=color, lw=2, marker="o", ms=4, label=name)
        for per_breath, text, height in ((1, "1 cell / breath", 0.97), (2, "2 cells / breath", 0.90)):
            ax.axvline(per_breath * breaths, color="#52514e", lw=1, ls=":")
            ax.annotate(text, (per_breath * breaths, height), xycoords=("data", "axes fraction"), xytext=(3, -8),
                        textcoords="offset points", fontsize=8, color="#52514e")
        ax.set(xscale="log", yscale="log", xlabel="Cells over the series (finest time-grid level)", title=title)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("RMS fit error / trend range (%)")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(out / "time_resolution.png", dpi=160)
    plt.close(fig)


def plot_svd(out, svd, ranks_shown=30):
    plt = pyplot(out)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for color, (phase, entry) in zip(COLORS, svd.items()):
        shown = min(ranks_shown, len(entry["singular"]))
        r = np.arange(1, shown + 1)
        axes[0].plot(r, entry["singular"][:shown] / entry["singular"][0], color=color, lw=2, marker="o", ms=3,
                     label=f"respiratory phase {phase}")
        axes[1].plot(r, 100 * entry["change"][:shown], color=color, lw=2, marker="o", ms=3,
                     label=f"respiratory phase {phase}")
    axes[0].set(yscale="log", xlabel="Index i", ylabel="sigma_i / sigma_1", title="Singular values, one respiratory phase")
    axes[1].set(yscale="log", xlabel="Rank r", ylabel="Rank-r error / change from first frame (%)",
                title="Unexplained intensity change")
    for ax in axes:
        ax.grid(alpha=.25)
    axes[1].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(out / "same_phase_svd.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slice_file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--breath-frames", type=int, default=50,
                        help="frames per breath (motion repeats every 50 frames at z = 100)")
    parser.add_argument("--phases", type=int, nargs="+", default=[0, 10, 20, 30, 40],
                        help="respiratory phases (source files) for the same-phase SVD")
    parser.add_argument("--cells", type=int, nargs="+", default=list(DEFAULT_CELLS))
    parser.add_argument("--zoom", type=float, nargs=2, default=[-10, 60], help="time range of the zoomed curve panel")
    args = parser.parse_args(argv)
    period = args.breath_frames
    args.out.mkdir(parents=True, exist_ok=False)

    with h5py.File(args.slice_file, "r") as f:
        times = f["time"][:].astype(np.float64)
        resp_phase = f["resp_phase"][:]
        weights = {name: f["indicator"][name][:].astype(np.float64) for name in TISSUES}
        p0 = f["p0"]
        frame_count = p0.shape[-1]

        # Periodicity: one pre-injection frame per respiratory phase (the first cycle).
        phase_count = weights[TISSUES[0]].shape[-1]
        first_cycle = np.flatnonzero(times < 0)[:phase_count]
        if not np.array_equal(resp_phase[first_cycle], np.arange(phase_count)):
            raise ValueError("Expected every respiratory phase once, in order, before injection")
        shifted, consecutive = periodicity(np.asarray(p0[..., first_cycle], dtype=np.float64), phase_count, period)
        indicator_shift = {name: float(np.abs(w[..., :phase_count - period] - w[..., period:]).sum()
                                       / max(w[..., :phase_count - period].sum(), 1e-12))
                           for name, w in weights.items() if w.sum() > 0}

        curves = tissue_curves(p0, resp_phase, weights)
        selected = {phase: np.flatnonzero(resp_phase == phase) for phase in args.phases}
        same_phase = {phase: np.asarray(p0[..., frames]) for phase, frames in selected.items()}
    absent = [name for name in TISSUES if name not in curves]

    # Before injection the frames of one respiratory phase must be identical (no contrast yet).
    before = times < 0
    pre_injection_repeat = {}
    for phase, frames in selected.items():
        stack = same_phase[phase][..., before[frames]]
        pre_injection_repeat[phase] = float(np.abs(stack - stack[..., :1]).max())

    trends = {name: breath_trend(times, curve, period) for name, curve in curves.items()}
    breaths = frame_count // period
    resolution = {name: resolution_errors(times, curves[name], trends[name], args.cells) for name in curves}
    tissues = {}
    for name, curve in curves.items():
        entry = first_pass(times, trends[name], before)
        trend_range = trends[name].max() - trends[name].min()
        # Breathing modulation before injection, where the kinetics are flat.
        entry["pre_injection_peak_to_peak_relative_to_trend_range"] = float(np.ptp(curve[before]) / trend_range)
        entry["modulation_rms_relative_to_trend_range"] = float(np.sqrt(np.mean((curve - trends[name]) ** 2)) / trend_range)
        entry["time_resolution"] = resolution[name]
        tissues[name] = entry

    svd = {}
    for phase, frames in same_phase.items():
        singular, total, change = svd_summary(frames)
        svd[phase] = dict(singular=singular, total=total, change=change)
    svd_report = {
        str(phase): dict(
            frames=len(selected[phase]), first_frame=int(selected[phase][0]),
            singular_values=entry["singular"].tolist(),
            rrmse_by_rank=entry["total"].tolist(), change_error_by_rank=entry["change"].tolist(),
            rank_for_rrmse_below={f"{t:g}": first_rank_below(entry["total"], t) for t in (0.01, 0.005, 0.001)},
            rank_for_change_error_below={f"{t:g}": first_rank_below(entry["change"], t) for t in (0.1, 0.05, 0.01)})
        for phase, entry in svd.items()}

    report = dict(
        schema_version=1, check="dce_moby whole-series data checks (no model)",
        source=dict(file=str(args.slice_file.expanduser().resolve()), sha256=file_sha256(args.slice_file),
                    frames=frame_count, time_range_s=[float(times[0]), float(times[-1])]),
        code_commit=git_commit(), command=" ".join(sys.argv),
        breath_frames=period, breaths=breaths, absent_tissues=absent,
        geometry=dict(
            pre_injection_max_difference_same_phase_frames={str(k): v for k, v in pre_injection_repeat.items()},
            breath_periodicity=dict(
                note="pre-injection, ||f(p) - f(p + P)|| / ||f(p)|| per respiratory phase p",
                median=float(np.median(shifted)), maximum=float(shifted.max()),
                phases_above_1_percent=np.flatnonzero(shifted > 0.01).tolist(),
                per_phase=shifted.tolist(),
                consecutive_phase_median=float(np.median(consecutive)),
                indicator_relative_l1_phase_p_vs_p_plus_breath=indicator_shift)),
        tissue_curves=tissues, same_phase_svd=svd_report)
    save_metrics(args.out / "series_checks.json", report)

    with (args.out / "tissue_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "resp_phase", *curves])
        writer.writerows(zip(times.tolist(), resp_phase.tolist(), *(c.tolist() for c in curves.values())))

    plot_curves(args.out, times, curves, trends, args.zoom)
    plot_resolution(args.out, resolution, breaths)
    plot_svd(args.out, svd)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
