"""Per-frame scores and the ROI-mean curve of a run's saved metrics.json, over time.

Reads only the saved report (no model, no data): the per-frame image RRMSE and SSIM of
one scored split and its `activity_curve`, the mean over the rectangular ROI of the truth
and of the NF in every frame (not the lesion activity curve of METRICS.md Section 3.6).
Curves are shown in raw p0 units (normalized values times the run's intensity scale).
Displays never enter metrics. Writes visuals/series_scores.png.

python -m motion_static.evaluation.series_score_plots motion_static/results/RUN
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .plots import pyplot

TRUTH_COLOR, NF_COLOR = "#222222", "#0072b2"  # near-black truth, blue NF (colorblind-safe pair)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--split", default="all_window_frames", help="scored split in metrics.json's image report")
    args = parser.parse_args(argv)

    config = json.loads((args.run / "config.json").read_text())
    report = json.loads((args.run / "metrics.json").read_text())["image"]
    if args.split not in report:
        parser.error(f"metrics.json has no image split {args.split!r}; it has {sorted(report)}")
    scores = report[args.split]
    scale = config["data"]["intensity_scale_value"]
    t = np.asarray(scores["times"], dtype=float)
    rrmse = 100 * np.asarray(scores["per_frame_rrmse"], dtype=float)
    ssim_gap = 1 - np.asarray(scores["per_frame_ssim"], dtype=float)
    curve = scores["activity_curve"]
    truth, nf = (scale * np.asarray(curve[key], dtype=float) for key in ("truth", "prediction"))

    out = args.run / "visuals"
    out.mkdir(exist_ok=True)
    plt = pyplot(out)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout="constrained")
    axes[0].plot(t, rrmse, color=NF_COLOR, lw=0.5)
    axes[0].set(yscale="log", ylabel="RRMSE (%)",
                title=f"Per-frame image RRMSE (all frames {100 * scores['rrmse']:.2f}%)")
    axes[1].plot(t, ssim_gap, color=NF_COLOR, lw=0.5)
    axes[1].set(yscale="log", ylabel="1 - SSIM",
                title=f"Per-frame spatial SSIM, shown as 1 - SSIM (whole array SSIM {scores['ssim']:.5f})")
    axes[2].plot(t, truth, color=TRUTH_COLOR, lw=0.6, label="Truth")
    axes[2].plot(t, nf, color=NF_COLOR, lw=0.6, label="NF")
    axes[2].legend(loc="lower right", frameon=False)
    roi = " x ".join(f"{a}-{b}" for a, b in scores["settings"]["roi"])
    axes[2].set(ylabel="ROI mean (raw p0)",
                title=f"Mean over the ROI (rows, columns, slices {roi}; curve RRMSE {100 * curve['rrmse']:.2f}%)")
    error = nf - truth
    axes[3].plot(t, error, color=NF_COLOR, lw=0.5)
    axes[3].axhline(0, color=TRUTH_COLOR, lw=0.6)
    # The first frames' large error would flatten the rest: scale the axis to the 0.1-99.9
    # percentiles and name the frames outside it instead of hiding them.
    low, high = np.percentile(error, [0.1, 99.9])
    margin = 0.1 * (high - low)
    low, high = low - margin, high + margin
    outside = np.flatnonzero((error < low) | (error > high))
    title = "Error of the ROI mean"
    if len(outside):
        worst = outside[np.argmax(np.abs(error[outside]))]
        title += (f" ({len(outside)} frames off scale, e.g. frames {outside[:5].tolist()}; "
                  f"largest {error[worst]:.2g} at frame {worst})")
    axes[3].set(ylim=(low, high), ylabel="NF - truth (raw p0)", xlabel="Time (s; contrast injection at 0)",
                title=title)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.axvline(0, color="#888888", lw=0.6, ls="--")
    fig.suptitle(f"{args.run.name}: {len(t)} frames, split {args.split}", fontsize=10)
    path = out / "series_scores.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
