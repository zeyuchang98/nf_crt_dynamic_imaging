"""Show the held-out blocks and distinguish scored samples in temporal curves."""
from pathlib import Path

import numpy as np

from .metric_plots import _numbers
from .plots import pyplot


def save_split_plot(out, data, split):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt = pyplot(out)
    from matplotlib.colors import ListedColormap
    test = split["test_mask"]
    # Show the frame with the most foreground test pixels, independent of a model.
    k = int(np.argmax((test & data["foreground_mask"]).sum(axis=(0, 1))))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    axes[0].imshow(data["truth"][..., k], origin="lower", cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*test.shape[:2], 4))
    overlay[test[..., k]] = (0.90, 0.40, 0.05, 0.4)
    axes[0].imshow(overlay, origin="lower")
    axes[0].set(title=f"Orange: held-out pixels at t={data['t'][k]:+.3f}", xlabel="x pixel", ylabel="y pixel")
    blocks = split["test_blocks"].reshape(-1, split["test_blocks"].shape[-1])
    axes[1].imshow(blocks, aspect="auto", origin="lower", interpolation="nearest",
                   cmap=ListedColormap(["#dddddd", "#d55e00"]), vmin=0, vmax=1)
    axes[1].set(title="Each row is one spatial tile", xlabel="Time block", ylabel="Spatial tile")
    axes[2].plot(data["t"], test.mean(axis=(0, 1)), color="#d55e00")
    axes[2].set(title="Held-out fraction per frame", xlabel="Time", ylabel="Fraction of pixels", ylim=(0, .5))
    axes[2].grid(alpha=.2)
    settings = split["settings"]
    fig.suptitle(f"Block split {settings['block_shape_yxt']} (y, x, t): "
                 f"{settings['actual_test_fraction']:.2%} held out; seed {settings['seed']}")
    fig.savefig(out / "split.png", dpi=170)
    plt.close(fig)


def save_test_curves(out, test, baseline):
    """Lines show the full time series; markers and scores identify test samples."""
    out = Path(out)
    plt = pyplot(out)
    curves = [(f"Fixed point: {name}", curve, baseline["point_curves"][name])
              for name, curve in test["point_curves"].items()]
    curves.append(("Mean over held-out ROI pixels\nSpatial subset changes with time",
                   test["activity_curve"], baseline["activity_curve"]))
    fig, axes = plt.subplots(2, len(curves), figsize=(4.8 * len(curves), 6), layout="constrained")
    times = np.asarray(test["times"])
    for col, (title, curve, reference) in enumerate(curves):
        truth, prediction = _numbers(curve["truth"]), _numbers(curve["prediction"])
        interpolated = _numbers(reference["prediction"])
        selected = np.asarray(curve["evaluated"], dtype=bool)
        axes[0, col].plot(times, truth, color="#333333", label="Ground truth")
        axes[0, col].plot(times, prediction, color="#d55e00", label="Model")
        axes[0, col].plot(times, interpolated, color="#777777", linestyle="--", label="Train-only interpolation")
        axes[0, col].scatter(times[selected], prediction[selected], color="#d55e00", s=12, label="Scored samples")
        score = curve["rrmse"]
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) and np.isfinite(score) else "undefined"
        axes[0, col].set(title=f"{title}\nTest curve RRMSE: {score_text}", ylabel="Intensity")
        axes[0, col].legend(fontsize=8)
        axes[1, col].plot(times[selected], (prediction - truth)[selected], ".-", color="#d55e00", label="Model")
        axes[1, col].plot(times[selected], (interpolated - truth)[selected], ".--", color="#777777", label="Interpolation")
        axes[1, col].axhline(0, color="gray", linewidth=.7)
        axes[1, col].set(xlabel="Time", ylabel="Error at scored samples")
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
    fig.suptitle("Temporal curves: scores use held-out samples only", fontsize=14)
    fig.savefig(out / "temporal_curves.png", dpi=170)
    plt.close(fig)
