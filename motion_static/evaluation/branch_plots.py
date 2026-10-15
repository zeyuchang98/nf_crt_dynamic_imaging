"""Plots for independent intensity and motion diagnostics; all inputs are explicit."""
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .branch_metrics import relative_geometry, true_geometry
from .plots import pyplot, save_comparison
from .motion_visuals import frozen_true_motion

BLUE, ORANGE, GRAY = "#0072b2", "#d55e00", "#777777"


def save_branch_plots(out, data, prediction, fit, stationary, motion, phantom, model, config):
    """Same artifact names for every nested model and supported toy phantom."""
    intensity_animation(out / "intensity_residual.gif", data, prediction["reference_intensity"], fit["parameters"])
    geometry_plot(out / "intensity_geometry.png", data, fit, stationary, phantom, model)
    save_comparison(out / "geometry_fit.gif", prediction["reference_intensity"], fit["images"], data["t"],
                    title=f"{model}: check the geometry estimator fit; not a model prediction",
                    labels=("Intensity network output", "Geometry + brightness fit", "Fit error"))
    true_effect, predicted_effect = frozen_true_motion(data, prediction["motion"], config)
    save_comparison(out / "motion_effect.gif", true_effect, predicted_effect, data["t"],
                    title="True and predicted motion | True intensity fixed at t0",
                    labels=("True motion", "Predicted motion", "Absolute image error"))
    motion_plot(out / "motion_outputs.png", data, prediction["motion"], motion, model)


def save_animation(path, frames, still_index):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    frames[still_index].save(path.with_suffix(".png"))


def intensity_animation(path, data, volume, parameters):
    """Compare observed truth, intensity alone, and motion-free true intensity."""
    font, small = ImageFont.load_default(size=17), ImageFont.load_default(size=14)
    spacing = np.array([data[axis][1] - data[axis][0] for axis in ("x", "y")])
    shift, scale = relative_geometry(parameters, spacing)
    frames, tile = [], 288
    volumes = (data["truth"], volume, data["reference_intensity"])
    labels = ("Observed object (truth)", "Intensity only: C(q,t)", "True reference: C_true(q,t)")
    for k, time in enumerate(data["t"]):
        canvas = Image.new("RGB", (3 * tile, tile + 130), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), "Does the intensity branch still move? Motion bypassed; unseen midpoints.", font=font, fill="black")
        for col, (array, label) in enumerate(zip(volumes, labels)):
            pixels = np.uint8(np.round(np.clip(np.flipud(array[..., k]), 0, 1) * 255))
            image = Image.fromarray(pixels).convert("RGB").resize((tile, tile), Image.Resampling.NEAREST)
            canvas.paste(image, (tile * col, 65))
            draw.text((10 + tile * col, 37), label, font=small, fill="black")
        draw.text((10, tile + 78),
                  f"t={time:+.3f} | Estimated shift=({shift[k, 0]:+.2f}, {shift[k, 1]:+.2f}) px; "
                  f"size=({scale[k, 0]:.2f}, {scale[k, 1]:.2f}) x", font=small, fill="black")
        draw.text((10, tile + 101), "Fixed display 0..1. Geometry fit allows brightness to vary independently.", font=small, fill="black")
        frames.append(canvas)
    save_animation(path, frames, int(np.argmax(np.linalg.norm(shift, axis=1))))


def geometry_plot(path, data, fit, stationary, phantom, model):
    plt = pyplot(path.parent)
    times = data["t"]
    spacing = np.array([data[axis][1] - data[axis][0] for axis in ("x", "y")])
    translation, scale = true_geometry(phantom, times)
    true_parameters = np.column_stack((translation, scale))
    series = [relative_geometry(p, spacing) for p in (true_parameters, fit["parameters"], stationary["parameters"])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    labels = ("Observed object: true motion", "Intensity branch: fitted geometry", "True reference: estimator control")
    for (shift, size), color, label in zip(series, (GRAY, ORANGE, BLUE), labels):
        for i in range(2):
            style = "--" if color == GRAY else "-"
            axes[0, i].plot(times, shift[:, i], color=color, linestyle=style, label=label)
            axes[1, i].plot(times, size[:, i], color=color, linestyle=style)
    for i in range(2):
        axes[0, i].set(xlabel="Time", ylabel=f"{'xy'[i]} shift from first midpoint (pixels)")
        axes[1, i].set(xlabel="Time", ylabel=f"{'xy'[i]} size relative to first midpoint")
    axes[0, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle(f"{model}: intensity geometry at unseen midpoints\n"
                 f"Known-shape fit with independent brightness gains; median fit RRMSE {np.median(fit['rrmse']):.1%}")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def motion_plot(path, data, motion, metrics, model):
    plt = pyplot(path.parent)
    times = data["t"]
    x, y = data["x"], data["y"]
    point = (len(y) // 2, len(x) // 2)
    spacing = (x[1] - x[0], y[1] - y[0])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for i, ax in enumerate(axes[0]):
        ax.plot(times, data["u_true"][point][:, i] / spacing[i], color=BLUE, label="True displacement")
        ax.plot(times, motion[point][:, i] / spacing[i], color=ORANGE, label="Motion network")
        ax.set(ylabel=f"u{'xy'[i]} at fixed position (pixels)", xlabel="Time")
    axes[0, 0].legend()
    axes[1, 0].plot(times, metrics["per_frame_full_foreground_zero_baseline_pixels"], color=BLUE, label="True motion RMS")
    axes[1, 0].plot(times, metrics["per_frame_full_foreground_predicted_rms_pixels"], color=ORANGE, label="Predicted motion RMS")
    axes[1, 0].set(ylabel="Displacement magnitude (pixels)", xlabel="Time")
    axes[1, 0].legend()
    axes[1, 1].plot(times, metrics["per_frame_full_foreground_endpoint_rmse_pixels"], color=ORANGE, label="Prediction error")
    axes[1, 1].plot(times, metrics["per_frame_full_foreground_zero_baseline_pixels"], color=GRAY, linestyle="--", label="Zero-displacement error")
    axes[1, 1].set(ylabel="Endpoint RMSE (pixels)", xlabel="Time")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle(f"{model}: what the motion network actually outputs\n"
                 f"Top: fixed (x,y)=({x[point[1]]:.5f},{y[point[0]]:.5f}). "
                 f"Bottom: true foreground, {len(times)} unseen midpoints.")
    fig.savefig(path, dpi=160)
    plt.close(fig)
