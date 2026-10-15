"""Readable animations and motion diagnostics. All comparisons use fixed scales."""
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save_comparison(path, truth, prediction, times, title,
                    labels=("Ground truth", "Prediction", "Absolute error"), error_mask=None, save_still=True):
    error = np.abs(prediction - truth)
    selected_error = error if error_mask is None else error[error_mask]
    error_max = max(float(selected_error.max()), 1e-8)
    tile = 288
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    frames = []
    for k, time in enumerate(times):
        canvas = Image.new("RGB", (3 * tile, tile + 120), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 9), title, font=font, fill="black")
        for col, (label, volume) in enumerate(zip(labels, (truth, prediction, error))):
            scale = error_max if col == 2 else 1
            values = np.flipud(volume[..., k]) / scale
            pixels = np.round(np.clip(values, 0, 1) * 255).astype(np.uint8)
            pixels = np.repeat(pixels[..., None], 3, axis=-1)
            if col == 2 and error_mask is not None:
                pixels[~np.flipud(error_mask[..., k])] = 220
            image = Image.fromarray(pixels).resize((tile, tile), Image.Resampling.NEAREST)
            canvas.paste(image, (col * tile, 76))
            draw.text((col * tile + 10, 35), label, font=font, fill="black")
            draw.text((col * tile + 10, 56), f"Fixed display range: 0..{scale:.3g}", font=small, fill="black")
        note = "Display clipped to 0..1; metrics use raw predictions."
        if error_mask is not None:
            note = "Gray error pixels are training data; only held-out pixels are scored."
        draw.text((10, tile + 88), f"t={time:+.3f} | {note}", font=small, fill="black")
        frames.append(canvas)
    path = Path(path)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0)
    if not save_still:
        return
    # Still frame chosen by worst image MSE, useful for inspecting the GIF without playing it.
    if error_mask is None:
        frame_mse = np.mean(error ** 2, axis=(0, 1))
    else:
        counts = error_mask.sum(axis=(0, 1))
        frame_mse = np.divide(np.where(error_mask, error ** 2, 0).sum(axis=(0, 1)), counts,
                              out=np.full(len(times), np.nan), where=counts > 0)
    worst = int(np.nanargmax(frame_mse))
    frames[worst].save(path.with_suffix(".png"))


def pyplot(out):
    # Keep plotting caches out of run records. Respect an explicitly configured cache.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "motion_static_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_motion_error(out, times, scores, sample_label="Unseen times"):
    """Plot the existing foreground motion error at each evaluation time."""
    plt = pyplot(out)
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    ax.plot(times, scores["per_frame_foreground_endpoint_rmse_pixels"])
    ax.set(xlabel="Time", ylabel="Foreground endpoint RMSE (pixels)", title=f"Motion accuracy: {sample_label.lower()}")
    ax.grid(alpha=.2)
    fig.savefig(Path(out) / "motion_error.png", dpi=150)
    plt.close(fig)


def save_phantom(out, data, name):
    """Object and motion-free intensity side by side (third panel is their difference)."""
    save_comparison(Path(out) / "phantom.gif", data["truth"], data["reference_intensity"], data["t"],
                    title=f"{name} phantom: observed object and motion-free intensity",
                    labels=("Object f(r,t)", "Reference intensity C(q,t)", "Absolute difference"))
