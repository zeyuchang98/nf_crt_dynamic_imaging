"""Image and branch GIFs for a saved dce_moby run, over every frame of the window.

There is no true motion or reference intensity for these data, so the branch views
show model outputs only; the observed truth appears only in the image view. Frames
are all window frames in time order (training, then held-out frames between them).

- image: truth | prediction | |prediction - truth|
- intensity: J(q, t) at fixed pixel coordinates q, no motion applied (the project's
  reference_intensity view) | J(q, t) - J(q, t0)
- motion: J(r + d(r, t), t0), the learned intensity frozen at the motion reference
  time t0 and moved by the learned motion (the motion_only view) | |d(r, t)| in mm

If the branches separate as intended, the intensity view changes only with contrast
and the motion view only with breathing. Images use raw p0 units (normalized values
times the run's intensity scale) on fixed display ranges; displays never enter metrics.
`branch_gifs.json` also stores, per frame, ||V(t) - V(t0)|| / ||V(t0)|| for each view.

python -m motion_static.evaluation.moby_branch_gifs motion_static/results/RUN
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

from .evaluate import predict
from .moby_gifs import FRAME_DURATION_MS, git_commit
from ..phantoms import moby
from ..phantoms.data import coordinate_grid
from ..run import build_model, prepare_data


DIVERGING = colormaps["RdBu_r"]
MAGNITUDE = colormaps["magma"]


def window_frames(data):
    """All splits of the window merged in time order: truth (y, x, t), times and split labels."""
    names = sorted(data, key=lambda name: data[name]["t"][0])
    times = np.concatenate([data[name]["t"] for name in names])
    order = np.argsort(times)
    truth = np.concatenate([data[name]["truth"] for name in names], axis=-1)[..., order]
    labels = np.concatenate([[name] * len(data[name]["t"]) for name in names])[order]
    return truth, times[order], labels


@torch.no_grad()
def branch_fields(model, x, y, t, reference_time):
    """Prediction, J(q, t), J(r + d, t0) and displacement on the (y, x, t) grid."""
    coordinates = coordinate_grid(x, y, t)
    shape = (len(y), len(x), len(t))
    image = predict(model, coordinates).reshape(shape)
    intensity = predict(model.intensity, coordinates).reshape(shape)
    displacement = predict(model.motion, coordinates).reshape(*shape, 2)
    frozen = coordinates.copy()
    frozen[:, 2] = reference_time
    frozen[:, :2] += displacement.reshape(-1, 2)
    motion_only = predict(model.intensity, frozen).reshape(shape)
    return image, intensity, motion_only, displacement


def to_rgb(panel, cmap, low, high):
    """(y, x) values to an RGB tile, x to the right and y upward like the other GIFs."""
    scaled = np.clip((np.flipud(panel) - low) / (high - low), 0, 1)
    if cmap is None:
        gray = np.round(scaled * 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=-1)
    return np.round(cmap(scaled)[..., :3] * 255).astype(np.uint8)


def shared_palette():
    """One fixed GIF palette: a gray ramp plus both colormaps, so frames need no dithering."""
    ramp = np.linspace(0, 1, 84)
    colors = np.concatenate([np.repeat(np.linspace(0, 1, 88)[:, None], 3, axis=1),
                             DIVERGING(ramp)[:, :3], MAGNITUDE(ramp)[:, :3]])
    image = Image.fromarray(np.round(colors * 255).astype(np.uint8)[None])
    return image.quantize(colors=256, dither=Image.Dither.NONE)


def save_view(path, panels, captions, title, scale):
    """panels: list of (volume (y, x, t), colormap or None, low, high, label)."""
    font, small = ImageFont.load_default(size=16), ImageFont.load_default(size=13)
    height, width = panels[0][0].shape[:2]
    gap = 6
    # Wide enough for the title and caption, so every frame stays distinct in the GIF.
    canvas_width = max(len(panels) * (scale * width + gap) - gap, 480)
    frames = []
    for k, caption in enumerate(captions):
        canvas = Image.new("RGB", (canvas_width, scale * height + 110), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), title, font=font, fill="black")
        for column, (volume, cmap, low, high, label) in enumerate(panels):
            left = column * (scale * width + gap)
            tile = Image.fromarray(to_rgb(volume[..., k], cmap, low, high))
            canvas.paste(tile.resize((scale * width, scale * height), Image.Resampling.NEAREST), (left, 52))
            draw.text((left + 4, 30), label, font=small, fill="black")
        draw.text((8, scale * height + 62), caption, font=small, fill="black")
        frames.append(canvas)
    frames[len(frames) // 2].save(path.with_suffix(".png"))
    palette = shared_palette()
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=FRAME_DURATION_MS, loop=0)


def relative_change(volume):
    """Per frame ||V(t) - V(t0)|| / ||V(t0)||, the first window frame being t0."""
    reference = volume[..., 0]
    return (np.linalg.norm(volume - reference[..., None], axis=(0, 1)) / np.linalg.norm(reference)).tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--display-max", type=float, default=0.05, help="image range 0..max, raw p0 units")
    parser.add_argument("--error-max", type=float, default=0.02, help="absolute-error range 0..max, raw p0 units")
    parser.add_argument("--change-max", type=float, default=0.01,
                        help="intensity-change range -max..max, raw p0 units")
    parser.add_argument("--scale", type=int, default=2, help="integer pixel upscaling")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for inference")
    args = parser.parse_args(argv)

    config = json.loads((args.run / "config.json").read_text())
    if config.get("phantom") != moby.NAME:
        parser.error("Run is not a dce_moby run")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.run / "model.pt", weights_only=True)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data, _ = prepare_data(config)
    truth, times, labels = window_frames(data)
    x, y = data["training"]["x"], data["training"]["y"]
    reference_time = float(model.motion.reference_time)
    image, intensity, motion_only, displacement = branch_fields(model, x, y, times, reference_time)
    scale = config["data"]["intensity_scale_value"]
    truth, image, intensity, motion_only = (volume * scale for volume in (truth, image, intensity, motion_only))
    magnitude = np.linalg.norm(displacement, axis=-1)  # same units as the coordinates: mm
    magnitude_max = float(np.ceil(magnitude.max() * 20) / 20) or 0.05

    out = args.run / "visuals"
    out.mkdir(exist_ok=True)
    captions = [f"t={t:.2f} s | {label} frame | window frame {k}" for k, (t, label) in enumerate(zip(times, labels))]
    gray = (None, 0, args.display_max)
    views = {
        "image": ([(truth, *gray, "Truth"), (image, *gray, "Prediction"),
                   (np.abs(image - truth), None, 0, args.error_max, f"|Prediction - truth|, 0..{args.error_max:g}")],
                  "Image: truth | prediction | absolute error"),
        "intensity": ([(intensity, *gray, "J(q, t), fixed q, no motion"),
                       (intensity - intensity[..., :1], DIVERGING, -args.change_max, args.change_max,
                        f"J(q, t) - J(q, t0), -{args.change_max:g}..{args.change_max:g}")],
                      f"Intensity branch at fixed coordinates (t0 = {reference_time:g} s)"),
        "motion": ([(motion_only, *gray, "J(r + d(r, t), t0)"),
                    (magnitude, MAGNITUDE, 0, magnitude_max, f"|d(r, t)| in mm, 0..{magnitude_max:g}")],
                   "Motion branch: intensity frozen at t0, moved by the learned motion"),
    }
    summary = dict(run=args.run.name, code_commit=git_commit(), frames=len(times), duration_ms=FRAME_DURATION_MS,
                   reference_time=reference_time, units="raw p0 (normalized values times intensity_scale_value)",
                   intensity_scale_value=scale, display_range=[0, args.display_max],
                   error_range=[0, args.error_max], change_range=[-args.change_max, args.change_max],
                   displacement_range_mm=[0, magnitude_max], displacement_max_mm=float(magnitude.max()),
                   model_step=checkpoint["step"], times=times.tolist(), splits=labels.tolist(),
                   relative_change_from_first_frame=dict(
                       truth=relative_change(truth), prediction=relative_change(image),
                       intensity=relative_change(intensity), motion_only=relative_change(motion_only)),
                   gifs={})
    for name, (panels, title) in views.items():
        path = out / f"window_{name}.gif"
        save_view(path, panels, captions, title, args.scale)
        summary["gifs"][name] = dict(file=path.name, megabytes=round(path.stat().st_size / 2**20, 2))
        print(f"wrote {path} ({summary['gifs'][name]['megabytes']} MB)")
    (out / "branch_gifs.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
