"""Truth-versus-prediction GIF of whole images for a saved dce_moby run.

Uses the run's `predictions.npz` (run with output.predictions = true) and reloads the
truth from the run's own data settings. Both images are shown in raw p0 units (the
normalized values times the run's intensity scale) with one fixed display range.

python -m motion_static.evaluation.moby_fit_gif motion_static/results/RUN
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .moby_gifs import save_animation
from ..phantoms import moby
from ..run import prepare_data


SEPARATOR_PIXELS = 2


def side_by_side(truth, prediction, separator_value):
    """Join two (y, x, t) volumes into one (x, y, t) array: truth left, prediction right."""
    truth_xy, prediction_xy = np.transpose(truth, (1, 0, 2)), np.transpose(prediction, (1, 0, 2))
    separator = np.full((SEPARATOR_PIXELS, *truth_xy.shape[1:]), separator_value, dtype=truth_xy.dtype)
    return np.concatenate([truth_xy, separator, prediction_xy], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--split", choices=("midpoint", "training", "validation", "test"), default=None,
                        help="held-out midpoint frames (default; test frames for the alternate split), "
                             "observed training frames, or the alternate split's validation frames")
    parser.add_argument("--display-max", type=float, default=0.05,
                        help="upper end of the fixed display range, raw p0 units (default 0.05, as the data GIFs)")
    parser.add_argument("--scale", type=int, default=2, help="integer pixel upscaling")
    args = parser.parse_args()

    config = json.loads((args.run / "config.json").read_text())
    if config.get("phantom") != moby.NAME:
        parser.error("Run is not a dce_moby run")
    predictions = np.load(args.run / "predictions.npz")
    scale = config["data"]["intensity_scale_value"]
    data, _ = prepare_data(config)
    if args.split is None:
        args.split = "test" if "test" in data else "midpoint"
    if args.split not in data:
        parser.error(f"This run has no {args.split} frames; it has {sorted(data)}")
    entry = data[args.split]
    prediction = predictions[f"{args.split}_image"]
    if prediction.shape != entry["truth"].shape or not np.allclose(predictions[f"{args.split}_t"], entry["t"]):
        raise ValueError("Saved predictions do not match the run's data window")

    images = side_by_side(entry["truth"] * scale, prediction * scale, args.display_max)
    label = {"midpoint": "held-out", "training": "training", "validation": "validation", "test": "test"}[args.split]
    captions = [f"t={t:.2f} s | {label} frame {k}" for k, t in enumerate(entry["t"])]
    out = args.run / "visuals"
    out.mkdir(exist_ok=True)
    info = save_animation(out / f"{args.split}_truth_vs_prediction.gif", images, captions,
                          f"Truth (left) | prediction (right), {label} frames", args.display_max, args.scale)
    info.update(split=args.split, units="raw p0 (normalized values times intensity_scale_value)",
                intensity_scale_value=scale)
    (out / f"{args.split}_truth_vs_prediction.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {out / info['file']}")


if __name__ == "__main__":
    main()
