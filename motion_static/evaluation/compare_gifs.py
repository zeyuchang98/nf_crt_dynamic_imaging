"""Compare saved intensity-grid runs at matching times with shared display scales.

python -m motion_static.evaluation.compare_gifs RUN_128 RUN_64 RUN_32 --out OUTPUT
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .compare_training import validate_configs


def save_animation(path, truth, predictions, times, labels, title):
    """Show predictions and absolute errors without resampling time or image values."""
    errors = [np.abs(prediction - truth) for prediction in predictions]
    error_max = max(max(float(error.max()) for error in errors), 1e-8)
    tile = 240
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    frames = []
    for frame_index, time in enumerate(times):
        canvas = Image.new("RGB", ((len(predictions) + 1) * tile, 636), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 9), title, font=font, fill="black")
        for col, (label, volume) in enumerate(zip(["Ground truth", *labels], [truth, *predictions])):
            draw.text((col * tile + 10, 37), label, font=font, fill="black")
            draw.text((col * tile + 10, 56), "Fixed display range: 0..1", font=small, fill="black")
            rows = [(volume, 1.0, 76)]
            if col:
                draw.text((col * tile + 10, 328), "Absolute error", font=font, fill="black")
                rows.append((errors[col - 1], error_max, 352))
            for values, scale, top in rows:
                pixels = np.round(np.clip(np.flipud(values[..., frame_index]) / scale, 0, 1) * 255).astype(np.uint8)
                image = Image.fromarray(pixels).convert("RGB").resize((tile, tile), Image.Resampling.NEAREST)
                canvas.paste(image, (col * tile, top))
        draw.multiline_text((10, 352), f"Absolute-error range:\n0..{error_max:.4g}\n\nShared across all runs\nand all displayed times.",
                            font=small, fill="black", spacing=6)
        draw.text((10, 610), f"t={time:+.3f} | Display clipped to 0..1; metrics use raw predictions.", font=small, fill="black")
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0)
    # One common frame is selected using mean MSE across the compared runs.
    frame_mse = np.mean([np.mean(error ** 2, axis=(0, 1)) for error in errors], axis=0)
    selected = int(np.argmax(frame_mse))
    frames[selected].save(path.with_suffix(".png"))
    return dict(frames=len(times), duration_ms=90, display_range=[0, 1],
                shared_error_range=[0, error_max], still_frame_index=selected,
                still_frame_time=float(times[selected]),
                still_selection="Maximum mean frame MSE across all compared runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    configs = [json.loads((run / "config.json").read_text()) for run in args.runs]
    validate_configs(configs, vary_intensity_grid=True)
    labels = [f"Intensity max {config['encoding']['max_resolution']}" for config in configs]
    if len(set(labels)) != len(labels):
        parser.error("Each run must have a distinct intensity maximum resolution")
    args.out.mkdir(parents=True, exist_ok=True)
    from .diagnose_branches import load_midpoints
    loaded = [load_midpoints(run) for run in args.runs]
    times = loaded[0][1]["t"]
    for _, data, _ in loaded[1:]:
        if not np.array_equal(times, data["t"]):
            raise ValueError("Comparison requires identical midpoint times")
    sources = {}
    for run in args.runs:
        for name in ("config.json", "model.pt", "predictions.npz", "arrays/predictions.npz"):
            path = run / name
            if path.exists():
                sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(runs=[str(run.resolve()) for run in args.runs], labels=labels,
                  time_count=len(times), source_sha256=sources,
                  analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), animations={})
    views = [("image", "truth", "image", "Image fitting: shared times and display scales"),
             ("intensity", "reference_intensity", "reference_intensity", "Intensity at fixed reference coordinates"),
             ("motion", "truth", "learned_motion_true_intensity",
              "Motion comparison: all panels use the same true time-varying intensity")]
    for name, truth_key, prediction_key, title in views:
        truth = loaded[0][1][truth_key]
        predictions = []
        for _, data, prediction in loaded:
            if not np.array_equal(truth, data[truth_key]):
                raise ValueError("Comparison requires identical ground truth")
            predictions.append(prediction[prediction_key])
        path = args.out / f"{name}_comparison.gif"
        report["animations"][path.name] = save_animation(path, truth, predictions, times, labels, title)
        print(f"Saved {path}", flush=True)
    (args.out / "gif_comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
