"""Evaluate block holdouts using the existing predictions and metric definitions."""
from pathlib import Path

import numpy as np

from .evaluate import bounding_roi, intensity_rank_report, motion_scores, predict_fields
from .masked_metrics import evaluate_masked_sequence
from .metrics import error_metrics, evaluate_sequence, save_metrics
from ..splits import interpolate_training_frames


def masked_report(out, name, truth, prediction, data, split, roi, points,
                  *, data_range=1, interpolation=None):
    """Keep observed-coordinate and held-out-coordinate scores separate."""
    evaluations = {}
    for label, mask in (("train_pixels", split["train_mask"]), ("test_pixels", split["test_mask"])):
        evaluations[label] = evaluate_masked_sequence(
            truth, prediction, mask, times=data["t"], data_range=data_range, roi=roi, points=points)
    if interpolation is not None:
        evaluations["temporal_interpolation"] = evaluate_masked_sequence(
            truth, interpolation, split["test_mask"], times=data["t"], data_range=data_range,
            roi=roi, points=points)
    if name in ("ux", "uy"):
        # Relative error is not informative at numerically zero-motion frames.
        for label, mask in (("train_pixels", split["train_mask"]), ("test_pixels", split["test_mask"])):
            for scores, selection in ((evaluations[label], mask), (evaluations[label]["roi"], mask[roi])):
                values = truth if selection.shape == truth.shape else truth[roi]
                scores["per_frame_rrmse_plot_valid"] = [
                    bool(selection[..., k].any() and
                         np.sqrt(np.mean(values[..., k][selection[..., k]].astype(float) ** 2)) > 1e-8 * data_range)
                    for k in range(len(data["t"]))]
    return evaluations


def evaluate_blocks(model, data, split, config, out, training_seconds):
    out = Path(out)
    options = config.get("output", {})
    prediction = predict_fields(model, data, config, include_visuals=options.get("gifs", False))
    roi = bounding_roi([data["truth"]])
    size = config["size"]
    points = {"center": (size // 2, size // 2),
              "spot_reference": (round(.525 * (size - 1)), round(.445 * (size - 1)))}
    baseline = interpolate_training_frames(data["truth"], split["train_mask"], data["t"])
    image = masked_report(out, "image", data["truth"], prediction["image"], data, split,
                          roi, points, interpolation=baseline)
    report = dict(schema_version=1, evaluation_split="blocks", split=split["settings"], parameters=sum(p.numel() for p in model.parameters()),
                  training_seconds=training_seconds, image=image)
    if "motion" in prediction:
        # Motion is evaluated at held-out image coordinates. Its labels were
        # never used to train either branch, including at observed coordinates.
        motion = motion_scores(data["u_true"], prediction["motion"], data["foreground_mask"],
                               float(data["x"][1] - data["x"][0]), evaluation_mask=split["test_mask"])
        report["motion"] = motion
        report["motion_components"] = {}
        for i, name in enumerate(("ux", "uy")):
            values = data["u_true"][..., i]
            data_range = float(values.max() - values.min()) or 1
            report["motion_components"][name] = masked_report(out, name, values, prediction["motion"][..., i], data, split,
                          roi, points, data_range=data_range)

        # Reference intensity is a separate decomposition diagnostic on the full
        # reference grid; the observation-space test mask does not define its split.
        reference = evaluate_sequence(
            data["reference_intensity"], prediction["reference_intensity"], data_range=1,
            times=data["t"], roi=bounding_roi([data["reference_intensity"]]), points=points)
        report["reference_intensity_full_grid"] = reference
        if hasattr(model.intensity, "rank"):
            rank_report = intensity_rank_report({"all_grid": prediction["reference_intensity"]}, model.intensity.rank)
            report["intensity_rank"] = rank_report
        test = split["test_mask"]
        names = ("true_motion_learned_intensity", "learned_motion_true_intensity")
        checks = {name: error_metrics(data["truth"][test], prediction[name][test], data_range=1) for name in names}
        checks["zero_motion_learned_intensity"] = error_metrics(
            data["truth"][test], prediction["reference_intensity"][test], data_range=1)
        report["diagnostics_on_test_pixels"] = checks
    elif hasattr(model, "rank"):
        report["image_rank"] = intensity_rank_report({"all_grid": prediction["image"]}, model.rank)
    save_metrics(out / "metrics.json", report)
    if options.get("predictions", False):
        np.savez_compressed(out / "predictions.npz", **prediction,
                            x=data["x"], y=data["y"], t=data["t"])
    if options.get("gifs", False):
        from .render import save_gifs
        save_gifs(out / "visuals", data, prediction, error_mask=split["test_mask"])
    score = image["test_pixels"]
    print(f"Held-out image RRMSE={score['rrmse']:.6f}, spatial-window SSIM={score['ssim']}", flush=True)

    return report
