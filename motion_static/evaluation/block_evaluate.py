"""Evaluate block holdouts using the existing predictions and metric definitions."""
from pathlib import Path

import numpy as np

from .block_plots import save_split_plot, save_test_curves
from .evaluate import bounding_roi, intensity_rank_report, motion_scores, predict_fields
from .masked_metrics import evaluate_masked_sequence
from .metric_plots import plot_evaluation
from .metrics import error_metrics, evaluate_sequence, save_metrics
from .plots import save_comparison, save_motion_error
from ..splits import interpolate_training_frames


def masked_report(out, name, truth, prediction, data, split, roi, points,
                  *, data_range=1, interpolation=None):
    """Keep observed-coordinate and held-out-coordinate scores separate."""
    evaluations = {}
    for label, mask in (("train_pixels", split["train_mask"]), ("test_pixels", split["test_mask"])):
        evaluations[label] = evaluate_masked_sequence(
            truth, prediction, mask, times=data["t"], data_range=data_range, roi=roi, points=points)
    arrays = dict(truth=truth, prediction=prediction, t=data["t"],
                  train_mask=split["train_mask"], test_mask=split["test_mask"])
    if interpolation is not None:
        evaluations["temporal_interpolation"] = evaluate_masked_sequence(
            truth, interpolation, split["test_mask"], times=data["t"], data_range=data_range,
            roi=roi, points=points)
        arrays["interpolation"] = interpolation
    source = out / "arrays" / f"{name}.npz"
    np.savez_compressed(source, **arrays)
    report = dict(source=str(source.resolve()), quantity=name, evaluations=evaluations,
                  framewise_title=f"{name}: observed vs held-out coordinates; spatial SSIM uses selected windows")
    if name in ("ux", "uy"):
        report["field_type"] = "displacement"
        # Relative error is not informative at numerically zero-motion frames.
        for label, mask in (("train_pixels", split["train_mask"]), ("test_pixels", split["test_mask"])):
            for scores, selection in ((evaluations[label], mask), (evaluations[label]["roi"], mask[roi])):
                values = truth if selection.shape == truth.shape else truth[roi]
                scores["per_frame_rrmse_plot_valid"] = [
                    bool(selection[..., k].any() and
                         np.sqrt(np.mean(values[..., k][selection[..., k]].astype(float) ** 2)) > 1e-8 * data_range)
                    for k in range(len(data["t"]))]
    save_metrics(out / "metrics" / f"{name}.json", report)
    plot_evaluation(report, out / "plots" / name, framewise_only=True)
    return evaluations


def evaluate_blocks(model, data, split, config, out, training_seconds):
    out = Path(out)
    for folder in ("arrays", "metrics", "plots"):
        (out / folder).mkdir(exist_ok=True)
    prediction = predict_fields(model, data, config)
    np.savez_compressed(out / "arrays" / "predictions.npz", **prediction,
                        x=data["x"], y=data["y"], t=data["t"])
    roi = bounding_roi([data["truth"]])
    size = config["size"]
    points = {"center": (size // 2, size // 2),
              "spot_reference": (round(.525 * (size - 1)), round(.445 * (size - 1)))}
    baseline = interpolate_training_frames(data["truth"], split["train_mask"], data["t"])
    image = masked_report(out, "image", data["truth"], prediction["image"], data, split,
                          roi, points, interpolation=baseline)
    report = dict(config=config, split=split["settings"], parameters=sum(p.numel() for p in model.parameters()),
                  training_seconds=training_seconds, image=image)
    save_split_plot(out / "plots", data, split)
    save_test_curves(out / "plots" / "image", image["test_pixels"], image["temporal_interpolation"])
    save_comparison(out / "plots" / "image.gif", data["truth"], prediction["image"], data["t"],
                    title=f"{config['phantom']} / {config['model']} / block holdout",
                    labels=("Ground truth", "Prediction", "Error: held-out pixels"), error_mask=split["test_mask"])
    if "motion" in prediction:
        # Motion is evaluated at held-out image coordinates. Its labels were
        # never used to train either branch, including at observed coordinates.
        motion = motion_scores(data["u_true"], prediction["motion"], data["foreground_mask"],
                               float(data["x"][1] - data["x"][0]), evaluation_mask=split["test_mask"])
        report["motion"] = motion
        save_metrics(out / "metrics" / "motion.json", motion)
        save_motion_error(out / "plots", data["t"], motion, sample_label="Held-out pixels")
        for i, name in enumerate(("ux", "uy")):
            values = data["u_true"][..., i]
            data_range = float(values.max() - values.min()) or 1
            masked_report(out, name, values, prediction["motion"][..., i], data, split,
                          roi, points, data_range=data_range)

        # Reference intensity is a separate decomposition diagnostic on the full
        # reference grid; the observation-space test mask does not define its split.
        reference = evaluate_sequence(
            data["reference_intensity"], prediction["reference_intensity"], data_range=1,
            times=data["t"], roi=bounding_roi([data["reference_intensity"]]), points=points)
        report["reference_intensity_full_grid"] = reference
        source = out / "arrays" / "reference_intensity.npz"
        np.savez_compressed(source, truth=data["reference_intensity"], prediction=prediction["reference_intensity"])
        reference_report = dict(source=str(source.resolve()), quantity="Reference intensity",
                                evaluations={"all_grid": reference},
                                framewise_title="Reference intensity: full-grid decomposition diagnostic")
        save_metrics(out / "metrics" / "reference_intensity.json", reference_report)
        plot_evaluation(reference_report, out / "plots" / "reference_intensity")
        save_comparison(out / "plots" / "reference_intensity.gif", data["reference_intensity"],
                        prediction["reference_intensity"], data["t"],
                        title="Reference intensity: full-grid decomposition diagnostic")
        save_comparison(out / "plots" / "motion_only.gif", prediction["frozen_reference_intensity"],
                        prediction["motion_only_frozen_intensity"], data["t"],
                        title="Motion only: learned intensity frozen at reference time; full-grid diagnostic",
                        labels=("Fixed learned C(r,t0)", "Warped C(r + u_pred,t0)", "Absolute change (not error)"))
        if hasattr(model.intensity, "rank"):
            rank_report = intensity_rank_report({"all_grid": prediction["reference_intensity"]}, model.intensity.rank)
            report["intensity_rank"] = rank_report
            save_metrics(out / "metrics" / "intensity_rank.json", rank_report)
        test = split["test_mask"]
        names = ("true_motion_learned_intensity", "learned_motion_true_intensity")
        checks = {name: error_metrics(data["truth"][test], prediction[name][test], data_range=1) for name in names}
        checks["zero_motion_learned_intensity"] = error_metrics(
            data["truth"][test], prediction["reference_intensity"][test], data_range=1)
        report["diagnostics_on_test_pixels"] = checks
        save_metrics(out / "metrics" / "part_checks.json", checks)
    save_metrics(out / "summary.json", report)
    score = image["test_pixels"]
    print(f"Held-out image RRMSE={score['rrmse']:.6f}, spatial-window SSIM={score['ssim']:.6f}", flush=True)
