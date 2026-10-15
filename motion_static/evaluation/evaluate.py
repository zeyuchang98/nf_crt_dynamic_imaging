"""Evaluate image fit and, separately, the two learned parts. No training here."""
from pathlib import Path

import numpy as np
import torch

from .metrics import evaluate_sequence, error_metrics, save_metrics
from .metric_plots import plot_evaluation
from .plots import save_comparison, save_motion_error
from ..phantoms.data import coordinate_grid, get_phantom


@torch.no_grad()
def predict(function, coordinates):
    pieces = [function(torch.as_tensor(batch, dtype=torch.float32)).cpu().numpy()
              for batch in np.array_split(coordinates, max(1, int(np.ceil(len(coordinates) / 8192))))]
    return np.concatenate(pieces)


def bounding_roi(volumes):
    mask = np.logical_or.reduce([np.any(volume > 0, axis=-1) for volume in volumes])
    indices = np.nonzero(mask)
    return tuple(slice(max(0, int(axis.min()) - 2), min(size, int(axis.max()) + 3))
                 for axis, size in zip(indices, mask.shape))


def scalar_report(out, name, data, truth, prediction, *, data_range=1, displacement=False):
    """Reuse the existing RRMSE/SSIM/PSNR, ROI, point-curve and mean-curve metrics."""
    roi = bounding_roi([entry["truth"] for entry in data.values()] if displacement else list(truth.values()))
    size = len(data["training"]["x"])
    points = {"center": (size // 2, size // 2),
              "spot_reference": (round(0.525 * (size - 1)), round(0.445 * (size - 1)))}
    evaluations, arrays = {}, {}
    for split, label, prefix in (("training", "training_grid", ""),
                                 ("midpoint", "unseen_time_midpoints", "midpoint_")):
        entry = evaluate_sequence(truth[split], prediction[split], data_range=data_range,
                                  times=data[split]["t"], roi=roi, points=points)
        if displacement:
            for scope, volume in ((entry, truth[split]), (entry["roi"], truth[split][roi])):
                rms = np.sqrt(np.mean(volume.astype(float) ** 2, axis=(0, 1)))
                scope["per_frame_rrmse_plot_valid"] = (rms > 1e-8 * data_range).tolist()
        evaluations[label] = entry
        arrays.update({prefix + "truth": truth[split], prefix + "prediction": prediction[split]})
    array_path = out / "arrays" / f"{name}.npz"
    np.savez_compressed(array_path, **arrays)
    quantity = "Reference intensity" if name == "reference_intensity" else name
    report = dict(source=str(array_path.resolve()), quantity=quantity, evaluations=evaluations)
    if displacement:
        report.update(field_type="displacement", cmap="coolwarm",
                      display_min=min(float(v.min()) for v in truth.values()))
    save_metrics(out / "metrics" / f"{name}.json", report)
    plot_evaluation(report, out / "plots" / name)
    return evaluations


def motion_scores(truth, prediction, mask, spacing, evaluation_mask=None):
    """Endpoint distance measures both components together, in pixel units."""
    endpoint = np.linalg.norm(prediction.astype(float) - truth, axis=-1) / spacing
    zero_endpoint = np.linalg.norm(truth.astype(float), axis=-1) / spacing
    whole_fov = np.ones_like(mask) if evaluation_mask is None else evaluation_mask
    foreground = mask & whole_fov
    result = {}
    for name, selected in (("whole_fov", whole_fov), ("foreground", foreground)):
        distances = endpoint[selected]
        result[name] = dict(endpoint_rmse_pixels=float(np.sqrt(np.mean(distances ** 2))),
                            mean_endpoint_error_pixels=float(distances.mean()),
                            max_endpoint_error_pixels=float(distances.max()),
                            zero_motion_endpoint_rmse_pixels=float(np.sqrt(np.mean(zero_endpoint[selected] ** 2))),
                            vector_rrmse=error_metrics(truth[selected], prediction[selected], data_range=1)["rrmse"])
    result["per_frame_foreground_endpoint_rmse_pixels"] = [
        float(np.sqrt(np.mean(endpoint[..., k][foreground[..., k]] ** 2)))
        if foreground[..., k].any() else None for k in range(mask.shape[-1])]
    # Finite-difference determinant of q(r)=r+u(r), not of the forward transform.
    ux_y, ux_x = np.gradient(prediction[..., 0], spacing, spacing, axis=(0, 1))
    uy_y, uy_x = np.gradient(prediction[..., 1], spacing, spacing, axis=(0, 1))
    determinant = (1 + ux_x) * (1 + uy_y) - ux_y * uy_x
    result["foreground_nonpositive_jacobian_fraction"] = float(np.mean(determinant[foreground] <= 0))
    return result


def predict_fields(model, entry, config):
    """Query the image and, for nested models, the independently inspectable parts."""
    phantom = get_phantom(config["phantom"])
    coordinates = coordinate_grid(entry["x"], entry["y"], entry["t"])
    shape = entry["truth"].shape
    pred = {"image": predict(model, coordinates).reshape(shape)}
    if hasattr(model, "motion") and hasattr(model, "intensity"):
        pred["motion"] = predict(model.motion, coordinates).reshape(*shape, 2)
        pred["reference_intensity"] = predict(model.intensity, coordinates).reshape(shape)
        true_warp = coordinates.copy()
        true_warp[:, :2] += entry["u_true"].reshape(-1, 2)
        learned_warp = coordinates.copy()
        learned_warp[:, :2] += pred["motion"].reshape(-1, 2)
        pred["true_motion_learned_intensity"] = predict(model.intensity, true_warp).reshape(shape)
        pred["learned_motion_true_intensity"] = phantom.reference_intensity(learned_warp, config["size"]).reshape(shape)
        # Freeze learned intensity at t0: all displayed time changes now come
        # from motion. This diagnostic does not need a true reference image.
        frozen = coordinates.copy()
        frozen[:, 2] = float(model.motion.reference_time)
        pred["frozen_reference_intensity"] = predict(model.intensity, frozen).reshape(shape)
        frozen[:, :2] += pred["motion"].reshape(-1, 2)
        pred["motion_only_frozen_intensity"] = predict(model.intensity, frozen).reshape(shape)
    return pred


def intensity_rank_report(volumes, rank):
    """Check rank on a fixed reference grid, without centering or warping it."""
    report = dict(rank_limit=rank, definition="Total uncentered space-by-time matrix rank", splits={})
    for name, volume in volumes.items():
        matrix = volume.reshape(-1, volume.shape[-1])
        singular = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
        norm = np.linalg.norm(singular)
        report["splits"][name] = dict(
            singular_values=singular.tolist(),
            relative_frobenius_tail_after_rank=float(np.linalg.norm(singular[rank:]) / norm) if norm else 0.,
            prediction_dtype=str(matrix.dtype),
        )
    return report


def evaluate(model, data, config, out, training_seconds):
    out = Path(out)
    for directory in ("arrays", "metrics", "plots"):
        (out / directory).mkdir(exist_ok=True)
    nested = hasattr(model, "motion") and hasattr(model, "intensity")
    predictions = {}
    motion, diagnostics = {}, {}
    for split, entry in data.items():
        pred = predict_fields(model, entry, config)
        if nested:
            motion[split] = motion_scores(entry["u_true"], pred["motion"], entry["foreground_mask"],
                                           float(entry["x"][1] - entry["x"][0]))
            diagnostics[split] = {
                name: error_metrics(entry["truth"], pred[name], data_range=1)
                for name in ("true_motion_learned_intensity", "learned_motion_true_intensity")}
            diagnostics[split]["zero_motion_learned_intensity"] = error_metrics(
                entry["truth"], pred["reference_intensity"], data_range=1)
        predictions[split] = pred
    np.savez_compressed(out / "arrays" / "predictions.npz",
                        **{f"{split}_{name}": values for split, pred in predictions.items()
                           for name, values in pred.items()},
                        training_t=data["training"]["t"], midpoint_t=data["midpoint"]["t"])
    image = scalar_report(out, "image", data,
                          {s: d["truth"] for s, d in data.items()},
                          {s: p["image"] for s, p in predictions.items()})
    report = dict(config=config, parameters=sum(p.numel() for p in model.parameters()),
                  training_seconds=training_seconds, image=image)
    save_comparison(out / "plots" / "image.gif", data["midpoint"]["truth"],
                    predictions["midpoint"]["image"], data["midpoint"]["t"],
                    title=f"{config['phantom']} / {config['model']} / unseen midpoints")
    if nested:
        report["reference_intensity"] = scalar_report(out, "reference_intensity", data,
            {s: d["reference_intensity"] for s, d in data.items()}, {s: p["reference_intensity"] for s, p in predictions.items()})
        report["motion"] = motion
        report["diagnostics"] = diagnostics
        for i, component in enumerate(("ux", "uy")):
            truth = {s: d["u_true"][..., i] for s, d in data.items()}
            data_range = max(float(v.max()) for v in truth.values()) - min(float(v.min()) for v in truth.values())
            # Simple phantom ux is identically zero; a declared unit range is used for SSIM/PSNR.
            scalar_report(out, component, data, truth,
                          {s: p["motion"][..., i] for s, p in predictions.items()},
                          data_range=data_range if data_range > 0 else 1, displacement=True)
        save_comparison(out / "plots" / "reference_intensity.gif", data["midpoint"]["reference_intensity"],
                        predictions["midpoint"]["reference_intensity"], data["midpoint"]["t"],
                        title="Reference intensity C(q,t): true vs learned; no motion applied")
        save_comparison(out / "plots" / "motion_only.gif", predictions["midpoint"]["frozen_reference_intensity"],
                        predictions["midpoint"]["motion_only_frozen_intensity"], data["midpoint"]["t"],
                        title="Motion only: learned intensity frozen at reference time; unseen midpoints",
                        labels=("Fixed learned C(r,t0)", "Warped C(r + u_pred,t0)", "Absolute change (not error)"))
        save_motion_error(out / "plots", data["midpoint"]["t"], motion["midpoint"])
        save_metrics(out / "metrics" / "motion.json", motion)
        save_metrics(out / "metrics" / "part_checks.json", diagnostics)
        if hasattr(model.intensity, "rank"):
            rank_report = intensity_rank_report(
                {s: p["reference_intensity"] for s, p in predictions.items()}, model.intensity.rank)
            report["intensity_rank"] = rank_report
            save_metrics(out / "metrics" / "intensity_rank.json", rank_report)
    save_metrics(out / "summary.json", report)
    selected = report["image"]["unseen_time_midpoints"]
    print(f"Midpoint image RRMSE={selected['rrmse']:.4%}, SSIM={selected['ssim']:.6f}", flush=True)
