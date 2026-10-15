"""Evaluate image fit and, separately, the two learned parts. No training here."""
from pathlib import Path

import numpy as np
import torch

from .metrics import evaluate_sequence, error_metrics, save_metrics
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
    evaluations = {}
    for split, label in (("training", "training_grid"), ("midpoint", "unseen_time_midpoints")):
        entry = evaluate_sequence(truth[split], prediction[split], data_range=data_range,
                                  times=data[split]["t"], roi=roi, points=points)
        if displacement:
            for scope, volume in ((entry, truth[split]), (entry["roi"], truth[split][roi])):
                rms = np.sqrt(np.mean(volume.astype(float) ** 2, axis=(0, 1)))
                scope["per_frame_rrmse_plot_valid"] = (rms > 1e-8 * data_range).tolist()
        evaluations[label] = entry
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


def predict_fields(model, entry, config, *, include_visuals=True):
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
        if include_visuals:
            # Freeze learned intensity at t0: all displayed time changes now come
            # from motion. This diagnostic does not need a true reference image.
            frozen = coordinates.copy()
            frozen[:, 2] = float(model.motion.reference_time)
            pred["frozen_reference_intensity"] = predict(model.intensity, frozen).reshape(shape)
            frozen[:, :2] += pred["motion"].reshape(-1, 2)
            pred["motion_only_frozen_intensity"] = predict(model.intensity, frozen).reshape(shape)
    return pred


def intensity_rank_report(volumes, rank):
    """Check rank on the supplied fixed spatial grid, without centering/warping."""
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
    """Compute the existing full-grid metrics; visual/export files are opt-in."""
    out = Path(out)
    options = config.get("output", {})
    visuals = options.get("gifs", False)
    nested = hasattr(model, "motion") and hasattr(model, "intensity")
    predictions = {}
    motion, diagnostics = {}, {}
    for split, entry in data.items():
        pred = predict_fields(model, entry, config, include_visuals=visuals)
        if nested:
            motion[split] = motion_scores(entry["u_true"], pred["motion"], entry["foreground_mask"],
                                         float(entry["x"][1] - entry["x"][0]))
            diagnostics[split] = {
                name: error_metrics(entry["truth"], pred[name], data_range=1)
                for name in ("true_motion_learned_intensity", "learned_motion_true_intensity")}
            diagnostics[split]["zero_motion_learned_intensity"] = error_metrics(
                entry["truth"], pred["reference_intensity"], data_range=1)
        predictions[split] = pred
    image = scalar_report(out, "image", data,
                          {s: d["truth"] for s, d in data.items()},
                          {s: p["image"] for s, p in predictions.items()})
    report = dict(schema_version=1, evaluation_split="midpoints",
                  parameters=sum(p.numel() for p in model.parameters()),
                  training_seconds=training_seconds, image=image)
    if nested:
        report["reference_intensity"] = scalar_report(out, "reference_intensity", data,
            {s: d["reference_intensity"] for s, d in data.items()},
            {s: p["reference_intensity"] for s, p in predictions.items()})
        report["motion"] = motion
        report["diagnostics"] = diagnostics
        report["motion_components"] = {}
        for i, component in enumerate(("ux", "uy")):
            truth = {s: d["u_true"][..., i] for s, d in data.items()}
            data_range = max(float(v.max()) for v in truth.values()) - min(float(v.min()) for v in truth.values())
            report["motion_components"][component] = scalar_report(out, component, data, truth,
                {s: p["motion"][..., i] for s, p in predictions.items()},
                data_range=data_range if data_range > 0 else 1, displacement=True)
        if hasattr(model.intensity, "rank"):
            report["intensity_rank"] = intensity_rank_report(
                {s: p["reference_intensity"] for s, p in predictions.items()}, model.intensity.rank)
    elif hasattr(model, "rank"):
        # A motion-free low-rank model constrains the image matrix at fixed r,
        # not a recovered reference-intensity matrix at q.
        report["image_rank"] = intensity_rank_report(
            {s: p["image"] for s, p in predictions.items()}, model.rank)
    save_metrics(out / "metrics.json", report)
    if options.get("predictions", False):
        np.savez_compressed(out / "predictions.npz",
            **{f"{split}_{name}": values for split, pred in predictions.items() for name, values in pred.items()},
            x=data["training"]["x"], y=data["training"]["y"],
            training_t=data["training"]["t"], midpoint_t=data["midpoint"]["t"])
    if visuals:
        from .render import save_gifs
        save_gifs(out / "visuals", data["midpoint"], predictions["midpoint"], config=config,
                  reference_time=float(model.motion.reference_time) if nested else None)
    selected = image["unseen_time_midpoints"]
    print(f"Midpoint image RRMSE={selected['rrmse']:.4%}, SSIM={selected['ssim']}", flush=True)
    return report
