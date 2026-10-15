"""Independent intensity-geometry and motion diagnostics. No model training."""
import numpy as np
from scipy.optimize import least_squares

from .metrics import error_metrics
from ..phantoms import complex, simple


def toy_geometry_template(phantom, x, y):
    """Build a toy-specific renderer and fit bounds; no fixed pixel resolution.

    Parameters are [tx, ty, sx, sy, brightness gains...]. Complex has two
    independent gains; simple has one. This uses known geometry for diagnosis.
    """
    xx, yy = np.meshgrid(x, y)
    grid = np.stack((xx, yy), axis=-1)
    gain_count = 2 if phantom == "complex" else 1
    extent = np.array([np.ptp(x), np.ptp(y)])
    initial = np.r_[0., 0., 1., 1., np.ones(gain_count)]
    lower = np.r_[-.4 * extent, .4, .4, np.zeros(gain_count)]
    upper = np.r_[.4 * extent, 2.5, 2.5, np.full(gain_count, 3.)]

    def render(parameters):
        translation, scale, gains = parameters[:2], parameters[2:4], parameters[4:]
        xy = ((grid - translation) / scale).reshape(-1, 2)
        if phantom == "complex":
            body, spot1, spot2 = complex.reference_shapes(xy, len(x))
            values = gains[0] * (.28 * body + .20 * spot2) + gains[1] * .38 * spot1
        elif phantom == "simple":
            xyt = np.column_stack((xy, np.zeros(len(xy))))
            values = gains[0] * simple.reference_intensity(xyt, len(x))
        else:
            raise ValueError(f"No known geometry template for {phantom}")
        return values.reshape(xx.shape)

    return render, initial, (lower, upper)


def fit_intensity_geometry(volume, x, y, render, initial, bounds):
    """Fit a renderer to each frame, separating geometry from brightness gains.

    Three starts reduce sensitivity to local minima: previous frame, identity,
    and image centroid. The returned fit error measures template mismatch.
    A different known shape can be supplied through render(parameters).
    """
    xx, yy = np.meshgrid(x, y)
    parameters, fitted, errors = [], [], []
    previous = initial.copy()
    for target in np.moveaxis(volume.astype(float), -1, 0):
        centered = initial.copy()
        positive = np.maximum(target, 0)
        if positive.sum() > 0:
            centroid = [(positive * axis).sum() / positive.sum() for axis in (xx, yy)]
            centered[:2] = np.clip(centroid, .875 * bounds[0][:2], .875 * bounds[1][:2])
        solutions = [least_squares(lambda p: (render(p) - target).ravel(), start,
                                   bounds=bounds, max_nfev=120, ftol=1e-8, xtol=1e-8, gtol=1e-8)
                     for start in (previous, initial, centered)]
        best = min(solutions, key=lambda fit: fit.cost)
        previous = best.x
        image = render(best.x)
        parameters.append(best.x)
        fitted.append(image)
        errors.append(error_metrics(target, image, data_range=1)["rrmse"])
    return dict(parameters=np.array(parameters), images=np.stack(fitted, axis=-1), rrmse=np.array(errors))


def relative_geometry(parameters, spacing):
    """Measure translation and size relative to the first evaluated frame."""
    shift = (parameters[:, :2] - parameters[0, :2]) / spacing
    scale = parameters[:, 2:4] / parameters[0, 2:4]
    return shift, scale


def geometry_statistics(fit, spacing, true_translation):
    shift, scale = relative_geometry(fit["parameters"], spacing)
    correlation = [float(np.corrcoef(shift[:, i], true_translation[:, i])[0, 1])
                   if np.ptp(true_translation[:, i]) > 1e-12 and np.ptp(shift[:, i]) > 1e-12
                   else None for i in range(2)]
    return dict(
        max_translation_from_first_midpoint_pixels=float(np.linalg.norm(shift, axis=1).max()),
        rms_translation_from_first_midpoint_pixels=float(np.sqrt(np.mean(np.sum(shift ** 2, axis=1)))),
        x_shift_range_pixels=[float(shift[:, 0].min()), float(shift[:, 0].max())],
        y_shift_range_pixels=[float(shift[:, 1].min()), float(shift[:, 1].max())],
        x_scale_range=[float(scale[:, 0].min()), float(scale[:, 0].max())],
        y_scale_range=[float(scale[:, 1].min()), float(scale[:, 1].max())],
        template_fit_median_rrmse=float(np.median(fit["rrmse"])),
        template_fit_max_rrmse=float(fit["rrmse"].max()),
        translation_correlation_with_observed_truth_xy=correlation)


def true_geometry(phantom, times):
    """Analytical forward translation and scale, used only for control plots."""
    if phantom == "complex":
        translation, scale, _, _ = complex.dynamics(times)
        return translation, scale
    translation = np.zeros((len(times), 2))
    translation[:, 1] = .15 * np.sin(2 * np.pi * (times + .5))
    return translation, np.ones_like(translation)


def calibrate_geometry(data, phantom, template):
    """Check brightness-only and known-motion controls before fitting predictions."""
    render, initial, bounds = template
    stationary = fit_intensity_geometry(data["reference_intensity"], data["x"], data["y"], render, initial, bounds)
    moving = fit_intensity_geometry(data["truth"], data["x"], data["y"], render, initial, bounds)
    spacing = np.array([data[axis][1] - data[axis][0] for axis in ("x", "y")])
    translation, scale = true_geometry(phantom, data["t"])
    stationary_shift, _ = relative_geometry(stationary["parameters"], spacing)
    scores = dict(
        stationary_max_shift_pixels=float(np.linalg.norm(stationary_shift, axis=1).max()),
        stationary_max_scale_error=float(np.max(np.abs(stationary["parameters"][:, 2:4] - 1))),
        moving_max_translation_error_pixels=float(np.linalg.norm((moving["parameters"][:, :2] - translation) / spacing, axis=1).max()),
        moving_max_scale_error=float(np.max(np.abs(moving["parameters"][:, 2:4] - scale))))
    if max(scores["stationary_max_shift_pixels"], scores["moving_max_translation_error_pixels"]) > .01 or max(
            scores["stationary_max_scale_error"], scores["moving_max_scale_error"]) > .001:
        raise ValueError(f"Known-geometry control failed: {scores}")
    return scores, stationary, moving


def motion_statistics(truth, prediction, foreground, spacing):
    """Score direct network displacements, converting x/y to their pixel units."""
    actual = truth.astype(float) / spacing
    predicted = prediction.astype(float) / spacing
    a, b = actual[foreground], predicted[foreground]
    component_rmse = np.sqrt(np.mean((a - b) ** 2, axis=0))
    true_energy, predicted_energy = np.sum(a * a), np.sum(b * b)
    dot = np.sum(a * b)
    frames = [[], [], []]
    for k in range(foreground.shape[-1]):
        selected = foreground[..., k]
        current_true, current_pred = actual[..., k, :][selected], predicted[..., k, :][selected]
        for scores, values in zip(frames, (current_pred - current_true, current_true, current_pred)):
            scores.append(float(np.sqrt(np.mean(np.sum(values ** 2, axis=-1)))) if selected.any() else None)
    return dict(
        midpoint_foreground_component_rmse_pixels=component_rmse.tolist(),
        midpoint_foreground_endpoint_rmse_pixels=float(np.linalg.norm(component_rmse)),
        true_displacement_rms_pixels=float(np.sqrt(np.mean(np.sum(a * a, axis=-1)))),
        predicted_displacement_rms_pixels=float(np.sqrt(np.mean(np.sum(b * b, axis=-1)))),
        pooled_vector_cosine=float(dot / np.sqrt(true_energy * predicted_energy)) if true_energy * predicted_energy > 0 else None,
        projection_gain_onto_true_motion=float(dot / true_energy) if true_energy > 0 else None,
        per_frame_full_foreground_endpoint_rmse_pixels=frames[0],
        per_frame_full_foreground_zero_baseline_pixels=frames[1],
        per_frame_full_foreground_predicted_rms_pixels=frames[2])
