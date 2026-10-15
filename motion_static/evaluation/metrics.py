"""Reusable NumPy evaluation for scalar 2D+t or 3D+t image sequences.

RRMSE is relative L2 error. Whole-sequence SSIM treats time as a windowed
dimension, not a channel; per-frame SSIM is also reported separately. These
are the metric families used in the NF papers, not an exact reproduction of
their unspecified window/ROI settings. Predictions are never clipped.
"""
import json
from pathlib import Path

import numpy as np
import skimage
from skimage.metrics import structural_similarity


def error_metrics(truth, prediction, *, data_range):
    """Scalar errors on matching finite arrays (also usable for activity curves).

    Zero-reference RRMSE is undefined (NaN), even for an exact zero prediction.
    Exact predictions have infinite PSNR. JSON export preserves these as strings.
    """
    truth, prediction = _pair(truth, prediction)
    _check_range(data_range)
    error = prediction - truth
    mse = float(np.mean(error ** 2))
    reference_energy = float(np.sum(truth ** 2))
    rrmse = float(np.sqrt(np.sum(error ** 2) / reference_energy)) if reference_energy else float("nan")
    return {
        "mse": mse, "rmse": float(np.sqrt(mse)), "rrmse": rrmse,
        "relative_l2": rrmse,
        "psnr_db": float(20 * np.log10(data_range) - 10 * np.log10(mse)) if mse else float("inf"),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def _pair(truth, prediction):
    arrays = []
    for value in (truth, prediction):
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise ValueError("Inputs must be real numeric arrays")
        array = array.astype(np.float64, copy=False)
        if not array.size or not np.isfinite(array).all():
            raise ValueError("Inputs must be nonempty and finite")
        arrays.append(array)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("Truth and prediction shapes must match exactly")
    return arrays


def _check_range(data_range):
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")


def _ssim(truth, prediction, data_range, win_size):
    # Do not silently change the window between sequences or pad short axes.
    if min(truth.shape) < win_size:
        return None
    return float(structural_similarity(
        truth, prediction, data_range=data_range, win_size=win_size,
        gaussian_weights=False, use_sample_covariance=True,
        K1=0.01, K2=0.03, channel_axis=None))


def _image_metrics(truth, prediction, data_range, win_size):
    result = error_metrics(truth, prediction, data_range=data_range)
    result["ssim"] = _ssim(truth, prediction, data_range, win_size)
    return result


def _volume_metrics(truth, prediction, data_range, win_size):
    result = _image_metrics(truth, prediction, data_range, win_size)
    frames = [_image_metrics(truth[..., k], prediction[..., k], data_range, win_size)
              for k in range(truth.shape[-1])]
    for key in frames[0]:
        result[f"per_frame_{key}"] = [frame[key] for frame in frames]
    return result


def evaluate_sequence(truth, prediction, *, data_range, time_axis=-1,
                      times=None, roi=None, activity_mask=None, points=None,
                      ssim_win_size=7):
    """Evaluate a scalar sequence; spatial axis order is preserved.

    roi: tuple of contiguous spatial slices, evaluated as a rectangular crop.
        SSIM windows stay inside the crop; no flattening or zero-filled mask.
    activity_mask: boolean spatial mask or boolean sequence in the input axis
        order. A time-dependent mask may follow a known lesion. The SAME mask
        is applied to truth and prediction; every frame must select pixels.
        A rectangular ROI supplies the default activity mask if none is given.
    points: mapping from names to spatial index tuples for fixed-point TACs.
    times: strictly increasing sample times, or None for integer frame indices.

    Curve errors weight sampled times equally (not quadrature over time).
    SSIM is None when any evaluated dimension is shorter than the fixed window.
    Whole-volume metrics are NOT averages of per-frame metrics.
    """
    truth, prediction = _pair(truth, prediction)
    _check_range(data_range)
    if truth.ndim not in (3, 4):
        raise ValueError("Expected a scalar 2D+t or 3D+t sequence")
    if not isinstance(time_axis, (int, np.integer)) or not -truth.ndim <= time_axis < truth.ndim:
        raise ValueError("Invalid time_axis")
    time_axis %= truth.ndim
    input_shape = truth.shape
    truth = np.moveaxis(truth, time_axis, -1)
    prediction = np.moveaxis(prediction, time_axis, -1)
    if not isinstance(ssim_win_size, (int, np.integer)) or ssim_win_size < 3 or ssim_win_size % 2 != 1:
        raise ValueError("ssim_win_size must be an odd integer >= 3")
    times = np.arange(truth.shape[-1], dtype=float) if times is None else np.asarray(times, dtype=float)
    if times.shape != (truth.shape[-1],) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite, strictly increasing, and match frame count")
    result = _volume_metrics(truth, prediction, data_range, ssim_win_size)
    result["times"] = times.tolist()
    result["settings"] = {
        "input_shape": list(input_shape), "time_axis": time_axis,
        "data_range": float(data_range), "prediction_clipping": False,
        "ssim": {"scope": "whole spatiotemporal array; separate spatial per-frame scores",
                 "win_size": int(ssim_win_size), "gaussian_weights": False,
                 "use_sample_covariance": True, "K1": 0.01, "K2": 0.03,
                 "channel_axis": None, "short_axes": "null; window is not changed",
                 "skimage_version": skimage.__version__},
        "roi": None,
    }
    if roi is not None:
        if len(roi) != truth.ndim - 1:
            raise ValueError("roi must contain one slice per spatial dimension")
        normalized = []
        for item, size in zip(roi, truth.shape[:-1]):
            if not isinstance(item, slice) or item.step not in (None, 1):
                raise ValueError("ROI slices must be contiguous")
            start = 0 if item.start is None else item.start
            stop = size if item.stop is None else item.stop
            if not all(isinstance(v, (int, np.integer)) for v in (start, stop)) or not 0 <= start < stop <= size:
                raise ValueError("ROI bounds must be nonempty and within the spatial grid")
            normalized.append(slice(int(start), int(stop)))
        roi = tuple(normalized)
        result["settings"]["roi"] = [[s.start, s.stop] for s in roi]
        result["roi"] = _volume_metrics(truth[roi], prediction[roi], data_range, ssim_win_size)
        if activity_mask is None:
            activity_mask = np.zeros(truth.shape[:-1], dtype=bool)
            activity_mask[roi] = True
    if activity_mask is not None:
        mask = np.asarray(activity_mask)
        if mask.dtype != np.bool_:
            raise ValueError("activity_mask must be boolean")
        if mask.shape == truth.shape[:-1]:
            mask = np.broadcast_to(mask[..., None], truth.shape)
            mask_kind = "fixed spatial mask"
        elif mask.shape == input_shape:
            mask = np.moveaxis(mask, time_axis, -1)
            mask_kind = "time-dependent mask"
        else:
            raise ValueError("activity_mask must match spatial shape or input sequence shape")
        spatial_axes = tuple(range(truth.ndim - 1))
        counts = mask.sum(axis=spatial_axes)
        if np.any(counts == 0):
            raise ValueError("activity_mask must select at least one pixel in every frame")
        actual = np.sum(np.where(mask, truth, 0), axis=spatial_axes) / counts
        estimated = np.sum(np.where(mask, prediction, 0), axis=spatial_axes) / counts
        result["activity_curve"] = {
            "mask_kind": mask_kind, "pixel_counts": counts.tolist(),
            "truth": actual.tolist(), "prediction": estimated.tolist(),
            **error_metrics(actual, estimated, data_range=data_range),
        }
    result["point_curves"] = {}
    for name, point in (points or {}).items():
        if len(point) != truth.ndim - 1 or any(
            not isinstance(i, (int, np.integer)) or not 0 <= i < size
            for i, size in zip(point, truth.shape[:-1])
        ):
            raise ValueError(f"Point {name!r} must contain valid spatial indices")
        point = tuple(point)
        actual, estimated = truth[point], prediction[point]
        result["point_curves"][name] = {
            "spatial_index": list(point), "truth": actual.tolist(),
            "prediction": estimated.tolist(),
            **error_metrics(actual, estimated, data_range=data_range),
        }
    return result


def save_metrics(path, report):
    """Write strict JSON; undefined values and infinities use explicit strings."""
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return "undefined" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        if isinstance(value, np.generic):
            return value.item()
        return value
    Path(path).write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
