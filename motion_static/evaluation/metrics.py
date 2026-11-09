"""Reusable NumPy evaluation for scalar 2D+t or 3D+t image sequences.

RRMSE is relative L2 error. Whole-sequence SSIM treats time as a windowed
dimension, not a channel; per-frame SSIM is also reported separately. These
are the metric families used in the NF papers, not an exact reproduction of
their unspecified window/ROI settings. Predictions are never clipped.
"""
import json
import math
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


# skimage keeps about 16 float64 arrays of the input's size for one SSIM call (local
# means, moments, the SSIM map), so space-time arrays above this many voxels are
# scored in time chunks instead (about 2 GB per chunk at 2**24 voxels).
SSIM_CHUNK_VOXELS = 2 ** 24


def _ssim_settings(data_range, win_size):
    return dict(data_range=data_range, win_size=win_size, gaussian_weights=False,
                use_sample_covariance=True, K1=0.01, K2=0.03, channel_axis=None)


def _ssim(truth, prediction, data_range, win_size, chunk_voxels=None):
    # Do not silently change the window between sequences or pad short axes.
    if min(truth.shape) < win_size:
        return None
    settings = _ssim_settings(data_range, win_size)
    chunk_voxels = SSIM_CHUNK_VOXELS if chunk_voxels is None else chunk_voxels
    if truth.ndim < 3 or truth.size <= chunk_voxels:
        return float(structural_similarity(truth, prediction, **settings))
    return _time_chunked_ssim(truth, prediction, settings, chunk_voxels)


def spatial_ssim(truth, prediction, *, data_range, win_size=7):
    """SSIM of one frame (a 2D image or a 3D volume) with the settings above; None if too small."""
    truth, prediction = _pair(truth, prediction)
    _check_range(data_range)
    return _ssim(truth, prediction, data_range, win_size)


def _time_chunked_ssim(truth, prediction, settings, chunk_voxels):
    """Whole-array SSIM computed over chunks of the last (time) axis.

    skimage averages the local SSIM map over every window inside the array: the map
    cropped by pad = (win_size - 1) / 2 on all sides. The map at frame k depends on
    frames k - pad .. k + pad only, so the kept frames [start, stop) of a chunk are
    computed from frames [start - pad, stop + pad), where chunk-edge reflection
    cannot reach them. Summing the kept map over all chunks gives the whole-array
    mean, equal up to floating-point rounding (skimage's running-sum filter and the
    order of summation differ).
    """
    pad = (settings["win_size"] - 1) // 2
    frames = truth.shape[-1]
    kept_per_chunk = max(1, chunk_voxels // (truth.size // frames) - 2 * pad)
    spatial = (slice(pad, -pad),) * (truth.ndim - 1)
    total, count = 0., 0
    for start in range(pad, frames - pad, kept_per_chunk):
        stop = min(start + kept_per_chunk, frames - pad)
        _, local = structural_similarity(truth[..., start - pad:stop + pad],
                                         prediction[..., start - pad:stop + pad], full=True, **settings)
        kept = local[spatial + (slice(pad, pad + stop - start),)]
        total += float(kept.sum(dtype=np.float64))
        count += kept.size
    return total / count


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


def _sequence_settings(input_shape, time_axis, data_range, ssim_win_size, size):
    settings = {
        "input_shape": list(input_shape), "time_axis": time_axis,
        "data_range": float(data_range), "prediction_clipping": False,
        "ssim": {"scope": "whole spatiotemporal array; separate spatial per-frame scores",
                 "win_size": int(ssim_win_size), "gaussian_weights": False,
                 "use_sample_covariance": True, "K1": 0.01, "K2": 0.03,
                 "channel_axis": None, "short_axes": "null; window is not changed",
                 "skimage_version": skimage.__version__},
        "roi": None,
    }
    if size > SSIM_CHUNK_VOXELS:
        settings["ssim"]["computation"] = (
            f"arrays above {SSIM_CHUNK_VOXELS} voxels are scored in time chunks with "
            f"{(ssim_win_size - 1) // 2}-frame halos; equal to the whole-array value up to rounding")
    return settings


def _normalize_roi(roi, spatial_shape):
    if len(roi) != len(spatial_shape):
        raise ValueError("roi must contain one slice per spatial dimension")
    normalized = []
    for item, size in zip(roi, spatial_shape):
        if not isinstance(item, slice) or item.step not in (None, 1):
            raise ValueError("ROI slices must be contiguous")
        start = 0 if item.start is None else item.start
        stop = size if item.stop is None else item.stop
        if not all(isinstance(v, (int, np.integer)) for v in (start, stop)) or not 0 <= start < stop <= size:
            raise ValueError("ROI bounds must be nonempty and within the spatial grid")
        normalized.append(slice(int(start), int(stop)))
    return tuple(normalized)


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
    result["settings"] = _sequence_settings(input_shape, time_axis, data_range, ssim_win_size, truth.size)
    if roi is not None:
        roi = _normalize_roi(roi, truth.shape[:-1])
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


def evaluate_sequence_in_blocks(truth, predict_frames, *, data_range, times, roi=None,
                                ssim_win_size=7, block_frames=None):
    """evaluate_sequence for sequences too large to score at once, computed frame block by block.

    truth: (*space, time) array; predict_frames(frames) returns the prediction of the
    frames in the slice `frames`, shaped (*space, frames). The prediction is never held
    whole, and only one block (plus SSIM halos) is converted to float64 at a time.
    Returns the report of evaluate_sequence(truth, prediction, data_range=..., times=...,
    roi=roi, points={}) with time last and the ROI as the activity mask:
    - pooled errors from sums of squared error and squared truth over all blocks;
    - per-frame scores frame by frame, exactly as there;
    - whole-array SSIM by the time-chunk scheme of _time_chunked_ssim: the local SSIM
      map of a block's kept frames, computed from inputs with pad-frame halos.
    The values equal evaluate_sequence up to floating-point rounding (summation order).
    """
    _check_range(data_range)
    if truth.ndim not in (3, 4):
        raise ValueError("Expected a scalar 2D+t or 3D+t sequence")
    if not isinstance(ssim_win_size, (int, np.integer)) or ssim_win_size < 3 or ssim_win_size % 2 != 1:
        raise ValueError("ssim_win_size must be an odd integer >= 3")
    count, spatial_shape = truth.shape[-1], truth.shape[:-1]
    times = np.asarray(times, dtype=float)
    if times.shape != (count,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite, strictly increasing, and match frame count")
    pad = (ssim_win_size - 1) // 2
    voxels = math.prod(spatial_shape)
    block_frames = block_frames or max(1, SSIM_CHUNK_VOXELS // voxels)
    regions = {"full": tuple(slice(None) for _ in spatial_shape)}
    if roi is not None:
        regions["roi"] = _normalize_roi(roi, spatial_shape)
    settings = _ssim_settings(data_range, ssim_win_size)
    totals = {name: dict(squared_error=0., squared_truth=0., samples=0, max_abs_error=0., ssim_sum=0., ssim_count=0)
              for name in regions}
    frames = {name: [] for name in regions}
    curves = dict(truth=[], prediction=[])
    spatial_axes = tuple(range(len(spatial_shape)))
    crop = (slice(pad, -pad),) * len(spatial_shape)

    for start in range(0, count, block_frames):
        stop = min(start + block_frames, count)
        low, high = max(0, start - pad), min(count, stop + pad)  # with SSIM halos
        observed, predicted = _pair(truth[..., low:high], predict_frames(slice(low, high)))
        inner = slice(start - low, stop - low)
        for name, region in regions.items():
            t, p = observed[region], predicted[region]
            t_inner, p_inner = t[..., inner], p[..., inner]
            error = p_inner - t_inner
            total = totals[name]
            total["squared_error"] += float(np.sum(error ** 2))
            total["squared_truth"] += float(np.sum(t_inner ** 2))
            total["samples"] += error.size
            total["max_abs_error"] = max(total["max_abs_error"], float(np.max(np.abs(error))))
            frames[name] += [_image_metrics(t_inner[..., k], p_inner[..., k], data_range, ssim_win_size)
                             for k in range(stop - start)]
            # Whole-array SSIM averages the local map over frames pad .. count - pad - 1.
            kept_start, kept_stop = max(start, pad), min(stop, count - pad)
            if min(t.shape[:-1] + (count,)) >= ssim_win_size and kept_start < kept_stop:
                window = slice(kept_start - pad - low, kept_stop + pad - low)
                _, local = structural_similarity(t[..., window], p[..., window], full=True, **settings)
                kept = local[crop + (slice(pad, pad + kept_stop - kept_start),)]
                total["ssim_sum"] += float(kept.sum(dtype=np.float64))
                total["ssim_count"] += kept.size
            if name == "roi":
                curves["truth"] += (t_inner.sum(axis=spatial_axes) / math.prod(t.shape[:-1])).tolist()
                curves["prediction"] += (p_inner.sum(axis=spatial_axes) / math.prod(t.shape[:-1])).tolist()

    def report(name):
        total = totals[name]
        mse = total["squared_error"] / total["samples"]
        rrmse = (float(np.sqrt(total["squared_error"] / total["squared_truth"])) if total["squared_truth"]
                 else float("nan"))
        result = {"mse": mse, "rmse": float(np.sqrt(mse)), "rrmse": rrmse, "relative_l2": rrmse,
                  "psnr_db": float(20 * np.log10(data_range) - 10 * np.log10(mse)) if mse else float("inf"),
                  "max_abs_error": total["max_abs_error"],
                  "ssim": total["ssim_sum"] / total["ssim_count"] if total["ssim_count"] else None}
        for key in frames[name][0]:
            result[f"per_frame_{key}"] = [frame[key] for frame in frames[name]]
        return result

    result = report("full")
    result["times"] = times.tolist()
    result["settings"] = _sequence_settings(truth.shape, truth.ndim - 1, data_range, ssim_win_size, truth.size)
    result["settings"]["computation"] = (
        f"streamed in blocks of {block_frames} frames with {pad}-frame halos: pooled sums, per-frame scores "
        f"and the time-chunked SSIM; equal to the whole-array evaluation up to rounding")
    if roi is not None:
        result["settings"]["roi"] = [[s.start, s.stop] for s in regions["roi"]]
        result["roi"] = report("roi")
        result["activity_curve"] = {
            "mask_kind": "fixed spatial mask",
            "pixel_counts": [math.prod(s.stop - s.start for s in regions["roi"])] * count,
            "truth": curves["truth"], "prediction": curves["prediction"],
            **error_metrics(curves["truth"], curves["prediction"], data_range=data_range),
        }
    result["point_curves"] = {}
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
