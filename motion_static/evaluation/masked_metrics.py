"""Score selected pixels; SSIM windows must stay entirely inside the selection."""
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity

from .metrics import error_metrics


def selected_errors(truth, prediction, mask, data_range):
    """Empty selections are undefined, not zero error."""
    if mask.any():
        return error_metrics(truth[mask], prediction[mask], data_range=data_range)
    return {key: None for key in ("mse", "rmse", "rrmse", "relative_l2", "psnr_db", "max_abs_error")}


def evaluate_masked_sequence(truth, prediction, mask, *, times, data_range=1,
                             roi=None, points=None, ssim_win_size=7):
    """Evaluate a (y, x, t) sequence without counting unselected observations.

    Overall SSIM is the mean over valid spatial window centers across frames.
    This differs from the spatiotemporal SSIM used by the midpoint experiment.
    """
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if truth.ndim != 3 or truth.shape != prediction.shape or mask.shape != truth.shape:
        raise ValueError("truth, prediction, and mask must share shape (y, x, t)")

    # The SSIM map is computed once; only centers with fully selected windows
    # contribute. No zero filling, and no train pixels inside test windows.
    ssim_map = np.empty_like(truth)
    for k in range(truth.shape[-1]):
        _, ssim_map[..., k] = structural_similarity(
            truth[..., k], prediction[..., k], data_range=data_range, win_size=ssim_win_size,
            gaussian_weights=False, use_sample_covariance=True, K1=0.01, K2=0.03,
            channel_axis=None, full=True)
    window = np.ones((ssim_win_size, ssim_win_size, 1), dtype=bool)

    def scope(selected):
        scores = selected_errors(truth, prediction, selected, data_range)
        valid = binary_erosion(selected, structure=window, border_value=0)
        scores["ssim"] = float(ssim_map[valid].mean()) if valid.any() else None
        scores["pixel_count"] = int(selected.sum())
        scores["ssim_window_count"] = int(valid.sum())
        frames = []
        for k in range(truth.shape[-1]):
            frame = selected_errors(truth[..., k], prediction[..., k], selected[..., k], data_range)
            centers = valid[..., k]
            frame["ssim"] = float(ssim_map[..., k][centers].mean()) if centers.any() else None
            frame["pixel_count"] = int(selected[..., k].sum())
            frame["ssim_window_count"] = int(centers.sum())
            frames.append(frame)
        for key in frames[0]:
            scores[f"per_frame_{key}"] = [frame[key] for frame in frames]
        return scores

    result = scope(mask)
    result["times"] = np.asarray(times).tolist()
    result["settings"] = dict(
        data_range=data_range, time_axis=2, prediction_clipping=False,
        roi=None if roi is None else [[s.start, s.stop] for s in roi],
        ssim=dict(win_size=ssim_win_size,
                  scope="Mean spatial SSIM over centers whose full window lies inside the evaluation mask",
                  gaussian_weights=False, use_sample_covariance=True, K1=0.01, K2=0.03))
    selected_roi = mask.copy()
    if roi is not None:
        spatial_roi = np.zeros(truth.shape[:2], dtype=bool)
        spatial_roi[roi] = True
        selected_roi &= spatial_roi[..., None]
        result["roi"] = scope(selected_roi)

    # ROI means use the same selected pixels in truth and prediction. This
    # subset changes with time and is not a fixed-region physiological curve.
    counts = selected_roi.sum(axis=(0, 1))
    actual = np.divide(np.where(selected_roi, truth, 0).sum(axis=(0, 1)), counts,
                       out=np.full(len(times), np.nan), where=counts > 0)
    estimated = np.divide(np.where(selected_roi, prediction, 0).sum(axis=(0, 1)), counts,
                          out=np.full(len(times), np.nan), where=counts > 0)
    result["activity_curve"] = dict(
        truth=actual.tolist(), prediction=estimated.tolist(), pixel_counts=counts.tolist(),
        evaluated=(counts > 0).tolist(), mask_kind="Selected ROI pixels; subset changes with time",
        **selected_errors(actual, estimated, counts > 0, data_range))
    result["point_curves"] = {}
    for name, point in (points or {}).items():
        point = tuple(point)
        result["point_curves"][name] = dict(
            spatial_index=list(point), truth=truth[point].tolist(), prediction=prediction[point].tolist(),
            evaluated=mask[point].tolist(),
            **selected_errors(truth[point], prediction[point], mask[point], data_range))
    return result
