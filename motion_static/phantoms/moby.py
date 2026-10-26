"""Load a time window of a DCE-MOBY z-slice file (phantoms/moby_slice.py) for training.

The window is returned in the toy-phantom layout: `truth` is (y, x, t) with pixel
coordinates x, y in mm and t in seconds. Models normalize these physical coordinates
with `coordinate_bounds`, so no manual coordinate scaling is applied here. Only the
intensity is normalized, by one fixed scale for the whole series.

Even frames of the window are observed; odd frames are the held-out midpoints, as in
the toy `midpoints` protocol. The window therefore starts and ends on observed frames.
With `held_out_split = "alternate_validation_test"` the midpoints are further divided
into validation frames (which may steer training) and test frames (only scored).
With `held_out_split = "none"` (the embedding protocol of Lozenski et al. 2022) every
window frame is fitted and scored.
"""
import hashlib
from pathlib import Path

import h5py
import numpy as np


NAME = "dce_moby"


def global_maximum(dataset, block=500):
    """Maximum over all frames, read in blocks so the full series is never in memory."""
    return max(float(dataset[..., start:start + block].max())
               for start in range(0, dataset.shape[-1], block))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def window_frames(data):
    """Source frame indices: `cycles` periods of `frames_per_cycle` frames, both ends included."""
    if data["frames_per_cycle"] % 2:
        raise ValueError("frames_per_cycle must be even so the window ends on an observed frame")
    return np.arange(data["first_frame"], data["first_frame"] + data["cycles"] * data["frames_per_cycle"] + 1)


def pixel_centers(fov, spacing, count):
    """Voxel centers inside the field of view, whose extent is count * spacing."""
    if not np.isclose(fov[1] - fov[0], count * spacing):
        raise ValueError("Field of view does not equal pixel count times spacing")
    return fov[0] + (np.arange(count) + 0.5) * spacing


def resolve_settings(data):
    """Fill in everything fixed by the file and window; store it in the run config.

    Returns (data settings, run settings). The intensity scale is stored as a number,
    so later inference reuses exactly the training normalization.
    """
    data = dict(data, file=str(Path(data["file"]).expanduser().resolve()))
    frames = window_frames(data)
    with h5py.File(data["file"], "r") as f:
        p0, time = f["p0"], f["time"]
        if frames[-1] >= p0.shape[-1]:
            raise ValueError(f"Window ends at frame {frames[-1]}, beyond the {p0.shape[-1]} frames in the file")
        fov, spacing = np.asarray(f.attrs["fov"], dtype=float), np.asarray(f.attrs["spacing"], dtype=float)
        if p0.shape[0] != p0.shape[1] or not np.isclose(spacing[0], spacing[1]):
            raise ValueError("Expected a square grid with equal x and y spacing")
        if data.get("intensity_scale", "global_maximum") == "global_maximum":
            data["intensity_scale"] = "global_maximum"
            data["intensity_scale_value"] = global_maximum(p0)
        else:
            data["intensity_scale_value"] = float(data["intensity_scale"])
        t_bounds = [float(time[frames[0]]), float(time[frames[-1]])]
        size = int(p0.shape[0])
        data.update(z_index=int(f.attrs["z_index"]), spacing_mm=float(spacing[0]))
    data["sha256"] = file_sha256(data["file"])
    run = dict(size=size, frames=len(frames[::2]), cycles=data["cycles"],
               joint_period=(t_bounds[1] - t_bounds[0]) / data["cycles"],
               coordinate_bounds=[fov[0].tolist(), fov[1].tolist(), t_bounds])
    return data, run


def read_window(data):
    """Every frame of the window: pixel centers x, y, times and normalized (y, x, t) truth."""
    frames = window_frames(data)
    with h5py.File(data["file"], "r") as f:
        values = f["p0"][..., frames[0]:frames[-1] + 1]  # (x, y, t)
        time = f["time"][frames[0]:frames[-1] + 1].astype(float)
        fov, spacing = np.asarray(f.attrs["fov"], dtype=float), np.asarray(f.attrs["spacing"], dtype=float)
    x = pixel_centers(fov[0], spacing[0], values.shape[0])
    y = pixel_centers(fov[1], spacing[1], values.shape[1])
    truth = np.transpose(values, (1, 0, 2)) / np.float32(data["intensity_scale_value"])
    return x, y, time, truth


def make_pair(data):
    """Observed (even) and held-out (odd) frames of the window, normalized intensity."""
    x, y, time, truth = read_window(data)
    return {split: dict(x=x, y=y, t=time[start::2], truth=np.ascontiguousarray(truth[..., start::2]))
            for split, start in (("training", 0), ("midpoint", 1))}


def make_all_frames(data):
    """Embedding protocol: every window frame is observed and scored; nothing is held out."""
    x, y, time, truth = read_window(data)
    return {"training": dict(x=x, y=y, t=time, truth=np.ascontiguousarray(truth))}


def split_midpoints(pair):
    """Alternate the held-out midpoints into validation and test frames.

    Midpoint j is window frame 2j + 1. Even j (window frames 1, 5, 9, ...) are
    validation frames, odd j (3, 7, 11, ...) test frames, so both cover the whole
    window and every breathing phase. Training frames are unchanged.
    """
    midpoint = pair["midpoint"]
    split = {"training": pair["training"]}
    for name, start in (("validation", 0), ("test", 1)):
        split[name] = dict(x=midpoint["x"], y=midpoint["y"], t=midpoint["t"][start::2],
                           truth=np.ascontiguousarray(midpoint["truth"][..., start::2]))
    return split
