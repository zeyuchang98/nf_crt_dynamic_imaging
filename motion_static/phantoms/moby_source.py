"""Read DCE-MOBY p0 frames directly from the source phase files: the whole volume or a box.

One reader for both: a box is only a region of every frame. Source layout
(phantoms/moby_slice.py): `p0/p0_XXXXX.h5` holds one respiratory phase with `p0`
shaped (x, y, z, frame); the series is all frames of all phases sorted by time, the
order moby_slice.extract_box writes. Reading one frame decompresses only the h5
chunks that overlap the region (the source chunks are (62, 62, 50, 1)).

A frame is returned in the training layout of moby.read_window_entry: (y, x, z),
divided by one fixed intensity scale. A box read here equals the same frame of an
extracted box file bit for bit, and its geometry (fov, pixel centers, times) is
computed the same way, so both give the same coordinates and targets.
"""
from pathlib import Path

import h5py
import numpy as np

from .moby import file_sha256, pixel_centers, window_frames
from .moby_slice import find_phases, frame_order


class SourceFrames:
    """The time-sorted frames of the source phase files, restricted to a region.

    box_start, box_shape: (x, y, z) first voxel and size in the source volume; both
    None for the whole volume. Files stay open until close() (or the end of a with block).
    """

    def __init__(self, root, box_start=None, box_shape=None):
        self.root = Path(root)
        files = find_phases(self.root / "p0", "p0", "p0", "resp_phase", None)
        info = frame_order(files)
        self.paths = list(files.values())
        file_of_phase = {phase: k for k, phase in enumerate(files)}
        order = info["order"]
        self.file_of_frame = np.array([file_of_phase[phase] for phase in info["phase"][order]])
        self.index_in_file = info["index"][order]
        self.times = info["times"][order].astype(float)
        self.phases = info["phase"][order]
        self.source_shape = tuple(int(n) for n in info["shape"])
        source_fov = np.asarray(info["geometry"]["fov"], dtype=float)
        self.spacing = np.asarray(info["geometry"]["spacing"], dtype=float)
        self.spacing_units = info["geometry"]["spacing_units"]

        if (box_start is None) != (box_shape is None):
            raise ValueError("Give both box_start and box_shape, or neither for the whole volume")
        if box_start is None:
            start, size = np.zeros(3, int), np.array(self.source_shape)
            self.fov = source_fov
        else:
            start, size = np.asarray(box_start, int), np.asarray(box_shape, int)
            if (start.shape != (3,) or size.shape != (3,) or np.any(start < 0) or np.any(size < 1)
                    or np.any(start + size > np.array(self.source_shape))):
                raise ValueError(f"Box start {start.tolist()} and shape {size.tolist()} do not fit in "
                                 f"{list(self.source_shape)}")
            # The box's own extent, computed as moby_slice.extract_box stores it.
            self.fov = np.stack([source_fov[:, 0] + start * self.spacing,
                                 source_fov[:, 0] + (start + size) * self.spacing], 1)
        self.box_start, self.shape = [int(v) for v in start], tuple(int(v) for v in size)
        self.region = tuple(slice(int(a), int(a + n)) for a, n in zip(start, size))
        self.x, self.y, self.z = (pixel_centers(self.fov[axis], self.spacing[axis], self.shape[axis])
                                  for axis in range(3))
        self._handles = {}

    @property
    def frame_count(self):
        return len(self.times)

    def read_raw(self, frame):
        """p0 of one series frame inside the region, (x, y, z) float32, as stored."""
        k = int(self.file_of_frame[frame])
        if k not in self._handles:
            self._handles[k] = h5py.File(self.paths[k], "r")
        index = int(self.index_in_file[frame])
        # Keep the frame axis (size 1): an h5py selection that drops the last axis
        # is read element by element and is about ten times slower.
        return self._handles[k]["p0"][self.region + (slice(index, index + 1),)][..., 0]

    def read(self, frame, scale):
        """One series frame in the training layout: (y, x, z) float32 divided by `scale`.

        C-contiguous, like the in-memory window, so numpy sums over it in the same order."""
        return np.ascontiguousarray(np.transpose(self.read_raw(frame), (1, 0, 2))) / np.float32(scale)

    def sha256(self, paths=None):
        """sha256 of the given source files (default all), keyed by file name."""
        return {Path(path).name: file_sha256(path) for path in (self.paths if paths is None else paths)}

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_source(data):
    """The SourceFrames of a data block: `source_root`, optional `box_start` and `box_shape`."""
    return SourceFrames(data["source_root"], data.get("box_start"), data.get("box_shape"))


def resolve_settings(data):
    """moby.resolve_settings for source data, from file metadata only.

    Returns (data settings, run settings) with the same run settings as an extracted file
    of the same region. The `global_maximum` scale needs a pass over every frame, so it is
    filled in later by the training run (run.execute_streamed); a numeric scale is stored here.
    """
    if "file" in data:
        raise ValueError("Give either data.file (an extracted file) or data.source_root, not both")
    data = dict(data, source_root=str(Path(data["source_root"]).expanduser().resolve()))
    frames = window_frames(data)
    with open_source(data) as source:
        if frames[-1] >= source.frame_count:
            raise ValueError(f"Window ends at frame {frames[-1]}, beyond the {source.frame_count} source frames")
        if source.shape[0] != source.shape[1] or not np.isclose(source.spacing[0], source.spacing[1]):
            raise ValueError("Expected a square grid with equal x and y spacing")
        if not np.isclose(source.spacing[2], source.spacing[0]):
            raise ValueError("Expected equal x, y and z spacing")
        t_bounds = [float(source.times[frames[0]]), float(source.times[frames[-1]])]
        data.update(box_start=source.box_start, box_shape=list(source.shape),
                    source_shape=list(source.source_shape), source_frames=source.frame_count,
                    spacing_mm=float(source.spacing[0]))
        fov = source.fov
    if data.get("intensity_scale", "global_maximum") == "global_maximum":
        data["intensity_scale"] = "global_maximum"
        data.pop("intensity_scale_value", None)
    else:
        data["intensity_scale_value"] = float(data["intensity_scale"])
    run = dict(size=int(source.shape[0]), frames=len(frames[::2]), cycles=data["cycles"],
               joint_period=(t_bounds[1] - t_bounds[0]) / data["cycles"],
               coordinate_bounds=[fov[0].tolist(), fov[1].tolist(), fov[2].tolist(), t_bounds],
               spatial_dimensions=3, depth=int(source.shape[2]))
    return data, run


def read_window_entry(data):
    """moby.read_window_entry for source data: every window frame in memory, (y, x, z, t).

    For post-hoc tools on windows that fit in memory; training streams frames instead.
    """
    frames = window_frames(data)
    with open_source(data) as source:
        truth = np.empty((source.shape[1], source.shape[0], source.shape[2], len(frames)), dtype=np.float32)
        for k, frame in enumerate(frames):
            truth[..., k] = source.read(frame, data["intensity_scale_value"])
        return dict(x=source.x, y=source.y, z=source.z, t=source.times[frames], truth=truth)

