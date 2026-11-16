"""Batch sources for run.train: the whole pool in memory, or frames streamed from h5.

Objective (unchanged): L = (1/M) sum_i l_i over all M = V F points of the window
(V voxels per frame, F frames).

Streamed training with P processes (P = 1 for one process):
- Frame shards: process p owns the contiguous window frames S_p (sizes differ by at
  most one) and has weight w_p = |S_p| / F, its share of the M points.
- Frame groups: when a shard does not fit in memory, each round shuffles S_p and splits
  it into n groups of near-equal size (n is the same on every process); the groups are
  loaded one after another and each is trained for `steps_per_group` steps. Without
  grouping, n = 1 and the whole shard stays loaded.
- Each step, process p draws B / P points uniformly with replacement from the points
  of its loaded group; the step gradient is g = sum_p w_p grad(batch mean loss of p),
  the gradients being summed over processes by an all-reduce (run.train).
Given the loaded groups G_p, E[g] = sum_p w_p grad L(G_p): biased toward the loaded
frames within a group, but over a round every frame is loaded once for the same number
of steps, so the round average is grad L (up to group sizes differing by one frame).
With n = 1 the step gradient is unbiased, with a variance no larger than one process's
(stratified sampling).

One process with the whole window loaded reproduces InMemorySampler draw for draw:
same buffer order (frame fastest), same random indices, same batches.

Support mask (optional): the points are only the mask voxels times the frames; the
mask is the voxels with p0 > 0 in any window frame, so the objective is the mean over
these points, and the buffers and batches hold mask voxels only. With every voxel in the
mask nothing changes.
"""
import math

import numpy as np
import torch

from .phantoms.data import CoordinateGrid, VoxelSubsetGrid


class InMemorySampler:
    """The whole observed pool on the device; batches with replacement (the original training)."""

    def __init__(self, coordinates, intensities, batch_size, device):
        if isinstance(coordinates, CoordinateGrid):
            self.coordinates = coordinates.to(device)  # rows computed per batch (3D grids)
        else:
            self.coordinates = torch.as_tensor(coordinates, dtype=torch.float32, device=device)
        self.values = torch.as_tensor(intensities.reshape(-1), dtype=torch.float32, device=device)
        self.batch_size, self.device = batch_size, device

    def batch(self, step):
        # The CPU generator draws the indices on every device, so a seed selects the same batches.
        indices = torch.randint(len(self.values), (self.batch_size,)).to(self.device)
        return self.coordinates[indices], self.values[indices]


def frame_shards(frame_count, world_size):
    """Contiguous window-frame positions of each process; sizes differ by at most one."""
    if not 1 <= world_size <= frame_count:
        raise ValueError(f"Cannot split {frame_count} frames over {world_size} processes")
    return [np.asarray(shard) for shard in np.array_split(np.arange(frame_count), world_size)]


def groups_per_round(shards, frames_per_group):
    """Groups per round, the same on every process (set by the largest shard); 1 without grouping."""
    if frames_per_group is None:
        return 1
    count = math.ceil(max(len(shard) for shard in shards) / frames_per_group)
    if count > min(len(shard) for shard in shards):
        raise ValueError(f"{count} groups per round leave some group of the smallest shard empty")
    return count


def round_groups(shard, count, generator):
    """One round's groups: a random partition of the shard into `count` groups whose sizes
    differ by at most one, each in frame order. A single group is the whole shard in order."""
    if count == 1:
        return [shard]
    order = generator.permutation(shard)
    return [np.sort(group) for group in np.array_split(order, count)]


class StreamedSampler:
    """One process's batches from the frames it holds in memory, loaded group by group.

    read_frame(position) returns window frame `position` as a flat (V,) float32 array in
    the training layout. The loaded values are (V, frames of the group): row v, column j
    is voxel v of the group's j-th frame, so flat index v * G + j follows the in-memory
    pool's order (frame fastest) and CoordinateGrid gives the coordinates directly.
    generator: torch generator for the batch indices (None: the default generator).
    group_generator: numpy generator for the round partitions.
    voxels: flat (y, x, z) indices of the support mask, ascending (None: every voxel); row v
    of the loaded values is then mask voxel voxels[v].
    """

    def __init__(self, read_frame, shard, axes, times, batch_size, groups, steps_per_group,
                 generator, group_generator, device, voxels=None):
        self.read_frame, self.shard = read_frame, np.asarray(shard)
        self.x, self.y, self.z = axes
        self.voxels = None if voxels is None else np.asarray(voxels)
        self.voxel_count = len(self.x) * len(self.y) * len(self.z) if voxels is None else len(self.voxels)
        self.times = np.asarray(times)
        self.batch_size, self.groups, self.steps_per_group = batch_size, groups, steps_per_group
        self.generator, self.group_generator, self.device = generator, group_generator, device
        self.pending, self.loaded = [], None
        self.values = self.coordinates = None

    def load(self, group):
        """Read the group's frames into memory, one frame at a time."""
        self.values = None  # release the previous group before allocating the next
        values = torch.empty((self.voxel_count, len(group)), dtype=torch.float32, device=self.device)
        for j, position in enumerate(group):
            frame = np.ascontiguousarray(self.read_frame(position)).reshape(-1)
            values[:, j] = torch.from_numpy(frame if self.voxels is None else frame[self.voxels])
        self.values, self.loaded = values, np.asarray(group)
        if self.voxels is None:
            self.coordinates = CoordinateGrid(self.x, self.y, self.times[group], self.z).to(self.device)
        else:
            self.coordinates = VoxelSubsetGrid(self.x, self.y, self.times[group], self.z, self.voxels).to(self.device)

    def batch(self, step):
        """Batch of 1-based optimizer step `step`; switches to the next group when it is due."""
        if self.loaded is None or (self.groups > 1 and (step - 1) % self.steps_per_group == 0):
            if not self.pending:
                self.pending = round_groups(self.shard, self.groups, self.group_generator)
            self.load(self.pending.pop(0))
        frames = len(self.loaded)
        indices = torch.randint(self.values.numel(), (self.batch_size,), generator=self.generator).to(self.device)
        return self.coordinates[indices], self.values[indices // frames, indices % frames]


def scan_shard(read_raw, positions, frame_count, sample_indices, voxels=None):
    """One pass over a shard's frames before training.

    read_raw(position) returns window frame `position` as stored, (x, y, z).
    Returns the maximum over these frames, the (y, x, z) voxels positive in any of them,
    and the raw values of the fixed samples (flat indices v * F + position of the in-memory
    pool, v the voxel or, with `voxels`, the mask voxel voxels[v]) that fall in these
    frames, zero elsewhere.
    """
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    sample_frame, sample_voxel = sample_indices % frame_count, sample_indices // frame_count
    if voxels is not None:
        sample_voxel = np.asarray(voxels)[sample_voxel]
    maximum, positive = -np.inf, None
    values = np.zeros(len(sample_indices), dtype=np.float32)
    for position in positions:
        frame = np.ascontiguousarray(np.transpose(read_raw(position), (1, 0, 2)))  # training layout (y, x, z)
        maximum = max(maximum, float(frame.max()))
        positive = frame > 0 if positive is None else positive | (frame > 0)
        selected = np.flatnonzero(sample_frame == position)
        values[selected] = frame.reshape(-1)[sample_voxel[selected]]
    return maximum, positive, values


class StreamedTruth:
    """(y, x, z, frame) window truth read on demand: truth[..., a:b] reads frames a .. b - 1.

    Enough of an array for metrics.sequence_block_partials, which only takes frame slices.
    read_frame(position) returns window frame `position` in the training layout (y, x, z).
    """

    def __init__(self, read_frame, spatial_shape, frame_count):
        self.read_frame = read_frame
        self.shape = (*spatial_shape, frame_count)
        self.ndim, self.size = len(self.shape), math.prod(self.shape)

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] is Ellipsis and isinstance(key[1], slice)):
            raise TypeError("StreamedTruth supports truth[..., first:stop] only")
        first, stop, step = key[1].indices(self.shape[-1])
        if step != 1:
            raise TypeError("StreamedTruth supports contiguous frame slices only")
        return np.stack([self.read_frame(position) for position in range(first, stop)], axis=-1)
