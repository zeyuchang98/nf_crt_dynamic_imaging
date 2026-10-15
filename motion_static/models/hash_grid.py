"""The grid encoder used by HashNeuralField.

Each level stores trainable features on a 2D spatial or 3D space-time grid.
A query blends the surrounding vertices, then joins features from all levels.
Small grids use direct indexing; large grids share entries through hashing.

Based on Instant-NGP: https://nvlabs.github.io/instant-ngp/
This is a PyTorch implementation, not NVIDIA's fused CUDA implementation.
"""
import itertools
import math
from numbers import Integral

import torch
from torch import nn


class HashGridEncoding(nn.Module):
    """Input: (N, dimensions) coordinates. Output: (N, levels * features).

    Coordinates normally lie in [0, 1]. The nested intensity field sets
    clamp_coordinates=False so motion can move queries outside the image.
    """

    def __init__(self, levels=8, features=2, log2_hash_size=15,
                 min_resolution=8, max_resolution=128, clamp_coordinates=True,
                 dimensions=3):
        super().__init__()
        if dimensions not in (2, 3):
            raise ValueError("Hash grids support two spatial or three space-time coordinates")
        self.dimensions = dimensions
        self.output_dim = levels * features
        self.clamp_coordinates = clamp_coordinates

        # Per-axis limits let motion use coarser space without coarsening time.
        # Keep scalar resolution buffers unchanged for existing checkpoints.
        minimum = self._resolution_axes(min_resolution, dimensions)
        maximum = self._resolution_axes(max_resolution, dimensions)
        axis_levels = []
        for low, high in zip(minimum, maximum):
            if high < low:
                raise ValueError("max_resolution must be at least min_resolution on every axis")
            growth = (high / low) ** (1 / max(levels - 1, 1))
            values = [math.floor(low * growth ** i) for i in range(levels)]
            if levels > 1:
                values[-1] = high
            axis_levels.append(values)
        per_axis = not (isinstance(min_resolution, Integral) and isinstance(max_resolution, Integral))
        resolutions = list(zip(*axis_levels)) if per_axis else axis_levels[0]

        # All levels share one tensor, but each level gets its own table section.
        capacity = 2 ** log2_hash_size
        vertex_counts = [math.prod(value + 1 for value in row) for row in zip(*axis_levels)]
        sizes = [min(capacity, count) for count in vertex_counts]
        offsets = [sum(sizes[:level]) for level in range(levels)]
        dense = [count <= capacity for count in vertex_counts]
        corners = list(itertools.product((0, 1), repeat=dimensions))

        # Buffers move with the model and are saved, but are not learned.
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.long))
        self.register_buffer("sizes", torch.tensor(sizes, dtype=torch.long))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.register_buffer("dense", torch.tensor(dense))
        self.register_buffer("corners", torch.tensor(corners))

        # These are the trainable features read by the decoder MLP.
        self.table = nn.Parameter(torch.empty(sum(sizes), features))
        nn.init.uniform_(self.table, -1e-4, 1e-4)

    @staticmethod
    def _resolution_axes(value, dimensions):
        values = [value] * dimensions if isinstance(value, Integral) else list(value)
        if len(values) != dimensions or any(
                not isinstance(item, Integral) or isinstance(item, bool) or item < 1 for item in values):
            raise ValueError("Resolution must be a positive integer or one positive integer per axis")
        return values

    def forward(self, coordinates):
        # Tensor axes: N queries, L levels, K=2^D corners, D coordinates.
        if self.clamp_coordinates:
            coordinates = coordinates.clamp(0, 1)

        # 1. Find each query's cell and its fractional position inside the cell.
        resolution = (self.resolutions[None, :, None] if self.resolutions.ndim == 1
                      else self.resolutions[None, :, :])  # (1, L, 1 or D)
        position = coordinates[:, None, :] * resolution   # (N, L, D)
        lower = position.floor().long()
        if self.clamp_coordinates:
            # Coordinate 1 belongs to the last cell, with fraction 1.
            lower = torch.minimum(lower, resolution - 1)
        fraction = position - lower
        vertices = lower[:, :, None, :] + self.corners    # (N, L, K, D)

        # 2. Read the feature vector at each corner.
        indices = self._table_indices(vertices)
        corner_features = self.table[indices]            # (N, L, K, features)

        # 3. Bilinear/trilinear interpolation: multiply weights over coordinates.
        fraction = fraction[:, :, None, :]               # (N, L, 1, D)
        axis_weights = torch.where(self.corners.bool(), fraction, 1 - fraction)
        weights = axis_weights.prod(dim=-1)              # (N, L, K)
        level_features = (corner_features * weights[..., None]).sum(dim=2)

        # 4. Join coarse and fine features into one vector for the decoder.
        return level_features.flatten(start_dim=1)

    def _table_indices(self, vertices):
        """Map grid vertices to rows of the shared feature tensor."""
        x, y = vertices[..., 0], vertices[..., 1]
        resolution = (self.resolutions[:, None].expand(-1, self.dimensions)
                      if self.resolutions.ndim == 1 else self.resolutions)
        sides = resolution + 1
        side_x, side_y = sides[None, :, 0, None], sides[None, :, 1, None]

        # Small grids fit in full. Large grids use a bounded XOR hash.
        dense_index = x + side_x * y
        hash_index = x ^ (y * 2654435761)
        if self.dimensions == 3:
            t = vertices[..., 2]
            dense_index = x + side_x * (y + side_y * t)
            hash_index = hash_index ^ (t * 805459861)
        hash_index = hash_index.remainder(self.sizes[None, :, None])
        use_dense = self.dense[None, :, None]

        if not self.clamp_coordinates:
            # Warped queries can leave the image. Hash outside vertices, while
            # keeping inside indices unchanged so interpolation stays continuous.
            inside = ((vertices >= 0) & (vertices <= resolution[None, :, None, :])).all(dim=-1)
            use_dense = use_dense & inside

        local_index = torch.where(use_dense, dense_index, hash_index)
        return local_index + self.offsets[None, :, None]
