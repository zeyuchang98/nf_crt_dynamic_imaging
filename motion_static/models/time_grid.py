"""Optional time network for low-rank intensity: 1D multi-resolution grid + MLP.

a(t) = MLP(concat_l interp_l(t)), with R outputs like the default Tanh MLP.
Each level stores features at equally spaced time nodes and linearly
interpolates the two nodes around a query, so a node only affects nearby
times. The finest resolution bounds how fast a(t) can change; there is no
periodicity assumption. 1D tables are small, so no hashing is needed.
"""
import math

import torch
from torch import nn


class TimeGridEncoding(nn.Module):
    """Input: (N, 1) unit time in [0, 1]. Output: (N, levels * features)."""

    def __init__(self, levels, features, min_resolution, max_resolution):
        super().__init__()
        if max_resolution < min_resolution:
            raise ValueError("max_resolution must be at least min_resolution")
        # Same geometric level spacing as HashGridEncoding; resolution counts
        # cells over the whole time span, so a level has resolution + 1 nodes.
        growth = (max_resolution / min_resolution) ** (1 / max(levels - 1, 1))
        resolutions = [math.floor(min_resolution * growth ** i) for i in range(levels)]
        if levels > 1:
            resolutions[-1] = max_resolution
        nodes = [value + 1 for value in resolutions]
        self.output_dim = levels * features
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.long))
        self.register_buffer("offsets", torch.tensor([sum(nodes[:i]) for i in range(levels)]))
        self.table = nn.Parameter(torch.empty(sum(nodes), features))
        nn.init.uniform_(self.table, -1e-4, 1e-4)

    def forward(self, unit_time):
        position = unit_time.clamp(0, 1) * self.resolutions        # (N, L)
        # Time 1 belongs to the last cell, with fraction 1.
        lower = torch.minimum(position.floor().long(), self.resolutions - 1)
        fraction = (position - lower)[..., None]                   # (N, L, 1)
        left = self.table[lower + self.offsets]                    # (N, L, F)
        right = self.table[lower + 1 + self.offsets]
        return ((1 - fraction) * left + fraction * right).flatten(start_dim=1)


class TimeGridField(nn.Module):
    """Signed time in [-1, 1] -> R temporal coefficients, as the default MLP."""

    def __init__(self, rank, width, levels, features, min_resolution, max_resolution):
        super().__init__()
        self.encoding = TimeGridEncoding(levels, features, min_resolution, max_resolution)
        # The decoder mirrors the spatial hash decoder.
        self.mlp = nn.Sequential(
            nn.Linear(self.encoding.output_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, rank),
        )

    def forward(self, signed_time):
        return self.mlp(self.encoding((signed_time + 1) / 2))
