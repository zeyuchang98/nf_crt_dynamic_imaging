"""Nested MLP: image(r, t) = intensity(r + motion(r, t), t).

Read NestedMLP.forward first. The two component networks are defined below.
Inputs are (N, 3) rows of (x, y, t); the output is one intensity per row.
"""
import torch
from torch import nn

from .coordinates import CoordinateNormalizer, DEFAULT_BOUNDS


class NestedMLP(nn.Module):
    def __init__(self, intensity_width=128, motion_width=32,
                 coordinate_bounds=DEFAULT_BOUNDS, reference_time=None):
        super().__init__()
        self.intensity = IntensityMLP(intensity_width, coordinate_bounds)
        self.motion = MotionMLP(motion_width, reference_time, coordinate_bounds)

    def forward(self, xyt):
        # 1. Predict backward displacement (ux, uy), in the same units as (x, y).
        displacement = self.motion(xyt)

        # 2. Move the spatial query to r + u. Keep its time coordinate unchanged.
        reference_xyt = xyt.clone()
        reference_xyt[:, :2] += displacement

        # 3. Query intensity at the displaced position and the original time.
        return self.intensity(reference_xyt)


class IntensityMLP(nn.Module):
    """Predict C(qx, qy, t). Time allows intrinsic brightness changes."""

    def __init__(self, width=128, coordinate_bounds=DEFAULT_BOUNDS):
        super().__init__()
        self.normalizer = CoordinateNormalizer(coordinate_bounds)
        self.net = nn.Sequential(
            nn.Linear(3, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, xyt):
        # Map the fixed dataset bounds to [-1, 1]. Do not clip displaced queries.
        scaled_xyt = self.normalizer.to_signed(xyt)
        intensity = self.net(scaled_xyt)
        return intensity.squeeze(-1)


class MotionMLP(nn.Module):
    """Predict u(r, t) = M(r, t) - M(r, reference_time)."""

    def __init__(self, width=32, reference_time=None, coordinate_bounds=DEFAULT_BOUNDS):
        super().__init__()
        self.normalizer = CoordinateNormalizer(coordinate_bounds)
        # The reference time uses the input time units; default to the lower bound.
        if reference_time is None:
            reference_time = coordinate_bounds[2][0]
        self.register_buffer("reference_time", torch.tensor(float(reference_time)))
        self.net = nn.Sequential(
            nn.Linear(3, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 2),
        )
        # Start training with zero displacement everywhere.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, xyt):
        # Compare the same spatial point at the current and reference times.
        reference_xyt = xyt.clone()
        reference_xyt[:, 2] = self.reference_time

        # Use the same coordinate scaling as the intensity MLP.
        current = self.net(self.normalizer.to_signed(xyt))
        reference = self.net(self.normalizer.to_signed(reference_xyt))

        # Anchor at the reference time, then convert fractions of the spatial
        # extent back to input units, so displacement can be added to (x, y).
        return (current - reference) * self.normalizer.span[:2]
