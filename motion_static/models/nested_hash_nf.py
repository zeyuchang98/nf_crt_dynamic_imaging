"""Nested Hash NF: image(r, t) = intensity(r + motion(r, t), t).

Read NestedHashNF.forward first. Each branch has its own hash table and MLP.
The composition is the same as NestedMLP; only the component networks differ.
"""
import torch
from torch import nn

from .coordinates import DEFAULT_BOUNDS
from .hash_nf import HashNeuralField


class NestedHashNF(nn.Module):
    def __init__(self, width=64, coordinate_bounds=DEFAULT_BOUNDS, reference_time=None,
                 **encoding_kwargs):
        super().__init__()
        # The displaced query r + u can leave the image. Allow those queries so
        # intensity still supplies coordinate gradients to the motion network.
        self.intensity = HashNeuralField(
            width=width, coordinate_bounds=coordinate_bounds,
            clamp_coordinates=False, **encoding_kwargs)
        self.motion = MotionHashNF(
            width=width, coordinate_bounds=coordinate_bounds,
            reference_time=reference_time, **encoding_kwargs)

    def forward(self, xyt):
        # 1. Predict backward displacement (ux, uy), in coordinate units.
        displacement = self.motion(xyt)

        # 2. Shift (x, y) by the displacement; keep t unchanged.
        reference_xyt = xyt.clone()
        reference_xyt[:, :2] += displacement

        # 3. Query the independent intensity field at (r + u, t).
        return self.intensity(reference_xyt)


class MotionHashNF(nn.Module):
    """Predict u(r, t) = M(r, t) - M(r, reference_time)."""

    def __init__(self, width=64, reference_time=None, coordinate_bounds=DEFAULT_BOUNDS,
                 **encoding_kwargs):
        super().__init__()
        # Use the same time units as the input, not a normalized reference time.
        if reference_time is None:
            reference_time = coordinate_bounds[2][0]
        self.register_buffer("reference_time", torch.tensor(float(reference_time)))
        self.field = HashNeuralField(
            width=width, output_dim=2, coordinate_bounds=coordinate_bounds, **encoding_kwargs)

        # The two outputs are ux and uy. Start both at zero.
        nn.init.zeros_(self.field.mlp[-1].weight)
        nn.init.zeros_(self.field.mlp[-1].bias)

    def forward(self, xyt):
        # Compare the same spatial point at two times, using the same network.
        reference_xyt = xyt.clone()
        reference_xyt[:, 2] = self.reference_time
        current = self.field(xyt)
        reference = self.field(reference_xyt)

        # Anchor motion, then convert relative displacement to input (x, y) units.
        return (current - reference) * self.field.normalizer.span[:2]
