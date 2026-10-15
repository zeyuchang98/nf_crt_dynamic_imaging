"""Image(r,t) = sum_k b_k(r + u(r,t)) a_k(t).

The existing motion hash field supplies backward displacement in input units.
The intensity module is independent of the image-supervised toy training code,
so the same differentiable field can be queried by a PACT forward operator.
"""
from torch import nn

from .coordinates import DEFAULT_BOUNDS
from .low_rank_intensity import LowRankIntensity
from .nested_hash_nf import MotionHashNF


class NestedLowRankHashNF(nn.Module):
    def __init__(self, rank=3, width=64, temporal_width=32,
                 coordinate_bounds=DEFAULT_BOUNDS, reference_time=None,
                 motion_encoding=None, **encoding_kwargs):
        super().__init__()
        self.intensity = LowRankIntensity(
            rank=rank, width=width, temporal_width=temporal_width,
            coordinate_bounds=coordinate_bounds, **encoding_kwargs)
        self.motion = MotionHashNF(
            width=width, coordinate_bounds=coordinate_bounds,
            reference_time=reference_time, **(encoding_kwargs | (motion_encoding or {})))

    def forward(self, xyt):
        reference_xyt = xyt.clone()
        reference_xyt[:, :2] += self.motion(xyt)
        return self.intensity(reference_xyt)
