"""Normalize x, y, t (2D) or x, y, z, t (3D) using fixed bounds shared by the whole dataset."""
import torch
from torch import nn


# Defaults for the existing toy experiments. Other datasets pass their own bounds.
DEFAULT_BOUNDS = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))


class CoordinateNormalizer(nn.Module):
    """Bounds are ((x_min, x_max), (y_min, y_max), (t_min, t_max)), with a z row before t in 3D.

    Time is always the last coordinate. Use the same bounds for every batch, frame,
    and displaced query. Bounds belong in the saved model config; they are not learned weights.
    """

    def __init__(self, bounds=DEFAULT_BOUNDS):
        super().__init__()
        bounds = torch.as_tensor(bounds, dtype=torch.float32)
        if bounds.ndim != 2 or bounds.shape[0] not in (3, 4) or bounds.shape[1] != 2 \
                or not torch.isfinite(bounds).all():
            raise ValueError("Expected finite (min, max) bounds for x, y, t or x, y, z, t")
        lower, upper = bounds[:, 0], bounds[:, 1]
        if not torch.all(upper > lower):
            raise ValueError("Each coordinate maximum must exceed its minimum")

        # Move these constants with the model. Config stores them, so old weight
        # files remain loadable without adding normalization keys to state_dict.
        self.register_buffer("lower", lower.clone(), persistent=False)
        self.register_buffer("span", upper - lower, persistent=False)
        self.register_buffer("center", (lower + upper) / 2, persistent=False)
        self.spatial_dimensions = bounds.shape[0] - 1

    def forward(self, xyt):
        """Map the supplied bounds to [0, 1], for hash grid queries."""
        return (xyt - self.lower) / self.span

    def to_signed(self, xyt):
        """Map the supplied bounds to [-1, 1], for coordinate MLPs."""
        return 2 * (xyt - self.center) / self.span
