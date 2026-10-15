"""Reference intensity C(q,t) = sum_k b_k(q) a_k(t), with total rank <= R.

The spatial hash network sees only q; the temporal MLP sees only t.
No shape, trajectory, or brightness law is supplied to either network.
"""
from torch import nn

from .coordinates import CoordinateNormalizer, DEFAULT_BOUNDS
from .hash_grid import HashGridEncoding


class LowRankIntensity(nn.Module):
    """Continuous (qx,qy,t) -> intensity; rank includes any static component."""

    def __init__(self, rank=3, width=64, temporal_width=32,
                 coordinate_bounds=DEFAULT_BOUNDS, **encoding_kwargs):
        super().__init__()
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("rank must be a positive integer")
        self.rank = rank
        self.normalizer = CoordinateNormalizer(coordinate_bounds)
        # Displaced spatial queries must retain gradients outside the image.
        self.encoding = HashGridEncoding(dimensions=2, clamp_coordinates=False,
                                         **encoding_kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoding.output_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, rank),
        )
        self.temporal = nn.Sequential(
            nn.Linear(1, temporal_width), nn.Tanh(),
            nn.Linear(temporal_width, temporal_width), nn.Tanh(),
            nn.Linear(temporal_width, rank),
        )

    def forward(self, xyt):
        basis = self.spatial_basis(xyt[:, :2])
        coefficients = self.temporal_coefficients(xyt[:, 2:])
        # A nonlinear decoder after this sum would remove the rank guarantee.
        return (basis * coefficients).sum(dim=-1)

    def spatial_basis(self, xy):
        """Return R spatial basis values at each physical reference coordinate."""
        unit_xy = (xy - self.normalizer.lower[:2]) / self.normalizer.span[:2]
        return self.mlp(self.encoding(unit_xy))

    def temporal_coefficients(self, times):
        """Input times have shape (N,1), in the dataset's original time units."""
        signed_t = 2 * (times - self.normalizer.center[2]) / self.normalizer.span[2]
        return self.temporal(signed_t)
