"""Hash NF: (x, y, t) -> hash features -> MLP -> intensity or displacement.

Read HashNeuralField.forward for the complete model.
The grid lookup and interpolation are in hash_grid.py.
"""
from torch import nn

from .coordinates import CoordinateNormalizer, DEFAULT_BOUNDS
from .hash_grid import HashGridEncoding


class HashNeuralField(nn.Module):
    """Use output_dim=1 for intensity, or output_dim=2 for (ux, uy)."""

    def __init__(self, width=64, output_dim=1, coordinate_bounds=DEFAULT_BOUNDS,
                 **encoding_kwargs):
        super().__init__()
        self.normalizer = CoordinateNormalizer(coordinate_bounds)
        self.encoding = HashGridEncoding(**encoding_kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoding.output_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, output_dim),
        )

    def forward(self, xyt):
        # 1. Map the fixed dataset bounds to [0, 1]. Displaced queries use
        # the same bounds, so they may still lie outside this unit box.
        unit_xyt = self.normalizer(xyt)

        # 2. Look up and interpolate features from all grid levels.
        features = self.encoding(unit_xyt)

        # 3. Decode features into intensity (N,) or displacement (N, 2).
        output = self.mlp(features)
        return output.squeeze(-1)
