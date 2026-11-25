"""POUnet (Lozenski et al. 2022; ProxNF): f(x, y, t) = sum_j psi_j(t) C[pixel(x, y), j].

psi is nfCRT's `POU_Siren` of time, imported unchanged: a SIREN whose softmax output is a
partition of unity (sum_j psi_j(t) = 1). C holds `partitions` coefficient images on the
pixel grid, so f has space-time rank <= `partitions`, continuous in time and discrete in
space. A query point uses its nearest pixel; in this project the field is only evaluated
at pixel centers.

Time scaling: psi sees (t - window center) / time_unit_s. By default (inverse/execute.py) the
unit is one sensor revolution, as in ProxNF's code (applications/proximal_splitting:
time_int = nrevs * linspace(-1/2, 1/2)), so psi sees the same time step per frame as in
ProxNF; `pou.time_half_range` instead maps the run's frames to -a..a.
"""
import torch
from torch import nn


class POUNet(nn.Module):
    def __init__(self, size, coordinate_bounds, time_unit_s, partitions=10, width=140, depth=4):
        super().__init__()
        from nfCRT.pou import POU_Siren  # the original implementation, from the repository root

        (x0, x1), (y0, y1), (t0, t1) = coordinate_bounds
        self.size = int(size)
        self.psi = POU_Siren(width, partitions, Depth=depth, geodim=1)
        self.coefficients = nn.Parameter(torch.zeros(self.size * self.size, partitions))
        self.register_buffer("origin", torch.tensor([x0, y0], dtype=torch.float32))
        self.register_buffer("pixel", torch.tensor([(x1 - x0) / size, (y1 - y0) / size], dtype=torch.float32))
        self.time_bounds = (float(t0), float(t1))
        self.time_center = 0.5 * (t0 + t1)
        self.time_unit_s = float(time_unit_s)

    def normalized_time(self, t):
        """Seconds -> psi's input: time from the window center in units of time_unit_s."""
        return (t - self.time_center) / self.time_unit_s

    def partition(self, t):
        """psi_j(t), (points, partitions); every row sums to 1."""
        return self.psi(self.normalized_time(t).reshape(-1, 1))

    def pixel_index(self, xy):
        """Flat x-fastest index of the pixel containing each (x, y) in mm."""
        cell = torch.floor((xy - self.origin) / self.pixel).long().clamp(0, self.size - 1)
        return cell[:, 1] * self.size + cell[:, 0]

    def forward(self, coordinates):
        """coordinates (points, 3): x, y in mm and t in seconds."""
        psi = self.partition(coordinates[:, -1])
        return (psi * self.coefficients[self.pixel_index(coordinates[:, :2])]).sum(dim=1)

    def frames(self, times):
        """Whole frames at `times` (frames,), (frames, size**2) x-fastest: psi(t) C^T."""
        return self.partition(times) @ self.coefficients.T
