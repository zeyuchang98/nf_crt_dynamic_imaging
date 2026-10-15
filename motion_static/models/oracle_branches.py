"""Ground-truth branches of the complex phantom, for oracle diagnostics only.

An oracle run replaces one learned branch of the nested model by the phantom's
true field and trains the other branch from image MSE:
  oracle "motion":    I(r,t) = phi_intensity(r + u_true(r,t), t)
  oracle "intensity": I(r,t) = C_true(r + phi_motion(r,t), t)
These runs use ground-truth branch labels and are not reconstruction methods.

The formulas repeat phantoms/complex.py in torch: the oracle intensity must pass
coordinate gradients to the learned motion, which the NumPy phantom cannot.
The phantom module is left unchanged so saved-checkpoint provenance stays valid;
tests/test_oracle_branches.py checks that both implementations agree.
"""
import math

import torch
from torch import nn


# The phantom's backward motion is zero at phase 0, i.e. at t = -0.5.
PHANTOM_REFERENCE_TIME = -0.5


def complex_dynamics(t):
    """Torch copy of phantoms.complex.dynamics; t has shape (N,)."""
    phase = 2 * math.pi * (t + 0.5)
    translation = torch.stack((0.10 * torch.sin(phase), 0.08 * torch.sin(2 * phase)), dim=-1)
    scale = torch.stack((1 + 0.30 * torch.sin(phase / 2) ** 2,
                         1 + 0.45 * torch.sin(phase) ** 2), dim=-1)
    global_gain = 1 + 0.25 * torch.sin(3 * phase)
    spot_gain = 1 + 0.35 * torch.sin(2 * phase)
    return translation, scale, global_gain, spot_gain


class TrueMotion(nn.Module):
    """u(r,t) = (r - T(t)) / S(t) - r, the phantom's backward displacement."""

    def __init__(self, reference_time=PHANTOM_REFERENCE_TIME):
        super().__init__()
        # The true field is anchored by construction; another reference time
        # would make it inconsistent with the learned-motion convention.
        if float(reference_time) != PHANTOM_REFERENCE_TIME:
            raise ValueError("True complex-phantom motion is anchored at t=-0.5")
        self.register_buffer("reference_time", torch.tensor(float(reference_time)))

    def forward(self, xyt):
        translation, scale, _, _ = complex_dynamics(xyt[:, 2])
        return (xyt[:, :2] - translation) / scale - xyt[:, :2]


class TrueReferenceIntensity(nn.Module):
    """C(q,t) = g(t) (0.28 body + 0.38 s(t) spot1 + 0.20 spot2), exactly rank 2."""

    def __init__(self, size):
        super().__init__()
        # The phantom's edge width depends on the pixel grid size.
        self.edge_width = 1.2 / (size - 1)

    def ellipse(self, x, y, cx, cy, rx, ry):
        # Clamping gives zero gradient on flat regions, as in the true image:
        # only the soft edges carry coordinate information.
        # Exactly at the center, use radius 0 with a zero gradient instead of
        # the 0/0 sqrt gradient; all returned values are unchanged.
        squared = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
        positive = squared > 0
        radius = torch.where(positive, torch.sqrt(torch.where(positive, squared, 1.)), 0.)
        return torch.clamp((1 - radius) * min(rx, ry) / self.edge_width + 0.5, 0, 1)

    def forward(self, xyt):
        x, y, t = xyt.unbind(-1)
        body = self.ellipse(x, y, 0, 0, 0.16, 0.12)
        spot1 = self.ellipse(x, y, -0.055, 0.025, 0.035, 0.035)
        spot2 = self.ellipse(x, y, 0.060, -0.030, 0.025, 0.020)
        _, _, global_gain, spot_gain = complex_dynamics(t)
        return global_gain * (0.28 * body + 0.38 * spot_gain * spot1 + 0.20 * spot2)
