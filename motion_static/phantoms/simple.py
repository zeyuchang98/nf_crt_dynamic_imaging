"""A constant-brightness disk translating vertically. Coordinates are [-.5, .5]."""
import numpy as np


def reference_intensity(xyt, size):
    """Reference-coordinate intensity C(q,t); this phantom ignores time."""
    radius = np.hypot(xyt[:, 0], xyt[:, 1])
    edge_width = 1.5 / (size - 1)
    return np.clip((0.15 - radius) / edge_width + 0.5, 0, 1)


def motion(xyt):
    """Backward displacement u(r,t): query the reference image at r + u."""
    displacement = np.zeros((len(xyt), 2), dtype=xyt.dtype)
    displacement[:, 1] = -0.15 * np.sin(2 * np.pi * (xyt[:, 2] + 0.5))
    return displacement
