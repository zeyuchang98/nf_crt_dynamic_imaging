"""An ellipse with two spots: xy translation, xy expansion, changing brightness."""
import numpy as np


def dynamics(t):
    phase = 2 * np.pi * (t + 0.5)
    translation = np.stack((0.10 * np.sin(phase), 0.08 * np.sin(2 * phase)), axis=-1)
    scale = np.stack((1 + 0.30 * np.sin(phase / 2) ** 2,
                      1 + 0.45 * np.sin(phase) ** 2), axis=-1)
    global_gain = 1 + 0.25 * np.sin(3 * phase)
    spot_gain = 1 + 0.35 * np.sin(2 * phase)
    return translation, scale, global_gain, spot_gain


def reference_shapes(xy, size):
    """Shared reference geometry for data generation and geometry diagnostics."""
    x, y = np.asarray(xy).T
    edge_width = 1.2 / (size - 1)

    def ellipse(cx, cy, rx, ry):
        radius = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
        return np.clip((1 - radius) * min(rx, ry) / edge_width + 0.5, 0, 1)

    body = ellipse(0, 0, 0.16, 0.12)
    spot1 = ellipse(-0.055, 0.025, 0.035, 0.035)
    spot2 = ellipse(0.060, -0.030, 0.025, 0.020)
    return body, spot1, spot2


def reference_intensity(xyt, size):
    """C(q,t): fixed reference geometry, with intrinsic brightness changes."""
    body, spot1, spot2 = reference_shapes(xyt[:, :2], size)
    t = xyt[:, 2]
    _, _, global_gain, spot_gain = dynamics(t)
    return global_gain * (0.28 * body + 0.38 * spot_gain * spot1 + 0.20 * spot2)


def motion(xyt):
    """Forward r = scale*q + translation; backward u = (r-translation)/scale-r."""
    translation, scale, _, _ = dynamics(xyt[:, 2])
    return (xyt[:, :2] - translation) / scale - xyt[:, :2]
