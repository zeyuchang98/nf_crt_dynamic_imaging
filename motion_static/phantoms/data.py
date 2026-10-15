"""Generate either phantom with the same axes, labels, and train/midpoint split."""
import argparse
from pathlib import Path

import numpy as np

from . import simple, complex


def get_phantom(name):
    return {"simple": simple, "complex": complex}[name]


def coordinate_grid(x, y, t):
    """Flatten a (y,x,t) volume into rows of (x,y,t)."""
    yy, xx, tt = np.meshgrid(y, x, t, indexing="ij")
    return np.stack((xx, yy, tt), axis=-1).reshape(-1, 3)


def sample(name, size, times, time_bounds=(-0.5, 0.5)):
    if size < 16:
        raise ValueError("Use at least 16 pixels per side")
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not t.size or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Times must be finite, nonempty, and strictly increasing")
    bounds = np.asarray(time_bounds, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
        raise ValueError("Time bounds must be finite and increasing")
    if t.min() < bounds[0] or t.max() > bounds[1]:
        raise ValueError(f"Times must lie in {tuple(bounds)}")
    phantom = get_phantom(name)
    x = y = np.linspace(-0.5, 0.5, size)
    xyt = coordinate_grid(x, y, t)
    u = phantom.motion(xyt)
    warped = xyt.copy()
    warped[:, :2] += u
    shape = (size, size, len(t))
    truth = phantom.reference_intensity(warped, size).reshape(shape).astype(np.float32)
    reference_intensity = phantom.reference_intensity(xyt, size).reshape(shape).astype(np.float32)
    return dict(x=x, y=y, t=t, truth=truth, reference_intensity=reference_intensity,
                u_true=u.reshape(*shape, 2).astype(np.float32), foreground_mask=truth > 0)


def make_pair(name, size=64, frames=64, cycles=1):
    if frames < 2:
        raise ValueError("Use at least two training frames")
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 1:
        raise ValueError("Use a positive integer number of complete cycles")
    # The joint period of both existing phantoms is one input-time unit.
    # Extend observation time; do not rescale the dynamics or wrap network inputs.
    bounds = (-0.5, cycles - 0.5)
    times = np.linspace(*bounds, frames)
    return {"training": sample(name, size, times, bounds),
            "midpoint": sample(name, size, (times[:-1] + times[1:]) / 2, bounds)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phantom", choices=("simple", "complex"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--cycles", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out or root / "results" / args.phantom / "phantom"
    out.mkdir(parents=True, exist_ok=False)
    data = make_pair(args.phantom, frames=args.frames, cycles=args.cycles)
    for split, arrays in data.items():
        np.savez_compressed(out / f"{split}.npz", **arrays)
    from ..evaluation.plots import save_phantom
    save_phantom(out, data["training"], args.phantom)
    print(out)


if __name__ == "__main__":
    main()
