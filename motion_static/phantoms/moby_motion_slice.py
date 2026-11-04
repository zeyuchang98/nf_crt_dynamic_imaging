"""Extract the true DCE-MOBY displacement at chosen z planes, in the layout of the p0 volumes.

Source `motion_vectors.h5` (inspected 2026-11-04, see PROGRESS.md):
    motion        (y, x, z, component, phase) float32 in mm, on the full-body grid of
                  `phantom_anatomy_0.h5`, which starts at z = 0 mm. Component k points along
                  source axis k, so the components are (uy, ux, uz).
    interpolated  (y, x, z, phase) int8, undocumented; -1 where the motion is 0 (mostly
                  outside the body)
u(X, p) moves tissue from its phase-0 position X to X + u(X, p) at phase p, so u = 0 at
phase 0. The p0 and indicator volumes cover the same x and y with these two axes in the
other order, and a z crop that starts at source plane fov_z0 / spacing_z (26.25 / 0.15 = 175).

Output (one HDF5 file), in the layout of p0 and of `moby_slice.py`'s output:
    displacement  (x, y, z, 3, phase) float32 in mm, components (ux, uy, uz)
    interpolated  (x, y, z, phase) int8
    z_index       (z,) p0 plane indices; source plane = z_index + attribute `z_offset`
    resp_phase    (phase,)
Values are copied, not resampled. Reads one phase (535 MB) at a time; run it on a compute node:
    python -m motion_static.phantoms.moby_motion_slice --data-root /path/to/moby-data \
        --z-index 100 --output out.h5
"""
import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

from .moby import file_sha256
from .moby_slice import git_commit


SOURCE_COMPONENTS_TO_P0 = [1, 0, 2]  # source (uy, ux, uz) -> (ux, uy, uz)


def to_p0_layout(source, has_components=True):
    """Source (y, x, ...) -> p0 (x, y, ...), with displacement components reordered to (ux, uy, uz)."""
    values = np.swapaxes(source, 0, 1)
    return values[..., SOURCE_COMPONENTS_TO_P0] if has_components else values


def crop_offset(fov, spacing):
    """Source plane of p0's first z plane; the source grid starts at z = 0."""
    offset = float(fov[2][0]) / float(spacing[2])
    if not np.isclose(offset, round(offset), atol=1e-6):
        raise ValueError(f"p0's z origin {fov[2][0]} is not a whole number of planes of {spacing[2]}")
    return int(round(offset))


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def extract(data_root, z_indices, output, overwrite=False):
    data_root, output = Path(data_root), Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass overwrite=True / --overwrite")
    with h5py.File(data_root / "p0" / "p0_00000.h5", "r") as f:
        shape = f["p0"].shape[:3]  # (x, y, z) of the p0 crop
        fov, spacing = np.asarray(f["p0"].attrs["fov"]), np.asarray(f["p0"].attrs["spacing"])
    z_indices = sorted(set(z_indices))
    if z_indices[0] < 0 or z_indices[-1] >= shape[2]:
        raise ValueError(f"z indices {z_indices} outside p0's [0, {shape[2]})")
    offset = crop_offset(fov, spacing)
    planes = [z + offset for z in z_indices]

    source_path = data_root / "motion_vectors.h5"
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as f, h5py.File(output, "w") as out:
        motion, flag = f["motion"], f["interpolated"]
        if motion.ndim != 5 or motion.shape[3] != 3 or motion.shape[:2] != (shape[1], shape[0]):
            raise ValueError(f"motion shape {motion.shape} is not (y, x, z, 3, phase) for p0 {shape}")
        if flag.shape != motion.shape[:3] + motion.shape[4:]:
            raise ValueError(f"interpolated shape {flag.shape} does not match motion {motion.shape}")
        if not np.allclose(motion.attrs["spacing"], spacing) or text(motion.attrs["units"]) != "mm":
            raise ValueError("motion spacing or units differ from p0")
        if planes[-1] >= motion.shape[2]:
            raise ValueError(f"source plane {planes[-1]} outside the motion grid of {motion.shape[2]} planes")
        phases = motion.shape[4]
        size = (shape[0], shape[1], len(planes))
        displacement = out.create_dataset("displacement", (*size, 3, phases), np.float32,
                                          chunks=(*size, 3, 1), compression="lzf")
        interpolated = out.create_dataset("interpolated", (*size, phases), np.int8,
                                          chunks=(*size, 1), compression="lzf")
        for p in range(phases):
            values = motion[:, :, planes, :, p]
            if p == 0 and np.any(values != 0):
                raise ValueError("phase 0 has nonzero displacement; the reference is not phase 0")
            displacement[..., p] = to_p0_layout(values)
            interpolated[..., p] = to_p0_layout(flag[:, :, planes, p], has_components=False)
            if p % 20 == 0 or p == phases - 1:
                print(f"phase {p + 1}/{phases}", flush=True)
        displacement.attrs.update(axes="x, y, z, component, phase", components="ux, uy, uz", units="mm",
                                  direction="tissue at its phase-0 position X is at X + u at phase p")
        interpolated.attrs.update(axes="x, y, z, phase", meaning="copied from the source, undocumented")
        out["z_index"] = np.array(z_indices)
        out["resp_phase"] = np.arange(phases)
        out.attrs.update(fov=fov, spacing=spacing, spacing_units="mm", z_offset=offset,
                         dt=motion.attrs["dt"], source_file=str(source_path.resolve()),
                         source_shape=motion.shape, code_commit=git_commit(), command=" ".join(sys.argv))
        out.attrs["source_sha256"] = file_sha256(source_path)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, help="folder with motion_vectors.h5 and p0/")
    parser.add_argument("--z-index", type=int, nargs="+", required=True, help="p0 z plane indices")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(f"wrote {extract(args.data_root, args.z_index, args.output, args.overwrite)}")


if __name__ == "__main__":
    main()
