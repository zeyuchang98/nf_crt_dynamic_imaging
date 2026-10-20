"""Extract one z slice of the DCE-MOBY p0 and indicator volumes into a single file.

Source layout (see the dataset README): `p0/p0_XXXXX.h5` holds one respiratory
phase, with `p0` shaped (x, y, z, frame) and `time` in seconds. Frames of
different phases are interlaced in time, so the dynamic object is obtained by
sorting all frames of all phases by `time`. `indicator/indicator_XXXXX.h5` holds
partial-volume maps (x, y, z) of each tissue for one phase.

Output (one HDF5 file):
    p0              (x, y, frame) float32, all frames of the selected phases sorted by time
    time            (frame,) seconds, injection at 0
    resp_phase      (frame,) source phase of each frame
    frame_in_phase  (frame,) index of each frame inside its source file
    indicator/<tissue>      (x, y, phase) float32 at the same z slice
    indicator/resp_phase    (phase,)
Axis order is kept as in the source; nothing is resampled or transposed.

This file only needs numpy and h5py so it can run on its own on a cluster:
    python moby_slice.py --data-root /path/to/moby-data --z-index 100 --output out.h5
"""
import argparse
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


INDICATOR_TISSUES = ("artery", "liver", "tumor", "tumor_core")
# Geometry attributes that must agree across every source file.
SHARED_ATTRIBUTES = ("fov", "spacing", "spacing_units")


def read_attributes(dataset, names):
    return {name: dataset.attrs[name] for name in names}


def check_same(reference, other, source):
    for name, value in reference.items():
        if not np.array_equal(np.asarray(value), np.asarray(other[name])):
            raise ValueError(f"{source}: attribute {name!r} differs from the first file")


def find_phases(folder, prefix, dataset_name, phase_attribute, phases):
    """Map respiratory phase -> file, using the phase stored in the file, not its name."""
    files = {}
    for path in sorted(Path(folder).glob(f"{prefix}_*.h5")):
        with h5py.File(path, "r") as f:
            phase = int(f[dataset_name].attrs[phase_attribute])
        if phase in files:
            raise ValueError(f"Phase {phase} appears in both {files[phase]} and {path}")
        files[phase] = path
    if not files:
        raise FileNotFoundError(f"No {prefix}_*.h5 files in {folder}")
    if phases is None:
        return files
    missing = sorted(set(phases) - set(files))
    if missing:
        raise ValueError(f"Phases {missing} not found in {folder}")
    return {phase: files[phase] for phase in sorted(set(phases))}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def extract(data_root, z_index, output, phases=None, with_indicator=True, overwrite=False):
    data_root, output = Path(data_root), Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass overwrite=True / --overwrite")
    p0_files = find_phases(data_root / "p0", "p0", "p0", "resp_phase", phases)

    # Pass 1: read times and metadata only, to fix the global time order before
    # touching the large arrays.
    times, frame_phase, frame_index = [], [], []
    geometry = wavelength = shape = None
    for phase, path in p0_files.items():
        with h5py.File(path, "r") as f:
            p0, t = f["p0"], f["time"][:]
            if p0.ndim != 4 or p0.shape[3] != len(t):
                raise ValueError(f"{path}: expected p0 (x, y, z, frame) matching time, got {p0.shape}")
            if geometry is None:
                geometry, wavelength, shape = read_attributes(p0, SHARED_ATTRIBUTES), p0.attrs["wavelength"], p0.shape[:3]
            check_same(geometry, read_attributes(p0, SHARED_ATTRIBUTES), path)
            check_same({"wavelength": wavelength}, {"wavelength": p0.attrs["wavelength"]}, path)
            if p0.shape[:3] != shape:
                raise ValueError(f"{path}: spatial shape {p0.shape[:3]} differs from {shape}")
            times.append(t)
            frame_phase.append(np.full(len(t), phase))
            frame_index.append(np.arange(len(t)))
    if not 0 <= z_index < shape[2]:
        raise ValueError(f"z_index {z_index} outside [0, {shape[2]})")
    times = np.concatenate(times)
    order = np.argsort(times, kind="stable")
    if np.any(np.diff(times[order]) <= 0):
        raise ValueError("Frame times are not unique, so the interlaced order is ambiguous")
    position = np.empty_like(order)
    position[order] = np.arange(len(order))  # sorted position of each source frame

    # Pass 2: read only the requested z plane of every phase into its sorted slots.
    p0_slice = np.empty((shape[0], shape[1], len(times)), dtype=np.float32)
    start = 0
    for count, (phase, path) in enumerate(p0_files.items(), 1):
        with h5py.File(path, "r") as f:
            plane = f["p0"][:, :, z_index, :]
        p0_slice[:, :, position[start:start + plane.shape[2]]] = plane
        start += plane.shape[2]
        print(f"p0 {count}/{len(p0_files)}: phase {phase}, {plane.shape[2]} frames", flush=True)

    indicator = None
    if with_indicator:
        indicator_files = find_phases(data_root / "indicator", "indicator", INDICATOR_TISSUES[0], "phase", p0_files)
        indicator = {tissue: np.empty((shape[0], shape[1], len(indicator_files)), dtype=np.float32)
                     for tissue in INDICATOR_TISSUES}
        for k, (phase, path) in enumerate(indicator_files.items()):
            with h5py.File(path, "r") as f:
                for tissue in INDICATOR_TISSUES:
                    volume = f[tissue]
                    if volume.shape != shape:
                        raise ValueError(f"{path}: {tissue} shape {volume.shape} differs from p0 {shape}")
                    check_same(geometry, read_attributes(volume, SHARED_ATTRIBUTES), f"{path}:{tissue}")
                    indicator[tissue][:, :, k] = volume[:, :, z_index]

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as f:
        dataset = f.create_dataset("p0", data=p0_slice, chunks=(shape[0], shape[1], 1), compression="lzf")
        dataset.attrs["axes"] = "x, y, frame"
        f["time"] = times[order]
        f["time"].attrs["units"] = "s"
        f["resp_phase"] = np.concatenate(frame_phase)[order]
        f["frame_in_phase"] = np.concatenate(frame_index)[order]
        if indicator is not None:
            group = f.create_group("indicator")
            for tissue, values in indicator.items():
                group.create_dataset(tissue, data=values, compression="lzf").attrs["axes"] = "x, y, phase"
            group["resp_phase"] = np.array(list(indicator_files))
        # Keep the source geometry unchanged; z_index locates the plane within it.
        for name, value in geometry.items():
            f.attrs[name] = value
        f.attrs["wavelength"] = wavelength
        f.attrs["z_index"] = z_index
        f.attrs["source_shape"] = shape
        f.attrs["source_root"] = str(data_root.resolve())
        f.attrs["code_commit"] = git_commit()
        f.attrs["command"] = " ".join(sys.argv)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, help="folder containing p0/ and indicator/")
    parser.add_argument("--z-index", type=int, required=True, help="z plane index in the source volumes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--phases", type=int, nargs="+", help="respiratory phases to include (default: all)")
    parser.add_argument("--no-indicator", action="store_true", help="skip the indicator maps")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = extract(args.data_root, args.z_index, args.output, args.phases,
                   with_indicator=not args.no_indicator, overwrite=args.overwrite)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
