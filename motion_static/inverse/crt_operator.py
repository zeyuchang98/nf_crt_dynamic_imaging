"""2D circular Radon transform (CRT) operator for the inverse problem, built with caSRT.

The operator is caSRT's `CircularRadonTransform` (precise-lab/SRT_cylindrical_aperture),
used unmodified from a local clone. Row `view * num_rings + ring` integrates the image
along the circle of radius `radii[ring]` around the sensor xi = R (cos theta, sin theta).
Its weights are arc lengths in mm, so the data are line integrals (intensity x mm), not
pressure. Column `ix + iy * N` is pixel (iy, ix), so an image in this project's (y, x)
layout enters as `image.reshape(-1)`.

Grid convention (caSRT): pixel i has its center at (i - (N // 2 - 1)) * dx relative to the
sensor-ring center. For even N this is half a pixel off the symmetric grid, so the ring
center lies at data coordinate fov0 + (N // 2 - 0.5) * dx: 18.525 mm for the 248-pixel,
0.15 mm DCE-MOBY slice, whose FOV center is 18.6 mm.

The saved operator holds every view on a regular angle grid; each frame uses the rows of
its own views (`frame_view_degrees`, ProxNF's 2D schedule). Build it once with `main`;
training loads the file and checks its geometry and hash.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import scipy.sparse as sp


DEFAULT_CASRT = Path(__file__).resolve().parents[2].parent / "SRT_cylindrical_aperture"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "inverse_data"
# Geometry entries that must match between a saved operator and a run configuration.
OPERATOR_KEYS = ("size", "fov_mm", "ring_radius_mm", "ring_spacing_mm", "angle_step_deg")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def load_casrt_crt(casrt_root=DEFAULT_CASRT):
    """Import caSRT/CRT.py as a module without CuPy.

    CRT.py imports cupy and cupyx at the top but uses them only in tocuda(), which this
    project never calls. Missing packages are replaced by empty modules during the import
    and removed again afterwards, so nothing else sees them.
    """
    path = Path(casrt_root).expanduser() / "caSRT" / "CRT.py"
    if not path.is_file():
        raise FileNotFoundError(f"caSRT not found: {path}")
    stubs = [name for name in ("cupy", "cupyx")
             if name not in sys.modules and importlib.util.find_spec(name) is None]
    for name in stubs:
        sys.modules[name] = type(sys)(name)
    try:
        spec = importlib.util.spec_from_file_location("casrt_crt", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
    return module


def git_state(folder, scope=None):
    """Commit and whether tracked files (under `scope`, if given) differ from it; None outside git."""
    try:
        commit = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        status = ["git", "-C", str(folder), "status", "--porcelain", "--untracked-files=no"]
        if scope is not None:
            status += ["--", str(scope)]
        dirty = bool(subprocess.run(status, capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None
    return dict(commit=commit, modified=dirty)


def code_state(*files):
    """Repository state of motion_static plus hashes of the generating files (new files are untracked)."""
    return dict(git=git_state(Path(__file__).resolve().parents[1], scope="."),
                files={Path(name).name: file_sha256(name) for name in files})


def operator_sha256(matrix):
    """Hash of the CSR arrays and shape, independent of the index dtypes."""
    matrix = matrix.tocsr()
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(matrix.indptr.astype(np.int64).tobytes())
    digest.update(matrix.indices.astype(np.int64).tobytes())
    digest.update(matrix.data.astype(np.float64).tobytes())
    return digest.hexdigest()


def circumscribed_radius(fov_mm):
    """Radius of the circle through the corners of the square FOV (ProxNF: 26.3 mm for 37.2 mm)."""
    return float(np.sqrt(2.) / 2. * fov_mm)


def ring_center(fov_start_mm, size, pixel_mm):
    """Data coordinate of the sensor-ring center along one axis (caSRT grid convention)."""
    return float(fov_start_mm + (size // 2 - 0.5) * pixel_mm)


def view_angles_deg(angle_step_deg):
    count = 360. / angle_step_deg
    if not np.isclose(count, round(count)):
        raise ValueError("angle_step_deg must divide 360")
    return angle_step_deg * np.arange(round(count))


def build_operator(size, fov_mm, ring_radius_mm, ring_spacing_mm, angle_step_deg=1., casrt_root=DEFAULT_CASRT):
    """Full-view CRT matrix (num_views * num_rings, size**2) in CSR form and its metadata."""
    module = load_casrt_crt(casrt_root)
    angles = view_angles_deg(angle_step_deg)
    start = time.perf_counter()
    crt = module.CircularRadonTransform(size, fov_mm, ring_radius_mm, angles=np.deg2rad(angles),
                                        dr=ring_spacing_mm)
    seconds = time.perf_counter() - start
    matrix = crt.A.tocsr()
    matrix.sort_indices()
    metadata = dict(
        size=int(size), fov_mm=float(fov_mm), pixel_mm=float(fov_mm / size),
        ring_radius_mm=float(ring_radius_mm), ring_spacing_mm=float(ring_spacing_mm),
        angle_step_deg=float(angle_step_deg), num_views=int(len(angles)), num_rings=int(crt.numCircles),
        radii_mm=[float(crt.radii[0]), float(crt.radii[-1])],
        shape=[int(v) for v in matrix.shape], nnz=int(matrix.nnz),
        operator_sha256=operator_sha256(matrix), build_seconds=seconds,
        casrt=dict(root=str(Path(casrt_root).expanduser().resolve()),
                   crt_py_sha256=file_sha256(Path(casrt_root).expanduser() / "caSRT" / "CRT.py"),
                   git=git_state(Path(casrt_root).expanduser())),
        code=code_state(__file__),
        row_order="view * num_rings + ring", column_order="ix + iy * size (image.reshape(-1) of a (y, x) image)",
        weights="arc length in mm (data are line integrals, not pressure)",
    )
    return matrix, metadata


def save_operator(path, matrix, metadata):
    """Write the CSR arrays (.npz) and the metadata next to them (.json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data=matrix.data, indices=matrix.indices, indptr=matrix.indptr,
             shape=np.asarray(matrix.shape))
    metadata = dict(metadata, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def check_geometry(metadata, expected):
    """Raise if any operator geometry entry in `expected` differs from the saved metadata."""
    for key in OPERATOR_KEYS:
        if key in expected and not np.isclose(float(metadata[key]), float(expected[key])):
            raise ValueError(f"Operator {key} is {metadata[key]}, expected {expected[key]}")


def load_operator(path, expected=None):
    """Load a saved operator, verify its hash and (optionally) its geometry."""
    path = Path(path).expanduser()
    with np.load(path) as arrays:
        matrix = sp.csr_matrix((arrays["data"], arrays["indices"], arrays["indptr"]),
                               shape=tuple(arrays["shape"]))
    metadata = json.loads(path.with_suffix(".json").read_text())
    if operator_sha256(matrix) != metadata["operator_sha256"]:
        raise ValueError(f"Operator arrays in {path} do not match their recorded hash")
    if expected is not None:
        check_geometry(metadata, expected)
    return matrix, metadata


def frame_view_degrees(frame_numbers, views_per_group=5, view_spacing_deg=1., rotation_deg=5., start_deg=0.,
                       opposite_groups=True):
    """Sensor angles of each frame, (frames, views) in [0, 360), ProxNF's 2D schedule.

    Frame f sees `views_per_group` sensors spaced `view_spacing_deg` apart, starting at
    start_deg + rotation_deg * f, plus the same group rotated by 180 degrees. With the
    defaults this is ProxNF's 5 + 5 sensors at 1 degree, rotated 5 degrees per frame
    (applications/proximal_splitting: angle index i * vpt + k + j * 180). Frame numbers
    are source frame indices, so a frame sees the same sensors in every window.
    """
    frames = np.atleast_1d(np.asarray(frame_numbers, dtype=float))
    group = start_deg + rotation_deg * frames[:, None] + view_spacing_deg * np.arange(views_per_group)[None, :]
    if opposite_groups:
        group = np.concatenate([group, group + 180.], axis=1)
    return np.mod(group, 360.)


def frame_rows(view_degrees, num_rings, angle_step_deg=1.):
    """Operator rows of one frame's views, ordered (view, ring) like caSRT's rows."""
    views = np.asarray(view_degrees, dtype=float) / angle_step_deg
    if not np.allclose(views, np.round(views)):
        raise ValueError("Frame views are not on the operator's angle grid")
    views = np.round(views).astype(np.int64)
    return (views[:, None] * num_rings + np.arange(num_rings)[None, :]).reshape(-1)


def default_path(metadata):
    return DEFAULT_OUT_DIR / (f"crt_N{metadata['size']}_fov{metadata['fov_mm']:g}_R{metadata['ring_radius_mm']:.3f}"
                              f"_dr{metadata['ring_spacing_mm']:g}_a{metadata['angle_step_deg']:g}.npz")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build and save the full-view 2D CRT operator for a slice file.")
    parser.add_argument("--data-file", required=True, help="slice file whose grid (size, spacing) the operator uses")
    parser.add_argument("--ring-radius-mm", type=float, default=None,
                        help="sensor-ring radius; default: circumscribed circle of the FOV (ProxNF)")
    parser.add_argument("--ring-spacing-mm", type=float, required=True, help="dr, spacing of the integration circles")
    parser.add_argument("--angle-step-deg", type=float, default=1.)
    parser.add_argument("--casrt", default=str(DEFAULT_CASRT))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    with h5py.File(Path(args.data_file).expanduser(), "r") as f:
        shape = f["p0"].shape
        spacing = np.asarray(f.attrs["spacing"], dtype=float)
        fov = np.asarray(f.attrs["fov"], dtype=float)
    size = int(shape[0])
    if shape[1] != size or not np.isclose(spacing[0], spacing[1]):
        raise ValueError("Expected a square grid with equal x and y spacing")
    fov_mm = size * float(spacing[0])
    radius = circumscribed_radius(fov_mm) if args.ring_radius_mm is None else args.ring_radius_mm
    matrix, metadata = build_operator(size, fov_mm, radius, args.ring_spacing_mm, args.angle_step_deg, args.casrt)
    metadata["ring_center_mm"] = [ring_center(fov[0, 0], size, spacing[0]), ring_center(fov[1, 0], size, spacing[1])]
    metadata["data_file"] = str(Path(args.data_file).expanduser().resolve())
    out = Path(args.out) if args.out else default_path(metadata)
    save_operator(out, matrix, metadata)
    print(f"Saved {out}: {metadata['shape'][0]} x {metadata['shape'][1]}, {metadata['nnz']:,} nonzeros, "
          f"{metadata['num_rings']} rings, built in {metadata['build_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
