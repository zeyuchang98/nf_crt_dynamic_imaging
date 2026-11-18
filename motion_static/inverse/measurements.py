"""Simulated CRT measurements of a DCE-MOBY window: noiseless data and one saved noise draw.

Each frame k is measured only by its own views (crt_operator.frame_view_degrees):
    clean_k = P_k f_k,  P_k = rows of the full-view operator for frame k's views,
with f_k the normalized truth frame (moby.read_window_entry), reshaped x-fastest.

Noise model (agreed 2026-11-18, ProxNF's 2D setting): i.i.d. zero-mean Gaussian added to
the CRT data, sigma = rnl * max(clean data over all frames in this file), with rnl = 0.04
(relative noise level, 1 / peak SNR in amplitude). The data are generated with the same
operator that reconstructions use (inverse crime, as in ProxNF).

All methods read the same file, so they share one noise realization. `clean` is stored
for diagnostics only; reconstructions must use `noisy`.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from ..phantoms import moby
from . import crt_operator
from .crt_operator import DEFAULT_OUT_DIR, code_state, file_sha256, frame_rows, frame_view_degrees, load_operator

# ProxNF's 2D frame schedule (agreed 2026-11-18): 5 + 5 sensors at 1 degree, 5 degrees per frame.
DEFAULT_SCHEDULE = dict(views_per_group=5, view_spacing_deg=1., rotation_deg=5., start_deg=0., opposite_groups=True)


def frame_data(matrix, images, view_degrees, num_rings, angle_step_deg=1.):
    """Noiseless data (frames, views, rings) of images (y, x, frames), each frame with its own views."""
    frames = images.shape[-1]
    views = view_degrees.shape[1]
    clean = np.empty((frames, views, num_rings))
    for k in range(frames):
        rows = frame_rows(view_degrees[k], num_rings, angle_step_deg)
        image = np.asarray(images[..., k], dtype=np.float64).reshape(-1)  # x fastest, as caSRT's columns
        clean[k] = (matrix[rows] @ image).reshape(views, num_rings)
    return clean


def add_noise(clean, rnl, seed):
    """clean + N(0, sigma^2) i.i.d., sigma = rnl * max(clean). Returns (noisy, sigma)."""
    sigma = float(rnl) * float(np.max(clean))
    noise = np.random.default_rng(seed).normal(0., sigma, size=clean.shape)
    return clean + noise, sigma


def generate(data, operator_path, rnl, seed, schedule=None):
    """Measurements of a moby window (`data` as in a run configuration's "data" entry)."""
    schedule = dict(DEFAULT_SCHEDULE, **(schedule or {}))
    data, _ = moby.resolve_settings(data)
    entry = moby.read_window_entry(data)
    frames = moby.window_frames(data)
    matrix, operator = load_operator(operator_path)
    size, pixel_mm = entry["truth"].shape[0], float(entry["x"][1] - entry["x"][0])
    if entry["truth"].ndim != 3:
        raise ValueError("The 2D CRT needs a slice file")
    if size != operator["size"] or not np.isclose(pixel_mm, operator["pixel_mm"]):
        raise ValueError(f"Operator grid {operator['size']} x {operator['pixel_mm']} mm does not match "
                         f"the data grid {size} x {pixel_mm} mm")
    view_degrees = frame_view_degrees(frames, **schedule)
    clean = frame_data(matrix, entry["truth"], view_degrees, operator["num_rings"], operator["angle_step_deg"])
    noisy, sigma = add_noise(clean, rnl, seed)
    metadata = dict(
        data=data, frames=[int(frames[0]), int(frames[-1])], frame_count=int(len(frames)),
        times_s=[float(entry["t"][0]), float(entry["t"][-1])],
        operator=dict(file=str(Path(operator_path).expanduser().resolve()),
                      operator_sha256=operator["operator_sha256"],
                      **{key: operator[key] for key in ("size", "fov_mm", "pixel_mm", "ring_radius_mm",
                                                        "ring_spacing_mm", "angle_step_deg", "num_rings")}),
        schedule=schedule, views_per_frame=int(view_degrees.shape[1]),
        noise=dict(model="iid Gaussian, sigma = rnl * max(clean over all frames)", rnl=float(rnl), seed=int(seed),
                   sigma=sigma, clean_max=float(clean.max())),
        shape=[int(v) for v in clean.shape], layout="(frame, view, ring)",
        code=code_state(__file__, crt_operator.__file__),
    )
    arrays = dict(noisy=noisy.astype(np.float32), clean=clean.astype(np.float32), frames=frames,
                  times=entry["t"], view_degrees=view_degrees)
    return arrays, metadata


def save_measurements(path, arrays, metadata):
    """Write the arrays (.npz) and metadata with the npz hash (.json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    metadata = dict(metadata, npz_sha256=file_sha256(path),
                    created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_measurements(path):
    """Arrays and metadata of a saved measurement file, after checking its hash."""
    path = Path(path).expanduser()
    metadata = json.loads(path.with_suffix(".json").read_text())
    if file_sha256(path) != metadata["npz_sha256"]:
        raise ValueError(f"{path} does not match its recorded hash")
    with np.load(path) as saved:
        arrays = {name: saved[name] for name in saved.files}
    return arrays, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulate noisy CRT measurements of a DCE-MOBY window.")
    parser.add_argument("--file", required=True, help="slice file (as data.file in a run configuration)")
    parser.add_argument("--first-frame", type=int, required=True)
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--frames-per-cycle", type=int, required=True)
    parser.add_argument("--operator", required=True, help="saved operator (crt_operator.py)")
    parser.add_argument("--rnl", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    data = dict(file=args.file, first_frame=args.first_frame, cycles=args.cycles,
                frames_per_cycle=args.frames_per_cycle, intensity_scale="global_maximum")
    arrays, metadata = generate(data, args.operator, args.rnl, args.seed)
    first, last = metadata["frames"]
    out = Path(args.out) if args.out else (
        DEFAULT_OUT_DIR / f"meas_z{metadata['data']['z_index']}_f{first}-{last}_rnl{args.rnl:g}_seed{args.seed}.npz")
    metadata = save_measurements(out, arrays, metadata)
    print(f"Saved {out}: {metadata['shape']} (frame, view, ring), sigma {metadata['noise']['sigma']:.6g}, "
          f"clean max {metadata['noise']['clean_max']:.6g}", flush=True)


if __name__ == "__main__":
    main()
