"""Animate a DCE-MOBY z-slice file (from phantoms/moby_slice.py) as two short GIFs.

The interlaced series mixes two effects, so each GIF isolates one of them:
- respiratory cycle: one full cycle of consecutive frames (every phase once). Taking the
  first cycle (before injection) shows almost pure motion.
- contrast series: every frame of one respiratory phase (one frame per cycle), so the
  anatomy is fixed and only the contrast agent changes.

python -m motion_static.evaluation.moby_gifs ~/Data/dce-moby/moby_z100.h5 --out motion_static/results/dce_moby_z100
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FRAME_DURATION_MS = 90  # same as the other project GIFs


def respiratory_cycle_frames(resp_phase, cycle=0):
    """Indices of the consecutive frames that form one full respiratory cycle."""
    phases = int(resp_phase.max()) + 1
    frames = np.arange(cycle * phases, (cycle + 1) * phases)
    if frames[-1] >= len(resp_phase) or not np.array_equal(resp_phase[frames], np.arange(phases)):
        raise ValueError(f"Cycle {cycle} does not contain phases 0..{phases - 1} in order")
    return frames


def fixed_phase_frames(resp_phase, phase):
    """Indices of every frame acquired at one respiratory phase."""
    frames = np.flatnonzero(resp_phase == phase)
    if not frames.size:
        raise ValueError(f"No frames at phase {phase}")
    return frames


def global_maximum(dataset, block=500):
    """Maximum over all frames, read in blocks so the full series is never in memory."""
    return max(float(dataset[..., start:start + block].max())
               for start in range(0, dataset.shape[-1], block))


def save_animation(path, images, captions, title, display_max, scale=2, still_index=None):
    """Write (x, y, frame) images as a grayscale GIF with one fixed display range.

    Arrays are indexed (x, y); they are shown with x to the right and y upward,
    matching the toy-phantom GIFs, which display (y, x) arrays with np.flipud.
    """
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    width, height = scale * images.shape[0], scale * images.shape[1]
    frames = []
    for k, caption in enumerate(captions):
        canvas = Image.new("RGB", (width, height + 84), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), title, font=font, fill="black")
        draw.text((10, 30), f"Fixed display range: 0..{display_max:.3g}", font=small, fill="black")
        pixels = np.round(np.clip(np.flipud(images[..., k].T) / display_max, 0, 1) * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(pixels).convert("RGB").resize((width, height), Image.Resampling.NEAREST), (0, 52))
        draw.text((10, height + 60), caption, font=small, fill="black")
        frames.append(canvas)
    path = Path(path)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=FRAME_DURATION_MS, loop=0)
    still_index = len(frames) // 2 if still_index is None else still_index
    frames[still_index].save(path.with_suffix(".png"))
    return dict(frames=len(frames), duration_ms=FRAME_DURATION_MS, display_range=[0, display_max],
                still_frame=still_index, file=path.name)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slice_file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cycle", type=int, default=0, help="respiratory cycle for the motion GIF")
    parser.add_argument("--phase", type=int, default=0, help="respiratory phase for the contrast GIF")
    parser.add_argument("--scale", type=int, default=2, help="integer pixel upscaling")
    parser.add_argument("--display-max", type=float,
                        help="upper end of the fixed display range (default: global maximum of all frames)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.slice_file, "r") as f:
        p0, time, resp_phase = f["p0"], f["time"][:], f["resp_phase"][:]
        z_index = int(f.attrs["z_index"])
        # One range for both GIFs, so their brightness is directly comparable. A few
        # bright vessel voxels set the global maximum, so a lower value may be clearer.
        maximum = global_maximum(p0)
        display_max = maximum if args.display_max is None else args.display_max
        views = {
            f"respiratory_cycle{args.cycle}": (respiratory_cycle_frames(resp_phase, args.cycle),
                                               f"DCE-MOBY z={z_index}: respiratory cycle {args.cycle}"),
            f"contrast_phase{args.phase}": (fixed_phase_frames(resp_phase, args.phase),
                                            f"DCE-MOBY z={z_index}: contrast series at phase {args.phase}"),
        }
        summary = dict(source=str(args.slice_file.resolve()), source_sha256=sha256(args.slice_file),
                       z_index=z_index, code_commit=git_commit(), global_maximum=maximum,
                       display_max_source="global maximum" if args.display_max is None else "--display-max",
                       axes="x to the right, y upward", gifs={})
        for name, (frames, title) in views.items():
            captions = [f"t={time[n]:+.2f} s | phase {resp_phase[n]} | frame {n}" for n in frames]
            info = save_animation(args.out / f"{name}.gif", p0[..., frames], captions, title, display_max, args.scale)
            summary["gifs"][name] = dict(info, frame_indices=[int(frames[0]), int(frames[-1])],
                                         frame_count=len(frames), time_range_s=[float(time[frames[0]]), float(time[frames[-1]])])
            print(f"wrote {args.out / info['file']}")
    (args.out / "gifs.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
