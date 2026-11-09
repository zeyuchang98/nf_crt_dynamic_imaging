"""Volume-render GIF of a 3D box run (dce_moby box file): truth on the left, NF on the right.

Each panel is a maximum-intensity volume render: every pixel shows the brightest voxel
along its ray, so the lesion and vessels inside the box are not hidden behind the body
surface. Both panels use one fixed display range 0..display_max in raw p0 units
(normalized values times the run's intensity scale), the same camera and the same
render settings; displays never enter metrics. Requires PyVista (optional dependency).

Frames are window frames in time order; `--frames` takes ranges START:STOP[:STEP] (stop
excluded) starting at window frame 0, e.g. `0:14401:50` for one frame per breath.
Writes visuals/NAME_volume.gif, a still PNG of its middle frame and NAME_volume_gif.json.

python -m motion_static.evaluation.moby_box_gif motion_static/results/RUN
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .evaluate import predict
from .moby_branch_gifs import selected_frames
from .moby_embedding_figures import load_model
from .moby_gifs import FRAME_DURATION_MS, git_commit
from ..phantoms import moby
from ..phantoms.data import entry_coordinates
from ..run import prepare_data

CAMERA = dict(position="iso", azimuth=30, elevation=10)  # one fixed oblique view
HEADER = 40  # pixels above the renders for the panel titles and the frame caption


def image_grid(pv, volume, axes):
    """(y, x, z) array -> PyVista ImageData on the voxel centers, x fastest, in mm."""
    x, y, z = axes
    spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    grid = pv.ImageData(dimensions=(len(x), len(y), len(z)), spacing=spacing, origin=(x[0], y[0], z[0]))
    grid.point_data["p0"] = np.transpose(volume, (1, 0, 2)).ravel(order="F")
    return grid


class VolumePanels:
    """Two side-by-side maximum-intensity renders whose volumes are replaced frame by frame."""

    def __init__(self, axes, shape, display_max, size):
        import pyvista as pv
        self.axes = axes
        self.plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(2 * size, size), border=False)
        self.plotter.set_background("black")
        self.grids = []
        for column in range(2):
            self.plotter.subplot(0, column)
            actor = self.plotter.add_volume(image_grid(pv, np.zeros(shape, np.float32), axes), scalars="p0",
                                            cmap="gray", clim=(0, display_max), opacity="linear",
                                            blending="maximum", show_scalar_bar=False)
            self.plotter.add_bounding_box(color="gray")
            self.plotter.camera_position = CAMERA["position"]
            self.plotter.camera.azimuth = CAMERA["azimuth"]
            self.plotter.camera.elevation = CAMERA["elevation"]
            # add_volume renders its own copy of the grid; later frames update that copy.
            self.grids.append(actor.mapper.dataset)

    def render(self, left, right):
        """RGB image of the two volumes, (y, x, z) arrays in raw p0 units."""
        for grid, volume in zip(self.grids, (left, right)):
            grid.point_data["p0"][:] = np.transpose(volume, (1, 0, 2)).ravel(order="F")
            grid.Modified()
        self.plotter.render()  # screenshot alone may return the previous frame
        return self.plotter.screenshot(return_img=True)

    def close(self):
        self.plotter.close()


def with_header(rgb, caption):
    """Panel titles over each half and the frame caption, drawn above the render."""
    height, width, _ = rgb.shape
    canvas = Image.new("RGB", (width, height + HEADER), "black")
    canvas.paste(Image.fromarray(rgb), (0, HEADER))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, title in enumerate(("Truth", "NF")):
        draw.text((column * width // 2 + 8, 4), title, fill="white", font=font)
    draw.text((8, 22), caption, fill="white", font=font)
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--display-max", type=float, default=0.05, help="display range 0..max, raw p0 units")
    parser.add_argument("--frames", nargs="*", default=[], metavar="START:STOP[:STEP]",
                        help="window frame ranges to show, starting at frame 0 (default: every frame)")
    parser.add_argument("--name", default="window", help="output prefix")
    parser.add_argument("--size", type=int, default=450, help="pixels per panel side")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for inference")
    parser.add_argument("--data-file", type=Path, help="local copy of the box file, checked by its sha256")
    args = parser.parse_args(argv)

    config = json.loads((args.run / "config.json").read_text())
    if config.get("phantom") != moby.NAME or config.get("spatial_dimensions") != 3:
        parser.error("Run is not a 3D dce_moby box run")
    if args.data_file is not None:
        if moby.file_sha256(args.data_file) != config["data"]["sha256"]:
            parser.error("--data-file does not have the sha256 recorded by the run")
        config["data"] = dict(config["data"], file=str(args.data_file.resolve()))
    torch.set_num_threads(args.threads)
    model = load_model(args.run)
    entry = prepare_data(config)[0]["training"]  # embedding protocol: every window frame
    try:
        picks = selected_frames(args.frames, len(entry["t"]))
    except ValueError as error:
        parser.error(str(error))
    scale = config["data"]["intensity_scale_value"]
    axes = (entry["x"], entry["y"], entry["z"])
    spatial = entry["truth"].shape[:-1]

    out = args.run / "visuals"
    out.mkdir(exist_ok=True)
    panels = VolumePanels(axes, spatial, args.display_max, args.size)
    frames = []
    for k in picks:
        nf = predict(model, entry_coordinates(entry, slice(k, k + 1))).reshape(spatial) * scale
        rgb = panels.render(entry["truth"][..., k] * scale, nf)
        frames.append(with_header(rgb, f"t = {entry['t'][k]:.2f} s | window frame {k}"))
    panels.close()

    path = out / f"{args.name}_volume.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=FRAME_DURATION_MS, loop=0)
    frames[len(frames) // 2].save(path.with_suffix(".png"))
    summary = dict(run=args.run.name, code_commit=git_commit(), file=path.name, frames=len(picks),
                   duration_ms=FRAME_DURATION_MS, panels=["truth", "NF"],
                   rendering="maximum-intensity volume render, gray, linear opacity, one fixed camera",
                   camera=CAMERA, units="raw p0 (normalized values times intensity_scale_value)",
                   intensity_scale_value=scale, display_range=[0, args.display_max],
                   frame_ranges=args.frames or "all window frames", window_frames=picks.tolist(),
                   times=entry["t"][picks].tolist(), megabytes=round(path.stat().st_size / 2**20, 2))
    (out / f"{args.name}_volume_gif.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {path} ({summary['megabytes']} MB)")


if __name__ == "__main__":
    main()
