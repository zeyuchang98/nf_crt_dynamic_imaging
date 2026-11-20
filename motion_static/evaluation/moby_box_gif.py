"""Volume-render GIF of a 3D box run (dce_moby box file): truth on the left, NF on the right.

Each panel is a maximum-intensity volume render: every pixel shows the brightest voxel
along its ray, so the lesion and vessels inside the box are not hidden behind the body
surface. Both panels use one fixed display range 0..display_max in raw p0 units
(normalized values times the run's intensity scale), the same camera and the same
render settings; displays never enter metrics. Requires PyVista (optional dependency).

Frames are window frames in time order; `--frames` takes ranges START:STOP[:STEP] (stop
excluded) starting at window frame 0, e.g. `0:14401:50` for one frame per breath.
Writes visuals/NAME_volume.gif, a still PNG of its middle frame and NAME_volume_gif.json.
A run trained on the source phase files (`data.source_root`) reads its truth from an
extracted file of its box (`--data-file`, moby_slice.extract_box over every series frame),
one frame at a time.

python -m motion_static.evaluation.moby_box_gif motion_static/results/RUN
"""
import argparse
import json
from pathlib import Path

import h5py
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


class BoxFileFrames:
    """Window frames of an extracted box file (moby_slice.extract_box), read one at a time.

    For runs trained on the source phase files of a box (`data.source_root`): an extracted
    file of every series frame of that box holds the same values (a streamed run reproduces
    the run on the extracted file bit for bit, tests/test_streamed_training.py), and reading
    frame by frame keeps the whole series (15 GB for the 64^3 box) out of memory. The file
    must hold the run's box (start and shape), every series frame (so its frame index is
    the series index), the run's extent and the window's first and last times.
    """

    def __init__(self, path, config):
        data, bounds = config["data"], config["coordinate_bounds"]
        self.handle = h5py.File(path, "r")
        p0 = self.handle["p0"]
        self.frames = moby.window_frames(data)
        fov = np.asarray(self.handle.attrs["fov"], dtype=float)
        spacing = np.asarray(self.handle.attrs["spacing"], dtype=float)
        times = self.handle["time"][...].astype(float)
        problems = []
        if [int(v) for v in self.handle.attrs["box_start"]] != list(data["box_start"]):
            problems.append(f"box start {list(self.handle.attrs['box_start'])}, run {data['box_start']}")
        if list(p0.shape[:3]) != list(data["box_shape"]):
            problems.append(f"box shape {list(p0.shape[:3])}, run {data['box_shape']}")
        if p0.shape[3] != data["source_frames"]:
            problems.append(f"{p0.shape[3]} frames, the series has {data['source_frames']}")
        elif not np.allclose(times[self.frames[[0, -1]]], bounds[3], rtol=0, atol=1e-9):
            problems.append(f"window times {times[self.frames[[0, -1]]].tolist()}, run {bounds[3]}")
        if not np.allclose(fov, bounds[:3], rtol=0, atol=1e-9):
            problems.append(f"extent {fov.tolist()}, run {bounds[:3]}")
        if problems:
            self.handle.close()
            raise ValueError("--data-file does not hold the run's box: " + "; ".join(problems))
        self.t = times[self.frames]
        self.x, self.y, self.z = (moby.pixel_centers(fov[axis], spacing[axis], p0.shape[axis]) for axis in range(3))

    def raw(self, k):
        """Window frame k in raw p0 units, (y, x, z) as the training layout."""
        frame = int(self.frames[k])
        # Keep the frame axis (size 1): dropping it makes h5py read element by element.
        return np.transpose(self.handle["p0"][..., frame:frame + 1][..., 0], (1, 0, 2))

    def close(self):
        self.handle.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--display-max", type=float, default=0.05, help="display range 0..max, raw p0 units")
    parser.add_argument("--frames", nargs="*", default=[], metavar="START:STOP[:STEP]",
                        help="window frame ranges to show, starting at frame 0 (default: every frame)")
    parser.add_argument("--name", default="window", help="output prefix")
    parser.add_argument("--size", type=int, default=450, help="pixels per panel side")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for inference")
    parser.add_argument("--data-file", type=Path,
                        help="runs on a box file: a local copy, checked by its sha256; runs on source "
                             "files (required): an extracted file of the run's box and every series frame")
    args = parser.parse_args(argv)

    config = json.loads((args.run / "config.json").read_text())
    if config.get("phantom") != moby.NAME or config.get("spatial_dimensions") != 3:
        parser.error("Run is not a 3D dce_moby box run")
    if config.get("held_out_split") != "none":
        # A held-out run's training entry holds only the even window frames.
        parser.error("Only embedding-protocol runs (held_out_split none) are drawn")
    scale = config["data"]["intensity_scale_value"]
    data_record = {}
    if "source_root" in config["data"]:
        # Trained on the source phase files: read the window frames from an extracted box file.
        if args.data_file is None:
            parser.error("A run on source files needs --data-file, an extracted file of its box")
        try:
            series = BoxFileFrames(args.data_file, config)
        except ValueError as error:
            parser.error(str(error))
        grid, truth_raw, close_data = dict(x=series.x, y=series.y, z=series.z, t=series.t), series.raw, series.close
        data_record = dict(data_file=args.data_file.name, data_file_sha256=moby.file_sha256(args.data_file))
    else:
        if args.data_file is not None:
            if moby.file_sha256(args.data_file) != config["data"]["sha256"]:
                parser.error("--data-file does not have the sha256 recorded by the run")
            config["data"] = dict(config["data"], file=str(args.data_file.resolve()))
        grid = prepare_data(config)[0]["training"]  # embedding protocol: every window frame
        truth_raw, close_data = (lambda k: grid["truth"][..., k] * scale), (lambda: None)
    torch.set_num_threads(args.threads)
    model = load_model(args.run)
    try:
        picks = selected_frames(args.frames, len(grid["t"]))
    except ValueError as error:
        parser.error(str(error))
    axes = (grid["x"], grid["y"], grid["z"])
    spatial = (len(grid["y"]), len(grid["x"]), len(grid["z"]))

    out = args.run / "visuals"
    out.mkdir(exist_ok=True)
    panels = VolumePanels(axes, spatial, args.display_max, args.size)
    frames = []
    for k in picks:
        nf = predict(model, entry_coordinates(grid, slice(k, k + 1))).reshape(spatial) * scale
        rgb = panels.render(truth_raw(k), nf)
        frames.append(with_header(rgb, f"t = {grid['t'][k]:.2f} s | window frame {k}"))
    panels.close()
    close_data()

    path = out / f"{args.name}_volume.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=FRAME_DURATION_MS, loop=0)
    frames[len(frames) // 2].save(path.with_suffix(".png"))
    summary = dict(run=args.run.name, code_commit=git_commit(), file=path.name, frames=len(picks),
                   duration_ms=FRAME_DURATION_MS, panels=["truth", "NF"],
                   rendering="maximum-intensity volume render, gray, linear opacity, one fixed camera",
                   camera=CAMERA, units="raw p0 (normalized values times intensity_scale_value)",
                   intensity_scale_value=scale, display_range=[0, args.display_max],
                   frame_ranges=args.frames or "all window frames", window_frames=picks.tolist(),
                   times=grid["t"][picks].tolist(), **data_record,
                   megabytes=round(path.stat().st_size / 2**20, 2))
    (out / f"{args.name}_volume_gif.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {path} ({summary['megabytes']} MB)")


if __name__ == "__main__":
    main()
