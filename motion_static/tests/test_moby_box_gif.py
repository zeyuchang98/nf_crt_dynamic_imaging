"""Truth | NF volume-render GIF of a 3D box run (needs the optional PyVista)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image, ImageSequence

from motion_static.phantoms import moby
from motion_static.run import main
from motion_static.tests.test_moby_3d import config_3d, write_box

HAS_PYVISTA = importlib.util.find_spec("pyvista") is not None


@unittest.skipUnless(HAS_PYVISTA, "PyVista is an optional dependency")
class BoxGifTests(unittest.TestCase):
    def test_updated_volumes_render_like_fresh_panels(self):
        from motion_static.evaluation.moby_box_gif import VolumePanels
        rng = np.random.default_rng(0)
        axes = (np.arange(16) * 0.15, np.arange(12) * 0.15, np.arange(10) * 0.15)
        volumes = [rng.random((12, 16, 10)) * 0.05 for _ in range(3)]
        reused = VolumePanels(axes, (12, 16, 10), 0.05, 120)
        first = reused.render(volumes[0], volumes[1])
        second = reused.render(volumes[2], volumes[0])
        reused.close()
        fresh = VolumePanels(axes, (12, 16, 10), 0.05, 120)
        expected = fresh.render(volumes[2], volumes[0])
        fresh.close()
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(second, expected)

    def test_gif_has_one_frame_per_selected_window_frame(self):
        from motion_static.evaluation import moby_box_gif
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_box(root / "box.h5")
            (root / "run.json").write_text(json.dumps(config_3d(root / "box.h5")))
            run = root / "results" / "run"
            main(["--config", str(root / "run.json"), "--out", str(run)])
            moby_box_gif.main([str(run), "--frames", "0:13:3", "--size", "96"])
            with Image.open(run / "visuals" / "window_volume.gif") as gif:
                self.assertEqual(gif.n_frames, 5)  # window frames 0, 3, 6, 9, 12
                self.assertEqual(gif.size, (192, 96 + moby_box_gif.HEADER))
            record = json.loads((run / "visuals" / "window_volume_gif.json").read_text())
            self.assertEqual(record["window_frames"], [0, 3, 6, 9, 12])
            self.assertEqual(record["display_range"], [0, 0.05])
            self.assertTrue((run / "visuals" / "window_volume.png").exists())

            two_d = root / "two_d"
            two_d.mkdir()
            (two_d / "config.json").write_text(json.dumps({"phantom": "dce_moby"}))
            with self.assertRaises(SystemExit):  # slice runs have no volume to render
                moby_box_gif.main([str(two_d)])
            held_out = root / "held_out"
            held_out.mkdir()
            (held_out / "config.json").write_text(json.dumps(
                {"phantom": "dce_moby", "spatial_dimensions": 3, "held_out_split": "midpoints"}))
            with self.assertRaises(SystemExit):  # its training entry has only the even frames
                moby_box_gif.main([str(held_out)])

    def test_source_run_reads_its_extracted_box_like_the_run_on_that_file(self):
        from motion_static.evaluation import moby_box_gif
        from motion_static.phantoms.moby_slice import extract_box
        from motion_static.tests.test_streamed_training import (BOX_SHAPE, BOX_START, WINDOW, captured_stderr,
                                                                source_config, write_source)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_source(root)
            box = extract_box(root, BOX_START, BOX_SHAPE, root / "box.h5")
            on_file = config_3d(box, steps=4)
            on_file["data"] = dict(WINDOW, file=str(box))
            runs = {}
            for name, config in (("file", on_file), ("source", source_config(root))):
                (root / f"{name}.json").write_text(json.dumps(config))
                runs[name] = root / "results" / name
                main(["--config", str(root / f"{name}.json"), "--out", str(runs[name])])
            options = ["--frames", "0:21:5", "--size", "96"]
            moby_box_gif.main([str(runs["file"]), *options])
            moby_box_gif.main([str(runs["source"]), *options, "--data-file", str(box)])
            # Same weights (the streamed run reproduces the file run) and the same truth, read
            # frame by frame instead of normalized in memory: equal stills up to one gray level.
            stills = [np.asarray(Image.open(runs[name] / "visuals" / "window_volume.png"), dtype=int)
                      for name in ("file", "source")]
            self.assertLessEqual(np.abs(stills[0] - stills[1]).max(), 1)
            with Image.open(runs["source"] / "visuals" / "window_volume.gif") as gif:
                self.assertEqual(gif.n_frames, 5)
            record = json.loads((runs["source"] / "visuals" / "window_volume_gif.json").read_text())
            self.assertEqual(record["window_frames"], [0, 5, 10, 15, 20])
            self.assertEqual((record["data_file"], len(record["data_file_sha256"])), ("box.h5", 64))
            expected = json.loads((runs["file"] / "visuals" / "window_volume_gif.json").read_text())
            self.assertEqual(record["times"], expected["times"])

            # Without --data-file the truth is read from the source files: the same still.
            moby_box_gif.main([str(runs["source"]), *options, "--name", "from_source"])
            from_source = np.asarray(Image.open(runs["source"] / "visuals" / "from_source_volume.png"), dtype=int)
            np.testing.assert_array_equal(from_source, stills[1])
            record = json.loads((runs["source"] / "visuals" / "from_source_volume_gif.json").read_text())
            source_root = json.loads((runs["source"] / "config.json").read_text())["data"]["source_root"]
            self.assertEqual(record["truth_source"], f"source files {source_root}")

            shifted = extract_box(root, (2, 2, 1), BOX_SHAPE, root / "shifted.h5")
            window_only = extract_box(root, BOX_START, BOX_SHAPE, root / "window.h5", frames=(0, 21))
            moved = root / "results" / "moved"  # a run whose source files are not on this machine
            moved.mkdir()
            config = json.loads((runs["source"] / "config.json").read_text())
            config["data"]["source_root"] = str(root / "missing")
            (moved / "config.json").write_text(json.dumps(config))
            for run, data_file in ((runs["source"], ["--data-file", str(shifted)]),
                                   (runs["source"], ["--data-file", str(window_only)]), (moved, [])):
                with self.subTest(data_file=data_file), self.assertRaises(SystemExit), captured_stderr():
                    moby_box_gif.main([str(run), *options, *data_file])

    def test_exported_frames_draw_the_same_gif_as_the_run_itself(self):
        from motion_static.evaluation import moby_box_gif
        from motion_static.evaluation.evaluate import predict
        from motion_static.evaluation.moby_embedding_figures import load_model
        from motion_static.phantoms.data import entry_coordinates
        from motion_static.tests.test_streamed_training import captured_stderr, source_config, write_source
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_source(root)
            (root / "run.json").write_text(json.dumps(source_config(root)))
            run = root / "results" / "run"
            main(["--config", str(root / "run.json"), "--out", str(run)])
            # A clip that does not start at window frame 0.
            frames = ["--frames", "5:21:5"]
            moby_box_gif.main([str(run), *frames, "--export", str(root / "frames.h5")])
            self.assertFalse((run / "visuals").exists())  # export draws nothing
            moby_box_gif.main([str(run), *frames, "--size", "96", "--name", "direct"])
            moby_box_gif.main([str(run), "--frames-file", str(root / "frames.h5"), "--size", "96", "--name", "stored"])

            # The file holds the source frames and the model's prediction, in raw p0 units.
            config = json.loads((run / "config.json").read_text())
            scale = config["data"]["intensity_scale_value"]
            series = moby_box_gif.SourceRunFrames(config)
            grid = dict(x=series.x, y=series.y, z=series.z, t=series.t)
            model = load_model(run)
            with h5py.File(root / "frames.h5", "r") as f:
                np.testing.assert_array_equal(f["window_frames"][...], [5, 10, 15, 20])
                self.assertEqual(f["truth"].shape, (16, 16, 9, 4))  # (y, x, z, frame) of the box
                for j, k in enumerate([5, 10, 15, 20]):
                    np.testing.assert_array_equal(f["truth"][..., j], series.raw(k))
                    nf = predict(model, entry_coordinates(grid, slice(k, k + 1))).reshape(16, 16, 9) * scale
                    np.testing.assert_array_equal(f["nf"][..., j], nf)
                self.assertEqual(f.attrs["model_sha256"], moby.file_sha256(run / "model.pt"))
            series.close()

            # The GIF from the file equals the GIF drawn from the run, frame for frame.
            gifs = {}
            for name in ("direct", "stored"):
                with Image.open(run / "visuals" / f"{name}_volume.gif") as gif:
                    gifs[name] = [np.asarray(frame.convert("RGB")) for frame in ImageSequence.Iterator(gif)]
            self.assertEqual(len(gifs["stored"]), 4)
            for direct, stored in zip(gifs["direct"], gifs["stored"]):
                np.testing.assert_array_equal(direct, stored)
            records = {name: json.loads((run / "visuals" / f"{name}_volume_gif.json").read_text())
                       for name in ("direct", "stored")}
            self.assertEqual(records["stored"]["window_frames"], [5, 10, 15, 20])
            self.assertEqual(records["stored"]["times"], records["direct"]["times"])
            self.assertEqual(records["stored"]["frames_file"], "frames.h5")

            # The file belongs to its run only, and carries its own frames.
            other = root / "results" / "other"
            other.mkdir()
            (other / "config.json").write_text(json.dumps(dict(config, started_at="another time")))
            for arguments in ([str(other), "--frames-file", str(root / "frames.h5")],
                              [str(run), "--frames-file", str(root / "frames.h5"), *frames]):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit), captured_stderr():
                    moby_box_gif.main(arguments)


if __name__ == "__main__":
    unittest.main()
