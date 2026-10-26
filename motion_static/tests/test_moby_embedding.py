"""Embedding protocol, compute device and paper-format embedding figures for dce_moby."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from motion_static.evaluation import moby_embedding_figures, moby_svd_baseline
from motion_static.phantoms import moby
from motion_static.run import main, parse_configuration
from motion_static.tests.test_moby_data import DATA, SIZE, write_slice


def add_tumor(path):
    """A small tumor indicator for each of the 4 phases of the synthetic slice."""
    tumor = np.zeros((SIZE, SIZE, 4), np.float32)
    tumor[5:8, 9:12] = 1
    with h5py.File(path, "a") as f:
        f["indicator/tumor"] = tumor


class MobyEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.path = self.root / "slice.h5"
        write_slice(self.path)
        add_tumor(self.path)

    def tearDown(self):
        self.folder.cleanup()

    def write_config(self, name="config.json", **overrides):
        template = json.loads((Path(moby.__file__).parents[1] / "experiments/template.json").read_text())
        template.update(phantom=moby.NAME, data=dict(DATA, file=str(self.path)), rank=2, steps=3,
                        batch_size=64, threads=1, monitor_samples=32, log_interval=1, **overrides)
        for key in ("size", "frames", "cycles"):
            template.pop(key)
        path = self.root / name
        path.write_text(json.dumps(template))
        return path

    def test_embedding_protocol_fits_and_scores_every_window_frame(self):
        out = self.root / "results" / "embedding"
        main(["--config", str(self.write_config(held_out_split="none")), "--out", str(out)])
        config = json.loads((out / "config.json").read_text())
        report = json.loads((out / "metrics.json").read_text())
        self.assertEqual(config["frames"], 13)  # 3 cycles of 4 frames, both ends included
        self.assertTrue(config["midpoint_labels_used_for_training"])
        self.assertEqual(sorted(report["image"]), ["all_window_frames"])
        self.assertEqual(len(report["image"]["all_window_frames"]["times"]), 13)
        with (out / "loss.csv").open() as stream:
            self.assertTrue(all(row["fixed_validation_mse"] == "" for row in csv.DictReader(stream)))
        with (out.parent / "summary.csv").open() as stream:
            self.assertEqual(next(csv.DictReader(stream))["evaluation_split"], "embedding, all frames")
        with self.assertRaises(SystemExit):  # the plateau schedule needs validation frames
            parse_configuration(["--config", str(self.write_config(held_out_split="none", schedule="plateau")),
                                 "--out", str(self.root / "other")])

    def test_device_option_is_validated_and_weights_are_saved_on_the_cpu(self):
        out = str(self.root / "run")
        with self.assertRaises(SystemExit):
            parse_configuration(["--config", str(self.write_config(device="tpu")), "--out", out])
        if not torch.cuda.is_available():
            with self.assertRaises(SystemExit):
                parse_configuration(["--config", str(self.write_config(device="cuda")), "--out", out])
        if torch.backends.mps.is_available():
            run = self.root / "results" / "mps"
            main(["--config", str(self.write_config(device="mps")), "--out", str(run)])
            self.assertEqual(json.loads((run / "config.json").read_text())["device"], "mps")
            state = torch.load(run / "model.pt", weights_only=True)["state_dict"]
            self.assertEqual({value.device.type for value in state.values()}, {"cpu"})

    def test_svd_embedding_mode_and_figures_for_both_protocols(self):
        runs = {}
        for name, split in (("embedding", "none"), ("heldout", "alternate_validation_test")):
            runs[name] = self.root / "results" / name
            main(["--config", str(self.write_config(f"{name}.json", held_out_split=split)), "--out", str(runs[name])])
        svd = self.root / "svd_embedding"
        moby_svd_baseline.main([str(runs["embedding"]), "--out", str(svd), "--protocol", "embedding"])
        summary = json.loads((svd / "svd_baseline.json").read_text())
        self.assertEqual((summary["protocol"], summary["training_frames"]), ("embedding", 13))
        self.assertEqual(summary["ranks"][0]["parameters"], SIZE * SIZE + 13)
        self.assertLess(summary["ranks"][-1]["training_rrmse"], 1e-6)  # full rank reproduces every frame
        for name, protocol in (("embedding", "embedding"), ("heldout", "heldout")):
            svd = self.root / f"svd_{name}"
            if not svd.exists():
                moby_svd_baseline.main([str(runs[name]), "--out", str(svd), "--protocol", protocol])
            out = self.root / f"figures_{name}"
            moby_embedding_figures.main(["--svd", str(svd), "--nf-runs", str(runs[name]), "--out", str(out)])
            for figure in ("fig_singular_values", "fig_frames", "fig_activity_curves", "fig_nf_decomposition", "table"):
                self.assertTrue((out / f"{figure}.png").exists(), figure)
            record = json.loads((out / "embedding_figures.json").read_text())
            self.assertEqual(record["scored_frames"], 13 if name == "embedding" else 3)
            self.assertEqual(record["points"][0]["name"], "Point 1: tumor centroid")
            self.assertEqual((record["points"][0]["iy"], record["points"][0]["ix"]), (10, 6))
            self.assertGreaterEqual(len(record["ranks"]), 1)
        with self.assertRaises(SystemExit):  # the NF run's split must match the SS(r) protocol
            moby_embedding_figures.main(["--svd", str(self.root / "svd_embedding"), "--nf-runs", str(runs["heldout"]),
                                         "--out", str(self.root / "mismatch")])


if __name__ == "__main__":
    unittest.main()
