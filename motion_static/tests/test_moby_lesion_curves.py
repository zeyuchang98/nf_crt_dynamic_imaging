"""Lesion activity curves: Gram-matrix SS(r) factors and curves checked against direct computation."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import h5py
import numpy as np

from motion_static.evaluation import moby_lesion_curves, moby_svd_baseline
from motion_static.evaluation.evaluate import predict
from motion_static.evaluation.moby_embedding_figures import load_model
from motion_static.phantoms import moby
from motion_static.phantoms.data import coordinate_grid
from motion_static.run import main, prepare_data
from motion_static.tests.test_moby_data import DATA, SIZE, write_slice


def add_lesion(path):
    """Shell and core partial volumes (file layout x, y, phase) that move one pixel in x per phase.

    The lesion is not symmetric in x and y, so a transposed mask would give other curves.
    """
    shell = np.zeros((SIZE, SIZE, 4), np.float32)
    core = np.zeros_like(shell)
    for phase in range(4):
        x = 3 + phase
        shell[x:x + 5, 8:12, phase] = 3 / 7
        shell[x + 1:x + 4, 9:11, phase] = 0
        core[x + 1:x + 4, 9:11, phase] = 1
        core[x + 1, 9, phase], shell[x + 1, 9, phase] = 2 / 7, 5 / 7  # a pixel shared by both
    with h5py.File(path, "a") as f:
        f["indicator/tumor"] = shell
        f["indicator/tumor_core"] = core
    return shell + core


def weighted_curve(weights, phases, image):
    """Direct LAC: weights (y, x, phase), phases per frame, image (y, x, t)."""
    w = weights[..., phases]
    return np.sum(w * image, axis=(0, 1)) / np.sum(w, axis=(0, 1))


class LesionCurveTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.path = self.root / "slice.h5"
        write_slice(self.path)
        self.whole = add_lesion(self.path)

    def tearDown(self):
        self.folder.cleanup()

    def run_model(self, name, split):
        template = json.loads((Path(moby.__file__).parents[1] / "experiments/template.json").read_text())
        template.update(phantom=moby.NAME, data=dict(DATA, file=str(self.path)), rank=2, steps=3, batch_size=64,
                        threads=1, monitor_samples=32, log_interval=1, held_out_split=split)
        for key in ("size", "frames", "cycles"):
            template.pop(key)
        config = self.root / f"{name}.json"
        config.write_text(json.dumps(template))
        out = self.root / "results" / name
        main(["--config", str(config), "--out", str(out)])
        return out

    def test_gram_factors_match_the_full_svd(self):
        frames = np.random.default_rng(1).random((50, 12)).astype(np.float32)
        spatial, temporal, sigma, energy = moby_lesion_curves.top_singular_factors(frames, 4)
        full_spatial, full_temporal = moby_svd_baseline.semiseparable_factors(dict(truth=frames[:, None, :]))
        np.testing.assert_allclose(spatial @ temporal.T, full_spatial[:, :4] @ full_temporal[:, :4].T, atol=1e-10)
        np.testing.assert_allclose(sigma, np.linalg.svd(frames.astype(np.float64), compute_uv=False)[:4], rtol=1e-10)
        self.assertAlmostEqual(energy, float(np.sum(frames.astype(np.float64) ** 2)), places=8)

    def test_curves_equal_weighted_means_of_truth_ss_and_nf_images(self):
        run = self.run_model("embedding", "none")
        svd = self.root / "svd"
        moby_svd_baseline.main([str(run), "--out", str(svd), "--protocol", "embedding"])
        out = self.root / "curves"
        moby_lesion_curves.main(["--svd", str(svd), "--nf-runs", str(run), "--ranks", "1", "3", "--out", str(out)])
        record = json.loads((out / "lesion_curves.json").read_text())
        curves = np.load(out / "lesion_curves.npz")
        self.assertTrue((out / "lesion_curves.png").exists())
        self.assertEqual(sorted(record["scores"]), ["core", "shell", "whole lesion"])
        for r in (1, 3):
            check = record["ss_check"][f"SS({r})"]
            self.assertAlmostEqual(check["rrmse_rebuilt"], check["rrmse_saved"], places=9)

        config = json.loads((run / "config.json").read_text())
        training = prepare_data(config)[0]["training"]
        scale = config["data"]["intensity_scale_value"]
        with h5py.File(self.path, "r") as f:
            phases = f["resp_phase"][moby.window_frames(config["data"])]
            shell = np.transpose(f["indicator/tumor"][...], (1, 0, 2))
            core = np.transpose(f["indicator/tumor_core"][...], (1, 0, 2))
        shape = training["truth"].shape
        spatial, temporal = moby_svd_baseline.semiseparable_factors(training)
        images = {"Object": training["truth"], "NF (ours)": predict(load_model(run), coordinate_grid(
            training["x"], training["y"], training["t"])).reshape(shape)}
        for r in (1, 3):
            images[f"SS({r})"] = moby_svd_baseline.reconstruct(spatial, temporal, training["t"], training["t"], r, shape)
        for region, weights in (("whole lesion", shell + core), ("shell", shell), ("core", core)):
            for name, image in images.items():
                expected = weighted_curve(weights.astype(np.float64), phases, image.astype(np.float64)) * scale
                np.testing.assert_allclose(curves[f"{region}|{name}"], expected, rtol=1e-5, atol=1e-9,
                                           err_msg=f"{region}, {name}")
            truth = curves[f"{region}|Object"]
            nf = curves[f"{region}|NF (ours)"]
            self.assertAlmostEqual(record["scores"][region]["NF (ours)"]["lac_rrmse"],
                                   float(np.linalg.norm(nf - truth) / np.linalg.norm(truth)), places=9)

        # A copy of the slice file is accepted by its sha256 and gives the same curves.
        copy = self.root / "copy.h5"
        shutil.copy(self.path, copy)
        again = self.root / "curves_copy"
        moby_lesion_curves.main(["--svd", str(svd), "--nf-runs", str(run), "--ranks", "1", "--data-file", str(copy),
                                 "--out", str(again)])
        np.testing.assert_array_equal(np.load(again / "lesion_curves.npz")["whole lesion|Object"],
                                      curves["whole lesion|Object"])

        other = self.root / "other.h5"
        write_slice(other)
        add_lesion(other)
        with h5py.File(other, "a") as f:
            f["p0"][0, 0, 0] = 0.123  # different content, different sha256
        heldout = self.run_model("heldout", "alternate_validation_test")
        for arguments in (["--data-file", str(other)], ["--ranks", "99"]):
            with self.assertRaises(SystemExit):
                moby_lesion_curves.main(["--svd", str(svd), "--nf-runs", str(run), "--ranks", "1",
                                         "--out", str(self.root / "rejected"), *arguments])
        with self.assertRaises(SystemExit):  # the NF run must use the embedding protocol
            moby_lesion_curves.main(["--svd", str(svd), "--nf-runs", str(heldout), "--ranks", "1",
                                     "--out", str(self.root / "rejected")])
        with self.assertRaises(SystemExit):  # outputs are never overwritten
            moby_lesion_curves.main(["--svd", str(svd), "--nf-runs", str(run), "--ranks", "1", "--out", str(out)])


if __name__ == "__main__":
    unittest.main()
