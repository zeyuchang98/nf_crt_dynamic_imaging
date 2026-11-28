"""2D inverse evaluation: fixed region pixels, rendering layout, lesion curves, scores and a complete run."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from skimage.metrics import structural_similarity
import torch

from motion_static.evaluation import inverse_evaluate as ie
from motion_static.evaluation.metrics import evaluate_sequence
from motion_static.inverse import crt_operator as crt
from motion_static.inverse import measurements
from motion_static.run import main as run_main
from motion_static.tests.test_inverse_data import DR, FOV, HAVE_CASRT, SIZE, write_slice
from motion_static.tests.test_inverse_direct import DATA, NESTED
from motion_static.tests.test_inverse_proximal import ProximalTestCase
from motion_static.tests.test_moby_lesion_curves import weighted_curve


class DeepestPixelTests(unittest.TestCase):
    def test_deepest_pixel_of_a_moving_rectangle(self):
        # Rows 2-8 at every phase; columns 2+p .. 8+p at phase p. Inside at every phase: rows 2-8 and
        # columns 4-8 (35 pixels). Depth in a rectangle is the distance to its edge along an axis, so
        # the worst-phase depth is min(r - 1, 9 - r, c - 3, 9 - c): at most 3, first reached at (4, 6).
        full = np.zeros((12, 14, 3), bool)
        for p in range(3):
            full[2:9, 2 + p:9 + p, p] = True
        self.assertEqual(ie.deepest_inside_pixel(full), (4, 6, 3.0, 35))

    def test_region_never_wholly_inside_at_every_phase(self):
        full = np.zeros((6, 6, 2), bool)
        full[1, 1, 0] = full[4, 4, 1] = True
        with self.assertRaises(ValueError):
            ie.deepest_inside_pixel(full)


class _Field(torch.nn.Module):
    """f(x, y, t) = x + 10 y + 100 t at query points (x, y, t)."""

    def forward(self, points):
        return points @ torch.tensor([1., 10., 100.])


class _Partitions(torch.nn.Module):
    """The same field rendered like POUnet: frames(t) gives (K, pixels), x-fastest."""

    def __init__(self, x, y):
        super().__init__()
        yy, xx = np.meshgrid(y, x, indexing="ij")
        self.space = torch.as_tensor(xx.ravel() + 10 * yy.ravel(), dtype=torch.float32)

    def frames(self, times):
        return self.space[None, :] + 100 * times[:, None]


class RenderTests(unittest.TestCase):
    def test_frames_are_t_y_x_for_fields_and_partitions(self):
        x, y, t = np.arange(5) * 0.5, np.arange(3) * 0.25, np.linspace(-1, 1, 7)  # 7 frames in chunks of 3
        expected = x[None, None, :] + 10 * y[None, :, None] + 100 * t[:, None, None]
        for model in (_Field(), _Partitions(x, y)):
            with self.subTest(type(model).__name__):
                images = ie.render_frames(model, x, y, t, torch.device("cpu"), chunk=3)
                self.assertEqual(images.shape, (7, 3, 5))
                np.testing.assert_allclose(images, expected, rtol=1e-6, atol=1e-5)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.truth = rng.random((12, 10, 9)).astype(np.float32) * 0.3  # (y, x, t)
        self.images = np.moveaxis(self.truth + rng.normal(0, 0.03, self.truth.shape).astype(np.float32), -1, 0)
        self.times = np.linspace(-0.2, 0.2, 9)
        weights = rng.random((12, 10, 2)) * (rng.random((12, 10, 2)) > 0.6)
        self.weights = {"whole lesion": weights, "core": weights ** 2}
        self.phases = np.arange(9) % 2

    def test_lesion_curves_match_the_direct_weighted_mean(self):
        curves = ie.lesion_curves(self.weights, self.phases, lambda picks: self.truth[..., picks])
        for region, weights in self.weights.items():
            np.testing.assert_allclose(curves[region], weighted_curve(weights, self.phases, self.truth), rtol=1e-12)

    def test_scores_match_whole_array_evaluation_and_direct_formulas(self):
        points = {"a": dict(row=3, col=7), "b": dict(row=10, col=0)}
        roi = (slice(1, 11), slice(2, 9))
        report, curves = ie.evaluate_reconstruction(self.truth, self.images, self.times, points, self.weights,
                                                    self.phases, roi)
        recon = np.moveaxis(self.images, 0, -1)
        whole = evaluate_sequence(self.truth, recon, data_range=1, times=self.times, roi=roi, points={})
        for key in ("rrmse", "psnr_db", "ssim"):
            self.assertAlmostEqual(report[key], whole[key], places=10)
            self.assertAlmostEqual(report["roi"][key], whole["roi"][key], places=10)
        np.testing.assert_allclose(curves["per_frame"]["ssim"], whole["per_frame_ssim"], rtol=1e-10)
        # ProxNF's call: skimage defaults on the whole space-time array, data range of the reconstruction.
        proxnf = structural_similarity(self.truth, recon, data_range=float(recon.max() - recon.min()))
        self.assertAlmostEqual(report["ssim_proxnf_convention"], proxnf, places=10)
        self.assertEqual(report["reconstruction_range"], [float(recon.min()), float(recon.max())])
        for name, p in points.items():
            truth_curve, curve = self.truth[p["row"], p["col"]], recon[p["row"], p["col"]]
            np.testing.assert_array_equal(curves["points"][name], curve)
            self.assertAlmostEqual(report["point_curve_rrmse"][name],
                                   np.linalg.norm(curve - truth_curve) / np.linalg.norm(truth_curve), places=6)
        lac = ie.lesion_curves(self.weights, self.phases, lambda picks: recon[..., picks])
        for region in self.weights:
            np.testing.assert_allclose(curves["lesion"][region], lac[region])


def add_regions(path, frames):
    """Phases 0-3 by frame and three indicators (file layout x, y, phase) that move one pixel per phase.

    Core and shell move in x, the artery in y; the artery has a partial-volume border (3/7), which
    is not wholly inside. None of the regions is symmetric in x and y.
    """
    core = np.zeros((SIZE, SIZE, 4), np.float32)
    shell, artery = np.zeros_like(core), np.zeros_like(core)
    for p in range(4):
        shell[5 + p:15 + p, 8:16, p] = 1
        shell[8 + p:12 + p, 10:14, p] = 0
        core[8 + p:12 + p, 10:14, p] = 1
        artery[17:23, 2 + p:8 + p, p] = 3 / 7
        artery[18:22, 3 + p:7 + p, p] = 1  # wholly inside at every phase: y = 6, x = 18-21
    with h5py.File(path, "a") as f:
        f["resp_phase"] = np.arange(frames) % 4
        f["indicator/tumor_core"], f["indicator/tumor"], f["indicator/artery"] = core, shell, artery
    return dict(tumor_core=core, tumor=shell, artery=artery)


@unittest.skipUnless(HAVE_CASRT, "caSRT clone not found next to the repository")
class CompleteEvaluationTests(ProximalTestCase):
    @classmethod
    def setUpClass(cls):
        # As InverseTestCase, with the region indicators written before the file's hash is recorded.
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        cls.p0 = write_slice(root / "slice.h5")
        cls.indicators = add_regions(root / "slice.h5", cls.p0.shape[-1])
        cls.data = dict(DATA, file=str(root / "slice.h5"))
        matrix, meta = crt.build_operator(SIZE, FOV, crt.circumscribed_radius(FOV), DR, 1.)
        cls.matrix, cls.meta = matrix, crt.save_operator(root / "op.npz", matrix, meta)
        arrays, info = measurements.generate(cls.data, root / "op.npz", rnl=0.04, seed=0)
        cls.measurements = root / "meas.npz"
        measurements.save_measurements(cls.measurements, arrays, info)

    def test_evaluation_of_a_complete_run(self):
        root = Path(self.folder.name)
        (root / "run.json").write_text(json.dumps(self.config(NESTED)))
        run = root / "inverse_full_20990101_proximal_nested"
        run_main(["--config", str(root / "run.json"), "--out", str(run)])
        out = root / "evaluation"
        ie.main(["--runs", str(run), "--out", str(out)])
        for name in ("inverse_eval.json", "inverse_eval.npz", "point_curves.png", "lesion_curves.png",
                     "per_frame_scores.png"):
            self.assertTrue((out / name).is_file(), name)
        report = json.loads((out / "inverse_eval.json").read_text())
        saved = json.loads((run / "metrics.json").read_text())["image"]["all_window_frames"]
        scores = report["runs"][run.name]
        self.assertAlmostEqual(scores["rrmse"], saved["rrmse"], places=6)
        self.assertAlmostEqual(scores["ssim"], saved["ssim"], places=6)
        self.assertLess(scores["check_against_saved_metrics"]["max_per_frame_rrmse_difference"], 1e-6)
        self.assertEqual(set(scores["point_curve_rrmse"]), {"core", "shell", "artery"})
        self.assertEqual(set(scores["lesion_lac_rrmse"]), {"whole lesion", "shell", "core"})
        window_phases = np.unique(np.arange(2, 11) % 4)
        for name, point in report["points"].items():
            indicator = self.indicators[point["indicator"]]
            # Wholly inside at every phase of the window (file layout x, y, phase).
            np.testing.assert_array_equal(indicator[point["col"], point["row"], window_phases], 1)
        # The truth curve of a point is the slice's p0 at that pixel over the window frames.
        curves = np.load(out / "inverse_eval.npz")
        core = report["points"]["core"]
        np.testing.assert_allclose(curves["point|core|Object"], self.p0[core["col"], core["row"], 2:11], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
