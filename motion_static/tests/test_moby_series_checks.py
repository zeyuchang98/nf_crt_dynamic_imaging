"""Whole-series data checks: moving-mask curves, piecewise-linear fits, same-phase SVD, outputs."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from motion_static.evaluation import moby_series_checks as checks


def moving_series(frames=24, phases=4, period=2, size=8):
    """p0 = static body + a moving tissue square whose value follows c(t).

    The square's position depends on the phase modulo `period`, so indicators repeat
    every breath; c(t) is zero before t = 0 and then rises linearly.
    """
    times = -0.5 + 0.1 * np.arange(frames)
    phase = np.arange(frames) % phases
    tissue = np.zeros((size, size, phases))
    for p in range(phases):
        shift = p % period
        tissue[2 + shift:5 + shift, 3:6, p] = 1.
    body = np.full((size, size), 0.2)
    curve = np.maximum(times, 0.) + 0.05
    p0 = body[..., None] + tissue[..., phase] * curve
    return times, phase, tissue, p0, curve


class SeriesCheckTests(unittest.TestCase):
    def test_tissue_curve_follows_the_moving_mask(self):
        times, phase, tissue, p0, curve = moving_series()
        weights = dict(tumor=tissue, liver=np.zeros_like(tissue))
        curves = checks.tissue_curves(p0, phase, weights, block=5)
        self.assertNotIn("liver", curves)  # absent tissues are left out
        np.testing.assert_allclose(curves["tumor"], 0.2 + curve, atol=1e-12)
        np.testing.assert_allclose(curves["whole_image"], p0.mean(axis=(0, 1)), atol=1e-12)

    def test_piecewise_linear_fit_is_exact_on_its_own_nodes(self):
        times = np.linspace(-1., 3., 81)
        nodes = np.linspace(-1., 3., 5)  # 4 cells
        node_values = np.array([0., 2., -1., 1., 1.])
        values = np.interp(times, nodes, node_values)
        np.testing.assert_allclose(checks.piecewise_linear_fit(times, values, 4), values, atol=1e-10)
        # Half as many cells cannot follow the kinks.
        self.assertGreater(np.abs(checks.piecewise_linear_fit(times, values, 2) - values).max(), 0.1)

    def test_same_phase_svd_finds_rank_two(self):
        _, _, tissue, p0, _ = moving_series()
        frames = p0[..., 0::2]  # one breathing phase: fixed geometry
        singular, total, change = checks.svd_summary(frames)
        self.assertLess(total[1], 1e-10)
        self.assertGreater(total[0], 1e-3)
        self.assertEqual(checks.first_rank_below(total, 1e-6), 2)
        self.assertEqual(checks.first_rank_below(change, 1e-6), 2)
        self.assertIsNone(checks.first_rank_below(total, -1.))

    def test_main_writes_report_curves_and_figures(self):
        times, phase, tissue, p0, _ = moving_series()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with h5py.File(root / "slice.h5", "w") as f:
                f["p0"] = p0.astype(np.float32)
                f["time"] = times.astype(np.float32)
                f["resp_phase"] = phase
                for name in checks.TISSUES:
                    f[f"indicator/{name}"] = (tissue if name == "tumor" else np.zeros_like(tissue)).astype(np.float32)
            checks.main([str(root / "slice.h5"), "--out", str(root / "out"), "--breath-frames", "2",
                         "--phases", "0", "1", "--cells", "2", "4", "--zoom", "-1", "2"])
            report = json.loads((root / "out/series_checks.json").read_text())
            self.assertEqual(report["absent_tissues"], ["artery", "liver", "tumor_core"])
            periodicity = report["geometry"]["breath_periodicity"]
            # The square repeats every 2 phases and nothing changes before injection.
            self.assertEqual(periodicity["maximum"], 0)
            self.assertEqual(periodicity["phases_above_1_percent"], [])
            self.assertGreater(periodicity["consecutive_phase_median"], 0.01)
            self.assertEqual(periodicity["indicator_relative_l1_phase_p_vs_p_plus_breath"], {"tumor": 0})
            self.assertEqual(set(report["geometry"]["pre_injection_max_difference_same_phase_frames"].values()), {0})
            self.assertEqual(report["same_phase_svd"]["0"]["rank_for_rrmse_below"]["0.001"], 2)
            for name in ("tissue_curves.csv", "tissue_curves.png", "time_resolution.png", "same_phase_svd.png"):
                self.assertTrue((root / "out" / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
