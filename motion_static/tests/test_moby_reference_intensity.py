"""Reference intensity at reference-configuration frames: frame choice and scores against direct computation."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from skimage.metrics import structural_similarity

from motion_static.evaluation import moby_reference_intensity as reference
from motion_static.evaluation.evaluate import bounding_roi, predict
from motion_static.evaluation.metrics import error_metrics
from motion_static.evaluation.moby_embedding_figures import load_model
from motion_static.phantoms import moby
from motion_static.phantoms.data import coordinate_grid
from motion_static.run import main, prepare_data
from motion_static.tests.test_moby_data import DATA, SIZE, SPACING, write_slice


SSIM_SETTINGS = dict(data_range=1.0, win_size=7, gaussian_weights=False, use_sample_covariance=True,
                     K1=0.01, K2=0.03)
# Offsets from the configuration of phase 2, the phase of t0 in the synthetic window (frame 10).
OFFSETS = {0: 1e-5, 1: 1e-2, 2: 0.0, 3: 2e-4}  # tolerance 1.5e-4 mm: phases 0 and 2 qualify


def write_displacement(path, z_index=7):
    """moby_motion_slice output for one plane: (x, y, 1, 3, phase), phase p shifted by OFFSETS[p] in uy."""
    field = np.random.default_rng(3).normal(size=(SIZE, SIZE, 1, 3)).astype(np.float32)
    displacement = np.stack([field + np.array([0, offset, 0], np.float32) for offset in OFFSETS.values()], -1)
    with h5py.File(path, "w") as f:
        f["displacement"] = displacement
        f["z_index"] = np.array([z_index])
        f.attrs.update(spacing=np.array([SPACING] * 3), source_sha256="synthetic", code_commit="test")


def direct_scores(truth, intensity, crop):
    observed, predicted = truth[crop].astype(np.float64), intensity[crop].astype(np.float64)
    ssim = [structural_similarity(observed[..., k], predicted[..., k], **SSIM_SETTINGS)
            for k in range(observed.shape[-1])]
    return error_metrics(observed, predicted, data_range=1.0), float(np.mean(ssim))


class ReferenceIntensityTests(unittest.TestCase):
    def test_deviation_is_the_largest_vector_difference_in_any_dimension(self):
        rng = np.random.default_rng(0)
        field = rng.normal(size=(4, 5, 6, 3)).astype(np.float32)  # a 3D grid
        displacement = np.stack([field, field, field, field], -1)
        displacement[1, 2, 3, :, 0] += [3e-5, 4e-5, 0]  # length 5e-5
        displacement[..., 3] += 1.0                      # length sqrt(3) everywhere
        with tempfile.TemporaryDirectory() as folder:
            with h5py.File(Path(folder) / "u.h5", "w") as f:
                f["u"] = displacement
                deviation = reference.configuration_deviation(f["u"], reference_phase=1)
        np.testing.assert_allclose(deviation, [5e-5, 0, 0, np.sqrt(3)], rtol=1e-3, atol=1e-7)
        frame_phases = np.array([1, 2, 3, 0, 1, 2, 3, 0])
        np.testing.assert_array_equal(reference.reference_frames(frame_phases, deviation, 1e-4), [0, 1, 3, 4, 5, 7])

    def test_scores_pool_samples_and_average_frame_ssim_on_3d_frames(self):
        rng = np.random.default_rng(1)
        truth = rng.random((9, 8, 10, 3))  # 3D frames (z, y, x), three frames
        intensity = truth + 0.05 * rng.normal(size=truth.shape)
        roi = (slice(1, 9), slice(0, 7), slice(2, 10))
        report = reference.reference_intensity_scores(truth, intensity, roi)
        for region, crop in (("full_grid", ()), ("roi", roi)):
            pooled, ssim = direct_scores(truth, intensity, crop)
            self.assertAlmostEqual(report[region]["rrmse"], pooled["rrmse"], places=12)
            self.assertAlmostEqual(report[region]["psnr_db"], pooled["psnr_db"], places=10)
            self.assertAlmostEqual(report[region]["ssim_mean_of_frames"], ssim, places=12)
            self.assertEqual(len(report[region]["per_frame_ssim"]), 3)

    def test_command_scores_the_intensity_branch_at_qualifying_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "slice.h5"
            write_slice(path)
            runs = {split: self.run_model(root, path, split) for split in ("none", "alternate_validation_test")}
            write_displacement(root / "motion.h5")
            out = root / "scores"
            reference.main(["--displacement", str(root / "motion.h5"), "--nf-runs", str(runs["none"]),
                            "--out", str(out)])
            record = json.loads((out / "reference_intensity.json").read_text())

            self.assertEqual(record["reference_phase"], 2)
            self.assertEqual(record["qualifying_phases"], [0, 2])
            window = moby.window_frames(DATA)
            with h5py.File(path, "r") as f:
                phases = f["resp_phase"][window]
            expected = np.flatnonzero(np.isin(phases, [0, 2]))
            self.assertEqual(record["frames"]["window_indices"], expected.tolist())

            config = json.loads((runs["none"] / "config.json").read_text())
            training = prepare_data(config)[0]["training"]
            truth = training["truth"][..., expected]
            intensity = predict(load_model(runs["none"]).intensity, coordinate_grid(
                training["x"], training["y"], training["t"][expected])).reshape(truth.shape)
            roi = bounding_roi([training["truth"]])
            scores = record["scores"][runs["none"].name]
            for region, crop in (("full_grid", ()), ("roi", roi)):
                pooled, ssim = direct_scores(truth, intensity, crop)
                self.assertAlmostEqual(scores[region]["rrmse"], pooled["rrmse"], places=6)
                self.assertAlmostEqual(scores[region]["ssim_mean_of_frames"], ssim, places=6)

            write_displacement(root / "other_plane.h5", z_index=8)
            rejected = root / "rejected"
            for arguments in (["--nf-runs", str(runs["alternate_validation_test"])],  # not the embedding protocol
                              ["--nf-runs", str(runs["none"]), "--displacement", str(root / "other_plane.h5")],
                              ["--nf-runs", str(runs["none"]), "--data-file", str(root / "other_plane.h5")]):
                with self.assertRaises(SystemExit):
                    reference.main(["--displacement", str(root / "motion.h5"), "--out", str(rejected), *arguments])
            with self.assertRaises(SystemExit):  # outputs are never overwritten
                reference.main(["--displacement", str(root / "motion.h5"), "--nf-runs", str(runs["none"]),
                                "--out", str(out)])

    @staticmethod
    def run_model(root, path, split):
        template = json.loads((Path(moby.__file__).parents[1] / "experiments/template.json").read_text())
        template.update(phantom=moby.NAME, data=dict(DATA, file=str(path)), rank=2, steps=3, batch_size=64,
                        threads=1, monitor_samples=32, log_interval=1, held_out_split=split)
        for key in ("size", "frames", "cycles"):
            template.pop(key)
        config = root / f"{split}.json"
        config.write_text(json.dumps(template))
        out = root / "results" / split
        main(["--config", str(config), "--out", str(out)])
        return out


if __name__ == "__main__":
    unittest.main()
