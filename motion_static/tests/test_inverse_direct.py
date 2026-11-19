"""Check stage 2 of the 2D inverse problem: torch operators, POUnet, the direct objective,
the warm start and complete reconstruction runs on a small synthetic slice."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from motion_static.inverse import crt_operator as crt
from motion_static.inverse import measurements, warm_start
from motion_static.inverse.direct import train_direct
from motion_static.inverse.frame_operator import FrameOperator
from motion_static.inverse.problem import InverseProblem
from motion_static.losses import compute_measurement_losses, image_regularizer
from motion_static.models.pou_net import POUNet
from motion_static.run import main, parse_configuration
from motion_static.tests.test_inverse_data import DR, FOV, HAVE_CASRT, SIZE, write_slice


DATA = dict(first_frame=2, cycles=2, frames_per_cycle=4, intensity_scale="global_maximum")  # frames 2-10
NESTED = dict(model="nested_low_rank_hash_nf", mlp_lr=1e-3, hash_lr=1e-2, rank=2, head_width=16, temporal_width=8,
              encoding=dict(levels=2, features=2, log2_hash_size=10, min_resolution=4, max_resolution=16),
              motion_encoding=dict(levels=2, features=2, log2_hash_size=8, min_resolution=[2, 2, 2],
                                   max_resolution=[4, 4, 4]))
POU = dict(model="pou_net", pou=dict(partitions=4, width=16, depth=2), pou_lr=1e-3, coefficient_lr=1e-3)


@unittest.skipUnless(HAVE_CASRT, "caSRT clone not found next to the repository")
class InverseTestCase(unittest.TestCase):
    """A 24 x 24 slice with 12 frames, its operator and noisy measurements of frames 2-10."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        cls.p0 = write_slice(root / "slice.h5")
        cls.data = dict(DATA, file=str(root / "slice.h5"))
        matrix, meta = crt.build_operator(SIZE, FOV, crt.circumscribed_radius(FOV), DR, 1.)
        cls.matrix, cls.meta = matrix, crt.save_operator(root / "op.npz", matrix, meta)
        arrays, info = measurements.generate(cls.data, root / "op.npz", rnl=0.04, seed=0)
        cls.measurements = root / "meas.npz"
        measurements.save_measurements(cls.measurements, arrays, info)
        cls.arrays = arrays

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def config(self, model, steps=4, **inverse):
        config = dict(phantom="dce_moby", seed=0, steps=steps, batch_size=1024, threads=1, device="cpu",
                      monitor_samples=256, monitor_seed=2026, log_interval=2, checkpoint_steps=[],
                      output=dict(gifs=False, predictions=False), data=dict(self.data), held_out_split="none",
                      evaluation_split="midpoints", **model)
        config["inverse"] = {**dict(method="direct", measurements=str(self.measurements), frames_per_step=2,
                                    regularizer=dict(type="none", weight=0.)), **inverse}
        return config

    def resolved(self, config):
        path = Path(self.folder.name) / f"config_{id(config)}.json"
        path.write_text(json.dumps(config))
        _, resolved, _ = parse_configuration(["--config", str(path), "--out", str(Path(self.folder.name) / "unused")])
        return resolved


class OperatorAndModelTests(InverseTestCase):
    def test_frame_operator_matches_scipy_and_its_adjoint(self):
        views = self.arrays["view_degrees"]
        operator = FrameOperator(self.matrix, views, self.meta["num_rings"], adjoint=True)
        self.assertEqual(len(operator.forward_blocks), len(np.unique(views, axis=0)))
        rng = np.random.default_rng(0)
        images = torch.as_tensor(rng.random((3, SIZE * SIZE)), dtype=torch.float32)
        frames = [0, 4, 8]
        forward = operator.forward(images, frames).numpy()
        for row, k in zip(forward, frames):
            rows = crt.frame_rows(views[k], self.meta["num_rings"])
            np.testing.assert_allclose(row, self.matrix[rows] @ images[frames.index(k)].numpy().astype(np.float64),
                                       rtol=1e-5, atol=1e-5)
        data = torch.as_tensor(rng.random(forward.shape), dtype=torch.float32)
        left = (operator.forward(images, frames) * data).sum()
        right = (images * operator.adjoint(data, frames)).sum()
        self.assertAlmostEqual(left.item() / right.item(), 1., places=5)

    def test_torch_problem_reproduces_the_stored_clean_data(self):
        # The training path (torch, x-fastest rendering) and the simulation path (scipy) agree.
        config = self.resolved(self.config(POU, warm_start=None))
        problem = InverseProblem(config, "cpu")
        truth = torch.as_tensor(problem.truth, dtype=torch.float32)
        frames = np.arange(len(problem))
        np.testing.assert_allclose(problem.predicted_data(truth, frames).numpy(),
                                   self.arrays["clean"].reshape(len(frames), -1), rtol=1e-5, atol=1e-5)
        self.assertLess(problem.evaluate(_Truth(problem), frames)["truth_rrmse"], 1e-6)

    def test_pou_net_partition_grid_and_frames(self):
        bounds = [[0, FOV], [0, FOV], [0., 1.]]
        model = POUNet(SIZE, bounds, time_unit_s=0.5, partitions=4, width=16, depth=2)
        torch.nn.init.normal_(model.coefficients)
        t = torch.linspace(0, 1, 7)
        np.testing.assert_allclose(model.partition(t).sum(dim=1).detach().numpy(), 1., rtol=1e-6)
        centers = (np.arange(SIZE) + 0.5) * FOV / SIZE
        y, x = np.meshgrid(centers, centers, indexing="ij")
        xy = torch.as_tensor(np.stack([x.reshape(-1), y.reshape(-1)], 1), dtype=torch.float32)
        np.testing.assert_array_equal(model.pixel_index(xy).numpy(), np.arange(SIZE * SIZE))
        points = torch.cat([xy.repeat(2, 1), torch.tensor([0.25, 0.75]).repeat_interleave(SIZE * SIZE)[:, None]], 1)
        np.testing.assert_allclose(model(points).reshape(2, -1).detach().numpy(),
                                   model.frames(torch.tensor([0.25, 0.75])).detach().numpy(), rtol=1e-5, atol=1e-6)


class _Truth(torch.nn.Module):
    """A 'model' that renders the problem's truth, to check the scoring path."""

    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def frames(self, times):
        index = [int(np.argmin(np.abs(self.problem.times.numpy() - float(t)))) for t in times]
        return torch.as_tensor(self.problem.truth[index], dtype=torch.float32)


def assert_float32_close(test, streamed, whole, path="report"):
    """Same keys (the streamed settings note aside) and values to float32 rounding."""
    if isinstance(whole, dict):
        keys = set(streamed) - ({"computation"} if path.endswith("settings") else set())
        test.assertEqual(keys, set(whole), path)
        for key in whole:
            assert_float32_close(test, streamed[key], whole[key], f"{path}.{key}")
    elif isinstance(whole, list):
        test.assertEqual(len(streamed), len(whole), path)
        for k, (a, b) in enumerate(zip(streamed, whole)):
            assert_float32_close(test, a, b, f"{path}[{k}]")
    elif isinstance(whole, float):
        test.assertTrue(np.isclose(streamed, whole, rtol=1e-6, atol=1e-9), f"{path}: {streamed} != {whole}")
    else:
        test.assertEqual(streamed, whole, path)


class _Sized:
    """Gives a namespace a length (the frame count), like InverseProblem."""

    def __init__(self, namespace, frames):
        self.__dict__.update(vars(namespace))
        self.frames = frames

    def __len__(self):
        return self.frames


class ObjectiveTests(unittest.TestCase):
    def test_gradient_is_the_scaled_adjoint_residual(self):
        rng = np.random.default_rng(1)
        matrix = torch.as_tensor(rng.random((7, 16)), dtype=torch.float64)
        image = torch.tensor(rng.random((1, 16)), dtype=torch.float64, requires_grad=True)
        data, sigma = torch.as_tensor(rng.random((1, 7))), 0.3
        losses = compute_measurement_losses(image.reshape(1, 4, 4), image @ matrix.T, data, sigma, dict(type="none"))
        losses["total_loss"].backward()
        residual = image.detach() @ matrix.T - data
        expected = residual @ matrix / (sigma ** 2 * residual.numel())
        np.testing.assert_allclose(image.grad.numpy(), expected.numpy(), rtol=1e-12)
        self.assertAlmostEqual(losses["data_fidelity"].item(), (residual / sigma).square().mean().item())

    def test_regularizers_on_a_ramp(self):
        ramp = 0.3 * torch.arange(5.)[None, None, :].repeat(2, 5, 1)  # f = 0.3 ix
        self.assertAlmostEqual(image_regularizer(ramp, dict(type="squared_gradient")).item(), 0.09, places=6)
        self.assertAlmostEqual(image_regularizer(ramp, dict(type="tv", epsilon=1e-4)).item(), 0.3, places=5)
        self.assertEqual(image_regularizer(ramp, dict(type="none")).item(), 0.)


class WarmStartTests(unittest.TestCase):
    def test_group_average_is_the_mean_of_each_view(self):
        views = np.array([[0, 180], [5, 185], [0, 180]], dtype=float)
        noisy = np.stack([np.full((2, 3), k + 1.) for k in range(3)])  # frame k has value k + 1
        degrees, average = warm_start.group_average(noisy, views, np.arange(3))
        np.testing.assert_array_equal(degrees, [0, 5, 180, 185])
        np.testing.assert_array_equal(average[:, 0], [2., 2., 2., 2.])  # views 0, 180: frames 1 and 3

    def test_interpolation_between_group_centers(self):
        weights = warm_start.interpolation_weights(np.array([0., 1., 2.]), np.array([-1., 0.5, 3.]))
        np.testing.assert_allclose(weights, [[1, .5, 0], [0, .5, 0], [0, 0, 1]])


class RunTests(InverseTestCase):
    def run_config(self, config, name):
        path = Path(self.folder.name) / f"{name}.json"
        path.write_text(json.dumps(config))
        out = Path(self.folder.name) / name
        main(["--config", str(path), "--out", str(out)])
        return out, json.loads((out / "config.json").read_text())

    def test_direct_runs_record_losses_metrics_and_provenance(self):
        cases = (("nested", self.config(NESTED)),
                 ("pou", self.config(POU, regularizer=dict(type="tv", weight=0.5, epsilon=1e-5),
                                     warm_start=dict(frames_per_group=4, partition_steps=20, lsqr_atol="proxnf"))))
        for name, config in cases:
            with self.subTest(name):
                out, saved = self.run_config(config, name)
                self.assertEqual(saved["status"], "complete")
                for file in ("loss.csv", "loss.png", "metrics.json", "model.pt", "monitor_frames.npy", "source.zip"):
                    self.assertTrue((out / file).exists(), file)
                with (out / "loss.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([int(row["step"]) for row in rows], [0, 1, 2, 4])
                self.assertIn("truth_rrmse", rows[0])
                problem = saved["inverse"]["problem"]
                self.assertEqual(problem["operator_sha256"], self.meta["operator_sha256"])
                self.assertEqual(problem["frames"], 9)
                metrics = json.loads((out / "metrics.json").read_text())
                self.assertIn("all_window_frames", metrics["image"])
                self.assertTrue(np.isfinite(saved["inverse"]["final_all_frames"]["data_fidelity"]))
        pou = json.loads((Path(self.folder.name) / "pou" / "config.json").read_text())
        record = pou["inverse"]["warm_start_result"]
        self.assertEqual(len(record["groups"]), 2)  # 9 frames, groups of 4
        self.assertEqual(record["groups"][0]["atol"], 2e2 * json.loads(self.measurements.with_suffix(".json")
                                                                       .read_text())["noise"]["sigma"] / 2)

    def test_lsqr_stopping_rules_on_a_static_object(self):
        # A static image seen by 36 frames (every view once), with noise of known sigma.
        views = crt.frame_view_degrees(np.arange(36))
        rings = self.meta["num_rings"]
        image = np.random.default_rng(2).random(SIZE * SIZE)
        clean = measurements.frame_data(self.matrix, np.repeat(image.reshape(SIZE, SIZE, 1), 36, axis=2), views, rings)
        sigma = 0.02 * clean.max()
        noisy, _ = measurements.add_noise(clean, 0.02, seed=5)
        problem = SimpleNamespace(noisy=noisy, view_degrees=views, sigma=sigma, num_rings=rings, angle_step_deg=1.,
                                  matrix=self.matrix, times=torch.arange(36.))
        problem = _Sized(problem, 36)
        _, _, record = warm_start.grouped_lsqr(problem, 36, atol="discrepancy")
        group = record["groups"][0]
        noise_norm = sigma * np.sqrt(360 * rings)  # every view seen once
        self.assertEqual(group["stop_reason"], 1)  # ||r|| <= btol ||b|| met
        self.assertLessEqual(group["residual_norm"], noise_norm * (1 + 1e-9))
        self.assertGreater(group["iterations"], 1)
        _, _, record = warm_start.grouped_lsqr(problem, 36, atol=2.)  # atol > 1 stops at once
        self.assertEqual(record["groups"][0]["iterations"], 1)

    def test_block_scoring_equals_the_whole_array_evaluation(self):
        # Inverse runs score frame block by frame block on the loaded truth; the result must
        # equal image_only_report (whole arrays, moby.make_all_frames) up to rounding.
        from motion_static.evaluation.evaluate import image_only_report
        from motion_static.phantoms import moby
        from motion_static.run import build_model

        out, saved = self.run_config(self.config(NESTED), "block_scoring")
        checkpoint = torch.load(out / "model.pt", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        whole, _ = image_only_report(model, moby.make_all_frames(saved["data"]), saved,
                                     json.loads((out / "metrics.json").read_text())["training_seconds"])
        # The two paths render the same points in different batches (float32), so values agree
        # to float32 rounding rather than bit for bit.
        assert_float32_close(self, json.loads((out / "metrics.json").read_text())["image"],
                             json.loads(json.dumps(whole["image"])))
        self.assertAlmostEqual(saved["inverse"]["final_all_frames"]["truth_rrmse"],
                               whole["image"]["all_window_frames"]["rrmse"], places=6)
        with self.assertRaises(SystemExit):  # predictions would need the whole prediction array
            self.resolved(dict(self.config(NESTED), output=dict(gifs=False, predictions=True)))

    def test_nested_warm_start_fits_intensity_and_keeps_motion(self):
        from motion_static.inverse.warm_start import warm_start_field
        from motion_static.run import build_model

        config = self.resolved(self.config(NESTED, warm_start=dict(frames_per_group=4, fit_steps=6)))
        self.assertEqual(config["inverse"]["warm_start"]["lsqr_atol"], "discrepancy")
        problem = InverseProblem(config, "cpu")
        torch.manual_seed(0)
        model = build_model(config)
        motion = {key: value.clone() for key, value in model.motion.state_dict().items()}
        intensity = {key: value.clone() for key, value in model.intensity.state_dict().items()}
        record = warm_start_field(model, problem, config["inverse"]["warm_start"], config)
        self.assertEqual(len(record["groups"]), 2)  # 9 frames, groups of 4
        self.assertEqual(record["fit_points"], 2 * SIZE * SIZE)
        for key, value in model.motion.state_dict().items():
            self.assertTrue(torch.equal(value, motion[key]), key)
        self.assertTrue(any(not torch.equal(value, intensity[key]) for key, value in model.intensity.state_dict().items()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.motion.parameters()))
        out, saved = self.run_config(self.config(NESTED, warm_start=dict(frames_per_group=4, fit_steps=6)), "nested_ws")
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(saved["inverse"]["warm_start_result"]["fit_steps"], 6)
        with self.assertRaises(SystemExit):  # nested warm starts need fit_steps
            self.resolved(self.config(NESTED, warm_start=dict(frames_per_group=4)))

    def test_training_never_reads_the_truth(self):
        histories = []
        for scale in (1., 5.):
            config = self.resolved(self.config(NESTED, steps=3))
            problem = InverseProblem(config, "cpu")
            problem.truth = problem.truth * scale
            torch.manual_seed(0)
            from motion_static.run import build_model
            model = build_model(config)
            histories.append(train_direct(model, problem, config)[0])
        self.assertEqual(histories[0], histories[1])

    def test_settings_are_checked(self):
        with self.assertRaises(SystemExit):
            self.resolved(dict(self.config(NESTED), held_out_split="midpoints"))
        with self.assertRaises(SystemExit):
            self.resolved(dict(self.config(NESTED), data=dict(self.data, first_frame=1)))  # other window
        with self.assertRaises(SystemExit):
            self.resolved(self.config(NESTED, frames_per_step=20))
        no_inverse = self.config(POU)
        no_inverse.pop("inverse")
        with self.assertRaises(SystemExit):
            self.resolved(no_inverse)


if __name__ == "__main__":
    unittest.main()
