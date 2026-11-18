"""Check stage 3 of the 2D inverse problem: proximal splitting (ProxNF's algorithm)."""
import csv
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from motion_static.inverse import crt_operator as crt
from motion_static.inverse import proximal
from motion_static.inverse.problem import InverseProblem
from motion_static.losses import image_regularizer
from motion_static.run import build_model, main
from motion_static.tests.test_inverse_direct import NESTED, POU, InverseTestCase
from motion_static.tests.test_inverse_data import SIZE


def proximal_settings(outer=2, inner=3, **extra):
    return dict(method="proximal", outer_iterations=outer, inner_steps=inner, **extra)


class ProximalTestCase(InverseTestCase):
    def config(self, model, steps=None, **inverse):
        inverse = {**proximal_settings(), **inverse}
        config = super().config(model, steps=inverse["outer_iterations"] * inverse["inner_steps"], **inverse)
        config["inverse"].pop("frames_per_step")
        return config

    def problem_and_model(self, model=POU, **inverse):
        config = self.resolved(self.config(model, warm_start=None, **inverse) if model is POU
                               else self.config(model, **inverse))
        torch.manual_seed(0)
        net = build_model(config)
        if model is POU:
            torch.nn.init.normal_(net.coefficients, std=0.1)
        return config, InverseProblem(config, "cpu", adjoint=True), net


class StepTests(ProximalTestCase):
    def test_gradient_step_is_the_exact_line_search_on_the_data_term(self):
        config, problem, model = self.problem_and_model()
        frames = np.arange(len(problem))
        with torch.no_grad():
            images = problem.render(model, frames).double()
        targets, step = proximal.gradient_step(model, problem)
        data = problem.data.double()
        rows = [crt.frame_rows(views, problem.num_rings) for views in problem.view_degrees]
        # g_k = P_k^T (P_k f_k - d_k) / sigma^2 with scipy, and D(f - alpha g) around tau.
        gradient = np.stack([problem.matrix[r].T @ (problem.matrix[r] @ images[k].numpy() - data[k].numpy())
                             for k, r in enumerate(rows)]) / problem.sigma ** 2
        np.testing.assert_allclose(targets.double().numpy(), images.numpy() - step["tau"] * gradient,
                                   rtol=1e-4, atol=1e-5 * np.abs(images.numpy()).max())

        def misfit(alpha):
            moved = images.numpy() - alpha * gradient
            return sum(np.square(problem.matrix[r] @ moved[k] - data[k].numpy()).sum() for k, r in enumerate(rows))

        self.assertLess(misfit(step["tau"]), misfit(0.9 * step["tau"]))
        self.assertLess(misfit(step["tau"]), misfit(1.1 * step["tau"]))
        self.assertAlmostEqual(step["data_fidelity_before"],
                               misfit(0.) / problem.sigma ** 2 / data.numel(), places=4)

    def test_laplacian_is_d_transpose_d(self):
        rng = np.random.default_rng(3)
        a, b = (torch.as_tensor(rng.random((6, 6, 2))) for _ in range(2))
        energy = (a[:, 1:] - a[:, :-1]).square().sum() + (a[1:] - a[:-1]).square().sum()
        self.assertAlmostEqual((proximal.laplacian(a) * a).sum().item(), energy.item(), places=10)
        self.assertAlmostEqual((proximal.laplacian(a) * b).sum().item(), (a * proximal.laplacian(b)).sum().item(),
                               places=10)

    def test_conjugate_gradient_solves_a_positive_definite_system(self):
        rng = np.random.default_rng(4)
        root = torch.as_tensor(rng.random((8, 8)))
        matrix = root @ root.T + torch.eye(8)
        rhs = torch.as_tensor(rng.random(8))
        solution, iterations, relative = proximal.conjugate_gradient(lambda x: matrix @ x, rhs, torch.zeros(8, dtype=torch.float64),
                                                                     50, 1e-10)
        np.testing.assert_allclose((matrix @ solution).numpy(), rhs.numpy(), rtol=1e-8)
        self.assertLessEqual(iterations, 8 + 2)
        self.assertLess(relative, 1e-10)

    def test_pou_coefficients_minimize_the_proximal_objective(self):
        config, problem, model = self.problem_and_model()
        targets, _ = proximal.gradient_step(model, problem)
        mu = 0.3
        # float32 (as on MPS, and as ProxNF's float32 system) limits the attainable residual.
        record = proximal.pou_coefficient_cg(model, problem, targets, mu, dict(max_iterations=500, tolerance=1e-7))
        self.assertLess(record["cg_relative_residual"], 1e-3)
        coefficients = model.coefficients.detach().clone().requires_grad_(True)
        with torch.no_grad():
            psi = model.partition(problem.times)
        frames = (psi @ coefficients.T).reshape(-1, SIZE, SIZE)
        dx, dy = frames[:, :, 1:] - frames[:, :, :-1], frames[:, 1:] - frames[:, :-1]
        objective = 0.5 * (frames.reshape(len(psi), -1) - targets).square().sum() + mu * (dx.square().sum()
                                                                                           + dy.square().sum())
        objective.backward()
        scale = (targets.T @ psi).abs().max()
        self.assertLess(coefficients.grad.abs().max().item() / scale.item(), 1e-3)

    def test_sampled_regularizer_with_every_point_equals_the_image_mean(self):
        config, problem, model = self.problem_and_model()
        frames, pixels = (index.reshape(-1) for index in torch.meshgrid(
            torch.arange(len(problem)), torch.arange(SIZE * SIZE), indexing="ij"))
        with torch.no_grad():
            values = model(problem.points(frames, pixels))
            images = problem.render(model, np.arange(len(problem))).reshape(-1, SIZE, SIZE)
            for regularizer in (dict(type="squared_gradient"), dict(type="tv", epsilon=1e-4)):
                sampled = proximal.sampled_regularizer(model, problem, frames, pixels, values, regularizer)
                self.assertAlmostEqual(sampled.item(), image_regularizer(images, regularizer).item(), places=6)


class ProximalRunTests(ProximalTestCase):
    def run_config(self, config, name):
        path = Path(self.folder.name) / f"{name}.json"
        path.write_text(json.dumps(config))
        out = Path(self.folder.name) / name
        main(["--config", str(path), "--out", str(out)])
        return out, json.loads((out / "config.json").read_text())

    def test_proximal_runs_record_outer_and_inner_losses(self):
        cases = (("prox_nested", self.config(NESTED)),
                 ("prox_pou", self.config(POU, regularizer=dict(type="squared_gradient", weight=5.),
                                          warm_start=dict(frames_per_group=4, partition_steps=20))))
        for name, config in cases:
            with self.subTest(name):
                out, saved = self.run_config(config, name)
                self.assertEqual(saved["status"], "complete")
                self.assertEqual(saved["steps"], 6)
                with (out / "loss.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([int(row["outer"]) for row in rows], [0, 1, 2])
                self.assertTrue(all(float(row["tau"]) > 0 for row in rows[1:]))
                with (out / "inner_loss.csv").open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 2 * 3)
                for file in ("loss.png", "metrics.json", "model.pt"):
                    self.assertTrue((out / file).exists(), file)
                cg_recorded = rows[1]["cg_iterations"] != ""
                self.assertEqual(cg_recorded, name == "prox_pou")
                if name == "prox_pou":
                    self.assertIn("lr_pou", rows[1])
                    self.assertNotIn("lr_coefficients", rows[1])  # C is solved by CG, psi alone by Adam

    def test_training_never_reads_the_truth(self):
        histories = []
        for scale in (1., 5.):
            config, problem, model = self.problem_and_model(NESTED)
            problem.truth = problem.truth * scale
            histories.append(proximal.train_proximal(model, problem, config)[0])
        self.assertEqual(histories[0], histories[1])

    def test_settings_are_checked(self):
        with self.assertRaises(SystemExit):
            self.resolved(dict(self.config(NESTED), steps=7))  # steps != outer * inner
        with self.assertRaises(SystemExit):
            self.resolved(self.config(POU, warm_start=None, regularizer=dict(type="tv", weight=1., epsilon=1e-5)))
        with self.assertRaises(SystemExit):
            self.resolved(self.config(NESTED, coefficient_cg=dict(max_iterations=10, tolerance=1e-6)))


if __name__ == "__main__":
    unittest.main()
