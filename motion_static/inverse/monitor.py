"""Loss curves of reconstruction runs: batch objective, full data fidelity and truth RRMSE.

At every log step the updated model is rendered on fixed monitor frames (all frames by
default) and scored by the normalized data fidelity mean(((P f - d) / sigma)^2), which is
about 1 at the noise level, and the objective 1/2 data fidelity + weight * R. The truth
RRMSE on the same frames is recorded for inspection only; nothing in training reads it.
Columns go to loss.csv, curves to loss.png, with the learning rates.
"""
import csv
from pathlib import Path

import numpy as np
import torch

from ..evaluation.plots import pyplot
from ..losses import image_regularizer


def monitor_frames(count, total, seed):
    """All frames, or a fixed sorted subset drawn with the monitor seed."""
    if count is None or count >= total:
        return np.arange(total)
    return np.sort(np.random.default_rng(seed).choice(total, count, replace=False))


class MeasurementMonitor:
    def __init__(self, out, problem, config, optimizer):
        self.out = Path(out)
        self.problem = problem
        self.regularizer = config["inverse"]["regularizer"]
        self.frames = monitor_frames(config["inverse"].get("monitor_frames"), len(problem), config["monitor_seed"])
        np.save(self.out / "monitor_frames.npy", self.frames)
        self.lr_names = [f"next_lr_{group.get('name', i)}" for i, group in enumerate(optimizer.param_groups)]
        self.columns = ["step", "batch_data_fidelity", "batch_regularization", "total_loss", "data_fidelity",
                        "regularization", "objective", "truth_rrmse", "elapsed_seconds", *self.lr_names]
        self.rows = []
        with (self.out / "loss.csv").open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.columns).writeheader()

    @torch.no_grad()
    def regularization(self, model, chunk=8):
        """Mean of R over the monitor frames (each chunk weighted by its frame count)."""
        if self.regularizer.get("type", "none") == "none":
            return 0.
        total = 0.
        for start in range(0, len(self.frames), chunk):
            part = self.frames[start:start + chunk]
            images = self.problem.render(model, part).reshape(len(part), self.problem.size, self.problem.size)
            total += len(part) * image_regularizer(images, self.regularizer).item()
        return total / len(self.frames)

    def record(self, model, optimizer, step, elapsed, batch_losses=None):
        scores = self.problem.evaluate(model, self.frames)
        was_training = model.training
        model.eval()
        try:
            regularization = self.regularization(model)
        finally:
            model.train(was_training)
        row = dict(step=step, elapsed_seconds=elapsed, regularization=regularization,
                   objective=0.5 * scores["data_fidelity"] + self.regularizer.get("weight", 0.) * regularization,
                   **scores)
        if batch_losses is not None:
            row.update(batch_data_fidelity=batch_losses["data_fidelity"],
                       batch_regularization=batch_losses["regularization"], total_loss=batch_losses["total_loss"])
        row.update(zip(self.lr_names, (group["lr"] for group in optimizer.param_groups)))
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"Nonfinite monitor value at step {step}")
        self.rows.append(row)
        with (self.out / "loss.csv").open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.columns).writerow(row)
        self.plot()
        return row

    def plot(self):
        plt = pyplot(self.out)
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, layout="constrained")
        steps = [row["step"] for row in self.rows]
        for key, label, color in (("batch_data_fidelity", "Random frames (before update)", "0.65"),
                                  ("data_fidelity", f"{len(self.frames)} monitor frames (after update)", "C0")):
            values = [row.get(key, np.nan) for row in self.rows]
            axes[0].semilogy(steps, values, label=label, color=color)
        axes[0].axhline(1., color="k", linestyle=":", linewidth=1, label="Noise level")
        axes[0].set(ylabel="mean((Pf - d)^2) / sigma^2", title="Data fidelity")
        axes[1].plot(steps, [100 * row["truth_rrmse"] for row in self.rows], color="C3")
        axes[1].set(ylabel="RRMSE (%)", title="Error against the truth (monitoring only, not used in training)")
        for key in self.lr_names:
            axes[2].plot(steps, [row[key] for row in self.rows], label=key.removeprefix("next_lr_"))
        axes[2].set(xlabel="Optimizer steps", ylabel="Learning rate for next update")
        for ax in axes:
            ax.grid(alpha=.2)
        for ax in (axes[0], axes[2]):
            ax.legend(fontsize=8)
        fig.savefig(self.out / "loss.png", dpi=160)
        plt.close(fig)
