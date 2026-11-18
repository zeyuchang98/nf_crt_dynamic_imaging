"""Loss curves of reconstruction runs: objective, full data fidelity and truth RRMSE.

At every log step (direct) or outer iteration (proximal) the updated model is rendered on
fixed monitor frames (all frames by default) and scored by the normalized data fidelity
mean(((P f - d) / sigma)^2), about 1 at the noise level, R and the objective
1/2 data fidelity + weight * R. The truth RRMSE on the same frames is recorded for
inspection only; nothing in training reads it. Rows go to loss.csv, curves to loss.png,
with the learning rates; proximal runs also write every inner step to inner_loss.csv.
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


@torch.no_grad()
def mean_regularization(model, problem, frames, regularizer, chunk=8):
    """Mean of R over `frames` (each chunk weighted by its frame count)."""
    if regularizer.get("type", "none") == "none":
        return 0.
    total = 0.
    for start in range(0, len(frames), chunk):
        part = frames[start:start + chunk]
        images = problem.render(model, part).reshape(len(part), problem.size, problem.size)
        total += len(part) * image_regularizer(images, regularizer).item()
    return total / len(frames)


class _Monitor:
    """Shared scoring and CSV writing; subclasses set the columns and the plot."""

    columns = ()

    def __init__(self, out, problem, config):
        self.out = Path(out)
        self.problem = problem
        self.regularizer = config["inverse"]["regularizer"]
        self.frames = monitor_frames(config["inverse"].get("monitor_frames"), len(problem), config["monitor_seed"])
        np.save(self.out / "monitor_frames.npy", self.frames)
        self.rows = []

    def write_header(self):
        with (self.out / "loss.csv").open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.columns).writeheader()

    def scores(self, model):
        scores = self.problem.evaluate(model, self.frames)
        was_training = model.training
        model.eval()
        try:
            regularization = mean_regularization(model, self.problem, self.frames, self.regularizer)
        finally:
            model.train(was_training)
        return dict(scores, regularization=regularization,
                    objective=0.5 * scores["data_fidelity"] + self.regularizer.get("weight", 0.) * regularization)

    def append(self, row):
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"Nonfinite monitor value: {row}")
        self.rows.append(row)
        with (self.out / "loss.csv").open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.columns).writerow(row)
        self.plot()
        return row

    def plot_scores(self, axes, steps, before_key, before_label):
        for key, label, color in ((before_key, before_label, "0.65"),
                                  ("data_fidelity", f"{len(self.frames)} monitor frames (after update)", "C0")):
            axes[0].semilogy(steps, [row.get(key, np.nan) for row in self.rows], label=label, color=color)
        axes[0].axhline(1., color="k", linestyle=":", linewidth=1, label="Noise level")
        axes[0].set(ylabel="mean((Pf - d)^2) / sigma^2", title="Data fidelity")
        axes[1].plot(steps, [100 * row["truth_rrmse"] for row in self.rows], color="C3")
        axes[1].set(ylabel="RRMSE (%)", title="Error against the truth (monitoring only, not used in training)")


class MeasurementMonitor(_Monitor):
    """Direct backpropagation: one row per log step."""

    def __init__(self, out, problem, config, optimizer):
        super().__init__(out, problem, config)
        self.lr_names = [f"next_lr_{group.get('name', i)}" for i, group in enumerate(optimizer.param_groups)]
        self.columns = ["step", "batch_data_fidelity", "batch_regularization", "total_loss", "data_fidelity",
                        "regularization", "objective", "truth_rrmse", "elapsed_seconds", *self.lr_names]
        self.write_header()

    def record(self, model, optimizer, step, elapsed, batch_losses=None):
        row = dict(step=step, elapsed_seconds=elapsed, **self.scores(model))
        if batch_losses is not None:
            row.update(batch_data_fidelity=batch_losses["data_fidelity"],
                       batch_regularization=batch_losses["regularization"], total_loss=batch_losses["total_loss"])
        row.update(zip(self.lr_names, (group["lr"] for group in optimizer.param_groups)))
        return self.append(row)

    def plot(self):
        plt = pyplot(self.out)
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, layout="constrained")
        steps = [row["step"] for row in self.rows]
        self.plot_scores(axes, steps, "batch_data_fidelity", "Random frames (before update)")
        for key in self.lr_names:
            axes[2].plot(steps, [row[key] for row in self.rows], label=key.removeprefix("next_lr_"))
        axes[2].set(xlabel="Optimizer steps", ylabel="Learning rate for next update")
        for ax in axes:
            ax.grid(alpha=.2)
        for ax in (axes[0], axes[2]):
            ax.legend(fontsize=8)
        fig.savefig(self.out / "loss.png", dpi=160)
        plt.close(fig)


class ProximalMonitor(_Monitor):
    """Proximal splitting: one row per outer iteration (row 0: the initial model)."""

    def __init__(self, out, problem, config, group_names):
        """group_names: the inner optimizer's parameter groups, whose constant rates are recorded."""
        super().__init__(out, problem, config)
        self.columns = ["outer", "tau", "gradient_norm", "data_fidelity_before", "cg_iterations",
                        "cg_relative_residual", "inner_loss_first", "inner_loss_last", "data_fidelity",
                        "regularization", "objective", "truth_rrmse", "elapsed_seconds",
                        *(f"lr_{name}" for name in group_names)]
        self.write_header()
        with (self.out / "inner_loss.csv").open("w", newline="") as stream:
            csv.writer(stream).writerow(["outer", "inner_step", "loss", "data_part", "regularization"])

    def record(self, model, outer, elapsed, step=None, inner=None, optimizer=None):
        row = dict(outer=outer, elapsed_seconds=elapsed, **self.scores(model), **(step or {}))
        if optimizer is not None:
            row.update({f"lr_{group['name']}": group["lr"] for group in optimizer.param_groups})
        if inner is not None:
            with (self.out / "inner_loss.csv").open("a", newline="") as stream:
                csv.writer(stream).writerows([(outer, *values) for values in inner])
        return self.append(row)

    def plot(self):
        plt = pyplot(self.out)
        fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True, layout="constrained")
        steps = [row["outer"] for row in self.rows]
        self.plot_scores(axes, steps, "data_fidelity_before", "All frames, before the outer iteration")
        later = [row for row in self.rows if "tau" in row]
        axes[2].semilogy([row["outer"] for row in later], [row["tau"] for row in later], color="C2")
        axes[2].set(ylabel="tau", title="Gradient step (exact line search)")
        for key in self.columns:
            if key.startswith("lr_") and any(key in row for row in later):
                axes[3].plot([row["outer"] for row in later], [row.get(key, np.nan) for row in later],
                             label=key.removeprefix("lr_"))
        axes[3].set(xlabel="Outer iterations", ylabel="Inner learning rate (constant)")
        for ax in axes:
            ax.grid(alpha=.2)
        for ax in (axes[0], axes[3]):
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        fig.savefig(self.out / "loss.png", dpi=160)
        plt.close(fig)
