"""Persist loss curves on fixed samples without changing training RNG or gradients."""
import csv
from pathlib import Path

import numpy as np
import torch

from .evaluation.plots import pyplot


def fixed_subset(coordinates, intensities, count, generator):
    indices = generator.choice(len(coordinates), min(count, len(coordinates)), replace=False)
    return (torch.as_tensor(coordinates[indices], dtype=torch.float32),
            torch.as_tensor(np.asarray(intensities).reshape(-1)[indices], dtype=torch.float32), indices)


@torch.no_grad()
def fixed_mse(model, coordinates, values):
    """Data MSE only; validation labels never contribute to the optimized loss."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    squared_error = 0.
    try:
        for x, y in zip(coordinates.split(8192), values.split(8192)):
            residual = model(x.to(device)) - y.to(device)
            if residual.device.type == "mps":
                residual = residual.cpu()  # MPS has no float64; accumulate on the CPU
            squared_error += residual.double().square().sum().item()
    finally:
        model.train(was_training)
    return squared_error / len(values)


class TrainingMonitor:
    def __init__(self, out, coordinates, intensities, validation, config, optimizer):
        self.out = Path(out)
        # This generator is separate from the torch generator used for batches.
        rng = np.random.default_rng(config.get("monitor_seed", 2026))
        count = config.get("monitor_samples", 65536)
        self.training = fixed_subset(coordinates, intensities, count, rng)
        self.validation = None if validation is None else fixed_subset(*validation, count, rng)
        indices = {"training_indices": self.training[2]}
        if self.validation is not None:
            indices["validation_indices"] = self.validation[2]
        np.savez_compressed(self.out / "monitor_samples.npz", **indices)
        self.lr_names = [f"next_lr_{group.get('name', i)}" for i, group in enumerate(optimizer.param_groups)]
        self.columns = ["step", "batch_mse", "temporal_l2", "total_loss",
                        "fixed_training_mse", "fixed_validation_mse", "elapsed_seconds", *self.lr_names]
        if config.get("schedule") == "plateau":
            # MSE on all validation frames, the quantity that steers the plateau schedule.
            self.columns.insert(self.columns.index("elapsed_seconds"), "validation_frames_mse")
        self.rows = []
        with (self.out / "loss.csv").open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.columns).writeheader()

    def record(self, model, optimizer, step, elapsed, batch_losses=None):
        """Record named scalar batch losses and evaluate the updated model.

        batch_losses maps CSV column names to floats, not tensors. When adding
        a loss term, add its name to self.columns and its curve to plot().
        """
        row = dict(step=step, elapsed_seconds=elapsed,
                   fixed_training_mse=fixed_mse(model, *self.training[:2]))
        if self.validation is not None:
            row["fixed_validation_mse"] = fixed_mse(model, *self.validation[:2])
        if batch_losses is not None:
            row.update(batch_losses)
        row.update(zip(self.lr_names, (group["lr"] for group in optimizer.param_groups)))
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"Nonfinite training monitor value at step {step}")
        self.rows.append(row)
        with (self.out / "loss.csv").open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.columns).writerow(row)
        self.plot()
        return row

    def plot(self):
        plt = pyplot(self.out)
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, layout="constrained")
        steps = [row["step"] for row in self.rows]
        labels = (("batch_mse", "Random training batch (before update)", "0.65"),
                  ("fixed_training_mse", "Fixed training samples (after update)", "C0"),
                  ("fixed_validation_mse", "Fixed held-out samples (after update)", "C1"),
                  ("validation_frames_mse", "All validation frames (steers the plateau schedule)", "C2"))
        for key, label, color in labels:
            values = [row.get(key, np.nan) for row in self.rows]
            if np.isfinite(values).any():
                axes[0].semilogy(steps, np.maximum(values, 1e-16), label=label, color=color)
        if any(row.get("temporal_l2", 0) > 0 for row in self.rows):
            axes[0].semilogy(steps, [row.get("total_loss", np.nan) for row in self.rows],
                             label="Batch total objective", linestyle=":")
        axes[0].set(ylabel="Mean squared error", title="Training convergence: fixed samples, no smoothing")
        for key in self.lr_names:
            axes[1].plot(steps, [row[key] for row in self.rows], label=key.removeprefix("next_lr_"))
        axes[1].set(xlabel="Optimizer steps", ylabel="Learning rate for next update")
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.savefig(self.out / "loss.png", dpi=160)
        plt.close(fig)
