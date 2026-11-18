"""Direct backpropagation through the CRT (the approach of Lozenski et al. 2022; a baseline).

Each step draws K frames uniformly with replacement, renders them on the whole pixel grid
(P_k needs every pixel of a frame, so points cannot be sampled), and minimizes
    1/2 mean_{k, rows} ((P_k f_theta(., t_k) - d_k) / sigma)^2 + weight * R(f_theta)
(losses.compute_measurement_losses) with Adam and the cosine schedule. Autograd applies
P_k^T. The truth never enters the loss; the monitor records it for inspection only.
"""
import time

import numpy as np
import torch

from ..losses import compute_measurement_losses
from .monitor import MeasurementMonitor


def train_direct(model, problem, config, out=None):
    """Optimize `model` in place; returns the logged history and training seconds."""
    from ..run import build_optimizer  # run.py imports this package; import here to avoid a cycle

    model.to(problem.device)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["steps"])
    settings = config["inverse"]
    regularizer, per_step = settings["regularizer"], settings["frames_per_step"]
    log_interval = config.get("log_interval", 500)
    monitor = None if out is None else MeasurementMonitor(out, problem, config, optimizer)
    start = time.perf_counter()
    if monitor is not None:
        monitor.record(model, optimizer, 0, 0.)
    history = []
    model.train()
    for step in range(1, config["steps"] + 1):
        # 1. Random whole frames; the CPU generator gives the same frames on every device.
        frames = torch.randint(len(problem), (per_step,)).numpy()
        frame_index = torch.as_tensor(frames, device=problem.device)

        # 2. Render, simulate the frames' data and compute the objective.
        images = problem.render(model, frames)
        losses = compute_measurement_losses(
            images.reshape(per_step, problem.size, problem.size), problem.predicted_data(images, frames),
            problem.data[frame_index], problem.sigma, regularizer)
        if not torch.isfinite(losses["total_loss"]):
            raise RuntimeError(f"Nonfinite loss at step {step}")

        # 3. Update every parameter group, then the learning rates.
        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        optimizer.step()
        scheduler.step()

        # 4. Log batch losses (before the update) and the monitor (after it).
        if step == 1 or step % log_interval == 0 or step == config["steps"]:
            scalars = {name: value.item() for name, value in losses.items()}
            history.append((step, scalars["data_fidelity"], scalars["regularization"], scalars["total_loss"]))
            text = ""
            if monitor is not None:
                row = monitor.record(model, optimizer, step, time.perf_counter() - start, batch_losses=scalars)
                text = f", all-frame data fidelity {row['data_fidelity']:.6g}, truth RRMSE {row['truth_rrmse']:.4%}"
            print(f"direct/{config['model']} step {step}: data fidelity {scalars['data_fidelity']:.6g}, "
                  f"R {scalars['regularization']:.6g}, total {scalars['total_loss']:.6g}{text}", flush=True)
    return history, time.perf_counter() - start
