"""Proximal splitting, ProxNF's algorithm (applications/proximal_splitting), for any model.

The objective is the one of direct backpropagation, so one weight w means one problem:
    J(f) = 1/2 mean_{frames, rows} ((P_k f_k - d_k) / sigma)^2 + w R(f),
R the mean over pixels and frames (losses.image_regularizer). In sum form,
M J = D(f) + M w R(f) with D = 1/2 sum_k ||P_k f_k - d_k||^2 / sigma^2 and M measurements.

Each outer iteration (forward-backward splitting):
1. Forward step on D in image space, every frame, as ProxNF's code:
       g_k = P_k^T (P_k f_k - d_k) / sigma^2,  tau = sigma^2 sum ||g_k||^2 / sum ||P_k g_k||^2
   (the exact minimizer of D along -g), and the target z_k = f_k - tau g_k.
2. Backward (proximal) step within the model, prox of tau M w R divided by the number of
   space-time points:
       min_theta 1/2 mean_points (f_theta - z)^2 + tau (rows / pixels) w R(f_theta),
   by Adam on random space-time points, a fresh optimizer with constant rates per outer
   iteration (as ProxNF). R is estimated from the same points and their +x and +y
   neighbours. For POUnet with `coefficient_cg`, the coefficients are first solved with psi
   fixed by conjugate gradients on the exact quadratic over all frames and pixels (ProxNF's
   coefficient_update; its warm start cast to int32 is corrected), then psi alone is
   updated by Adam.
"""
import time

import numpy as np
import torch

from .monitor import ProximalMonitor


@torch.no_grad()
def gradient_step(model, problem, chunk=8):
    """Targets z (frames, pixels) on the device and the step record (tau, data fidelity before)."""
    was_training = model.training
    model.eval()
    frames, pixels = len(problem), problem.size ** 2
    images = torch.empty((frames, pixels), device=problem.device)
    gradient = torch.empty_like(images)
    misfit = squared_gradient = squared_projection = 0.
    try:
        for start in range(0, frames, chunk):
            part = np.arange(start, min(start + chunk, frames))
            index = torch.as_tensor(part, device=problem.device)
            current = problem.render(model, part)
            residual = problem.predicted_data(current, part) - problem.data[index]
            step = problem.operator.adjoint(residual, part) / problem.sigma ** 2
            projected = problem.predicted_data(step, part)
            images[index], gradient[index] = current, step
            # MPS has no float64: accumulate on the CPU.
            misfit += (residual / problem.sigma).cpu().double().square().sum().item()
            squared_gradient += step.cpu().double().square().sum().item()
            squared_projection += projected.cpu().double().square().sum().item()
    finally:
        model.train(was_training)
    tau = problem.sigma ** 2 * squared_gradient / squared_projection
    images -= tau * gradient
    return images, dict(tau=tau, gradient_norm=float(np.sqrt(squared_gradient)),
                        data_fidelity_before=misfit / (frames * problem.operator.rows_per_frame))


def sampled_regularizer(model, problem, frames, pixels, values, regularizer):
    """Estimate of R from sampled points: +x / +y neighbour differences (losses.image_regularizer)."""
    size, kind = problem.size, regularizer["type"]
    column, row = pixels % size, pixels // size

    def difference(mask, offset):
        mask_device = mask.to(problem.device)
        return model(problem.points(frames[mask], pixels[mask] + offset)) - values[mask_device]

    if kind == "squared_gradient":
        return difference(column < size - 1, 1).square().mean() + difference(row < size - 1, size).square().mean()
    if kind == "tv":
        inside = (column < size - 1) & (row < size - 1)
        return (difference(inside, 1).square() + difference(inside, size).square()
                + regularizer["epsilon"] ** 2).sqrt().mean()
    raise ValueError(f"Unknown regularizer {kind!r}")


def inner_fit(model, problem, targets, optimizer, steps, batch_size, regularizer, coefficient):
    """Adam on 1/2 mean (f - z)^2 + coefficient * R over random space-time points; returns the losses."""
    model.train()
    flat_targets = targets.reshape(-1)
    pixels_per_frame = problem.size ** 2
    use_regularizer = regularizer["type"] != "none" and coefficient > 0
    losses = []
    for step in range(1, steps + 1):
        index = torch.randint(flat_targets.numel(), (batch_size,))  # CPU generator, device independent
        frames, pixels = index // pixels_per_frame, index % pixels_per_frame
        values = model(problem.points(frames, pixels))
        data_part = 0.5 * (values - flat_targets[index.to(problem.device)]).square().mean()
        regularization = (sampled_regularizer(model, problem, frames, pixels, values, regularizer)
                          if use_regularizer else data_part.new_zeros(()))
        loss = data_part + coefficient * regularization
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite proximal loss at inner step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append((step, loss.item(), data_part.item(), regularization.item()))
    return losses


def laplacian(images):
    """D^T D of (size, size, channels) images, forward differences in x and y (no flux at edges)."""
    out = torch.zeros_like(images)
    dx = images[:, 1:] - images[:, :-1]
    out[:, 1:] += dx
    out[:, :-1] -= dx
    dy = images[1:] - images[:-1]
    out[1:] += dy
    out[:-1] -= dy
    return out


def conjugate_gradient(apply, rhs, start, max_iterations, tolerance):
    """Solve apply(x) = rhs (symmetric positive definite) from `start`; returns x, iterations, relative residual."""
    x = start.clone()
    residual = rhs - apply(x)
    direction = residual.clone()
    squared = residual.square().sum()
    rhs_norm = rhs.square().sum().sqrt()
    iterations = 0
    while iterations < max_iterations and squared.sqrt() > tolerance * rhs_norm:
        image = apply(direction)
        alpha = squared / (direction * image).sum()
        x += alpha * direction
        residual -= alpha * image
        new_squared = residual.square().sum()
        direction = residual + (new_squared / squared) * direction
        squared = new_squared
        iterations += 1
    return x, iterations, float(squared.sqrt() / rhs_norm)


@torch.no_grad()
def pou_coefficient_cg(model, problem, targets, mu, settings):
    """POUnet coefficients with psi fixed: (I + 2 mu D^T D) C G = Z^T Psi, G = Psi^T Psi.

    This is the minimizer over C of 1/2 sum_k ||C psi_k - z_k||^2 + mu sum_k ||D C psi_k||^2,
    the proximal objective in sum form (mu = tau * lambda, lambda the sum-form weight of the
    squared gradient). ProxNF solves the same system by CG (at most 100 iterations, 1e-6).
    """
    size, partitions = problem.size, model.coefficients.shape[1]
    psi = model.partition(problem.times)  # (frames, partitions)
    gram = psi.T @ psi
    rhs = (targets.T @ psi).reshape(size, size, partitions)

    def apply(coefficients):
        return (coefficients + 2 * mu * laplacian(coefficients)) @ gram

    start = model.coefficients.detach().reshape(size, size, partitions)
    solution, iterations, relative = conjugate_gradient(apply, rhs, start, settings["max_iterations"],
                                                        settings["tolerance"])
    model.coefficients.copy_(solution.reshape(size * size, partitions))
    return dict(cg_iterations=iterations, cg_relative_residual=relative)


def train_proximal(model, problem, config, out=None):
    """Optimize `model` in place by proximal splitting; returns the outer records and seconds."""
    from ..run import build_optimizer  # run.py imports this package; import here to avoid a cycle

    model.to(problem.device)
    settings = config["inverse"]
    regularizer = settings["regularizer"]
    weight = regularizer.get("weight", 0.)
    rows, pixels = problem.operator.rows_per_frame, problem.size ** 2
    cg = settings.get("coefficient_cg") if config["model"] == "pou_net" else None

    def inner_optimizer():
        """A fresh Adam with constant rates (ProxNF); POUnet with CG updates psi only."""
        if cg is not None:
            return torch.optim.Adam([{"params": model.psi.parameters(), "lr": config["pou_lr"], "name": "pou"}])
        return build_optimizer(model, config)

    names = [group["name"] for group in inner_optimizer().param_groups]
    monitor = None if out is None else ProximalMonitor(out, problem, config, names)
    start = time.perf_counter()
    if monitor is not None:
        monitor.record(model, 0, 0.)
    history = []
    for outer in range(1, settings["outer_iterations"] + 1):
        targets, step = gradient_step(model, problem)
        coefficient = step["tau"] * rows / pixels * weight  # on R in the mean-form proximal objective
        record = dict(step)
        if cg is not None:
            # Sum-form weight of the squared gradient: M w R = lambda sum ||D f||^2, R = sum / (frames N (N - 1)).
            mu = step["tau"] * rows * weight / (problem.size * (problem.size - 1))
            record.update(pou_coefficient_cg(model, problem, targets, mu, cg))
        optimizer = inner_optimizer()
        inner = inner_fit(model, problem, targets, optimizer, settings["inner_steps"], config["batch_size"],
                          regularizer, coefficient)
        del targets
        record.update(inner_loss_first=inner[0][1], inner_loss_last=inner[-1][1])
        history.append(record)
        text = ""
        if monitor is not None:
            row = monitor.record(model, outer, time.perf_counter() - start, record, inner, optimizer)
            text = f", data fidelity after {row['data_fidelity']:.6g}, truth RRMSE {row['truth_rrmse']:.4%}"
        print(f"proximal/{config['model']} outer {outer}: tau {step['tau']:.4g}, data fidelity before "
              f"{step['data_fidelity_before']:.6g}, inner loss {inner[0][1]:.4g} -> {inner[-1][1]:.4g}"
              + (f", CG {record['cg_iterations']} it" if "cg_iterations" in record else "") + text, flush=True)
    return history, time.perf_counter() - start
