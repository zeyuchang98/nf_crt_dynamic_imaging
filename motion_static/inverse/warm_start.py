"""POUnet warm start, adapted from ProxNF's code (applications/proximal_splitting).

1. Coefficients: consecutive groups of `frames_per_group` frames are averaged per view
   (with ProxNF's schedule 36 frames see every 1-degree view once), and each group's
   static image is solved by LSQR on the full set of views it saw. The average is the true
   mean of each view's data; ProxNF's code weights it by 10/9 of the mean (line 157), a
   quirk corrected here (agreed 2026-11-18).
   LSQR's `atol`: "proxnf" reproduces the code's 2e2 * sigma / cover (cover = sensor groups
   per frame). It mixes the data scale into a relative tolerance; on our data it exceeds 1,
   so LSQR stops after one iteration (a back-projection scaled by an exact line search).
   "discrepancy" stops at the noise level (grouped_lsqr); a number sets atol directly.
2. Time partitions: psi is first fitted to piecewise-linear hat functions with nodes
   spread evenly over the window (ProxNF's cg_basis / initialize_paritions: 10^4 Adam steps,
   10^3 uniform times per step, rate 1e-4).
3. Each partition's coefficient image is the group images interpolated linearly in time at
   the partition's node (constant beyond the first and last group centers). ProxNF has one
   group per partition and copies the first group into partition 0; with fewer groups than
   partitions, interpolation is the direct generalization.
"""
import numpy as np
import scipy.sparse.linalg as ssla
import torch

from .crt_operator import frame_rows


def group_average(noisy, view_degrees, positions):
    """Mean data of each view seen by the frames at `positions`: (views, rings) and the view angles."""
    degrees = view_degrees[positions].reshape(-1)
    values = noisy[positions].reshape(len(degrees), -1).astype(np.float64)
    views, inverse, counts = np.unique(degrees, return_inverse=True, return_counts=True)
    sums = np.zeros((len(views), values.shape[1]))
    np.add.at(sums, inverse.reshape(-1), values)
    return views, sums / counts[:, None]


def grouped_lsqr(problem, frames_per_group, atol="proxnf", cover=2):
    """Static LSQR images of consecutive frame groups: (groups, pixels), group center times, records.

    atol "discrepancy" stops LSQR once ||A x - b|| <= the expected norm of the averaged data's
    noise, sigma * sqrt(sum over views of rings / count), through LSQR's own test
    ||r|| <= btol ||b|| + atol ||A|| ||x|| with atol = 0 (an option added 2026-11-18: LSQR on
    these data semi-converges, so iterating to convergence fits the noise).
    """
    groups = len(problem) // frames_per_group
    if groups < 1:
        raise ValueError(f"{len(problem)} frames make no group of {frames_per_group}")
    times = problem.times.cpu().numpy()
    images, centers, records = [], [], []
    for group in range(groups):
        positions = np.arange(group * frames_per_group, (group + 1) * frames_per_group)
        views, average = group_average(problem.noisy, problem.view_degrees, positions)
        rows = frame_rows(views, problem.num_rings, problem.angle_step_deg)
        b = average.reshape(-1)
        if atol == "discrepancy":
            counts = np.unique(problem.view_degrees[positions].reshape(-1), return_counts=True)[1]
            noise_norm = problem.sigma * np.sqrt(np.sum(problem.num_rings / counts))
            options = dict(atol=0., btol=noise_norm / np.linalg.norm(b))
        else:
            options = dict(atol=2e2 * problem.sigma / cover if atol == "proxnf" else float(atol))
        solution, stop, iterations, residual = ssla.lsqr(problem.matrix[rows], b, **options)[:4]
        images.append(solution)
        centers.append(times[positions].mean())
        records.append(dict(frames=[int(positions[0]), int(positions[-1])], views=int(len(views)),
                            iterations=int(iterations), stop_reason=int(stop), residual_norm=float(residual),
                            **{key: float(value) for key, value in options.items()}))
    return np.asarray(images), np.asarray(centers), dict(atol_rule=str(atol), groups=records)


def hat_functions(t, nodes):
    """Piecewise-linear interpolation basis with evenly spaced nodes, (points, nodes)."""
    spacing = nodes[1] - nodes[0]
    return torch.relu(1. - (t[:, None] - nodes[None, :]).abs() / spacing)


def fit_partitions(model, steps=10_000, points=1_000, rate=1e-4):
    """Fit psi to hat functions over the window's normalized time range (ProxNF's initialization)."""
    device = model.coefficients.device
    low, high = (float(model.normalized_time(t)) for t in model.time_bounds)
    nodes = torch.linspace(low, high, model.coefficients.shape[1], device=device)
    optimizer = torch.optim.Adam(model.psi.parameters(), lr=rate)
    for _ in range(steps):
        optimizer.zero_grad()
        t = low + (high - low) * torch.rand(points, 1, device=device)
        loss = (model.psi(t) - hat_functions(t[:, 0], nodes)).square().mean() / 2
        loss.backward()
        optimizer.step()
    return nodes, float(loss)


def interpolation_weights(centers, nodes):
    """(groups, nodes) weights: linear in time between group centers, constant outside them."""
    identity = np.eye(len(centers))
    return np.array([[np.interp(node, centers, identity[group]) for node in nodes]
                     for group in range(len(centers))])


def warm_start_pou(model, problem, settings):
    """Set POUnet's psi and coefficients from the measurements; returns a record for config.json."""
    images, centers, record = grouped_lsqr(problem, settings["frames_per_group"], settings.get("lsqr_atol", "proxnf"),
                                           settings.get("cover", 2))
    nodes, partition_loss = fit_partitions(model, settings.get("partition_steps", 10_000))
    center_nodes = model.normalized_time(torch.as_tensor(centers)).numpy()
    weights = interpolation_weights(center_nodes, nodes.cpu().numpy())
    with torch.no_grad():
        model.coefficients.copy_(torch.as_tensor(images.T @ weights, dtype=torch.float32))
    return dict(record, group_centers_s=centers.tolist(), partition_nodes=nodes.cpu().tolist(),
                partition_fit_loss=partition_loss, partition_steps=settings.get("partition_steps", 10_000))
