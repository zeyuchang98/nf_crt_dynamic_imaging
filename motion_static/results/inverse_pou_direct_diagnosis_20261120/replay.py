"""Diagnose the whole-series direct POUnet collapse: replay the start of its training.

Same problem, LSQR group images, hat fit, optimizer, cosine schedule (T_max = the run's 76,000
steps) and step as inverse/direct.py, for the first `steps` steps, logging psi's spread.
Variants change one setting each (diagnosis only, not a proposal for the agreed runs).
Usage: python replay.py OUT_DIR STEPS VARIANT [VARIANT ...]
"""
import copy
import json
import sys
import time

import numpy as np
import torch

from motion_static.inverse.problem import InverseProblem
from motion_static.inverse.warm_start import grouped_lsqr, fit_partitions, interpolation_weights
from motion_static.inverse.monitor import monitor_frames
from motion_static.losses import compute_measurement_losses
from motion_static.run import build_model, build_optimizer

RUN = "motion_static/results/inverse_full_20261119_direct_pou/config.json"
VARIANTS = {
    "as_run": {},
    "psi_lr_1e-5": {"pou_lr": 1e-5},
    "coef_lr_1e-4": {"coefficient_lr": 1e-4},
    "time_unit_360s": {"time_unit_s": 360.0},  # series spans -1..1 units instead of -100..100
    "no_tv": {"tv_weight": 0.0},
    "time360_coef_lr_1e-4": {"time_unit_s": 360.0, "coefficient_lr": 1e-4},
    "time360_psi_lr_1e-5": {"time_unit_s": 360.0, "pou_lr": 1e-5},
}


@torch.no_grad()
def psi_stats(model, times):
    psi = model.partition(times).cpu().double().numpy()
    entropy = -(psi * np.log(psi + 1e-12)).sum(1).mean()
    return dict(psi_entropy=float(entropy), psi_max_mean=float(psi.max(1).mean()),
                psi_range_max=float((psi.max(0) - psi.min(0)).max()),
                psi_frame_change=float(np.abs(np.diff(psi, axis=0)).sum(1).mean()))


@torch.no_grad()
def temporal_spread(model, times):
    frames = model.frames(times)
    return float(frames.std(0).mean() / frames.mean().abs())


def main(out, steps, names):
    base = json.load(open(RUN))
    device = torch.device("mps")
    torch.manual_seed(base["seed"])
    problem = InverseProblem(base, device)
    start = time.perf_counter()
    images, centers, record = grouped_lsqr(problem, 757, "discrepancy", 2)
    print(f"LSQR {len(centers)} groups in {time.perf_counter() - start:.0f} s", flush=True)
    mon = monitor_frames(151, len(problem), base["monitor_seed"])
    all_times = problem.times
    sample = all_times[::48]  # every 48th frame for the image spread
    results = {}
    for name in names:
        config = copy.deepcopy(base)
        change = VARIANTS[name]
        if "pou_lr" in change: config["pou_lr"] = change["pou_lr"]
        if "coefficient_lr" in change: config["coefficient_lr"] = change["coefficient_lr"]
        if "time_unit_s" in change: config["pou"]["time_unit_s"] = change["time_unit_s"]
        if "tv_weight" in change: config["inverse"]["regularizer"]["weight"] = change["tv_weight"]
        regularizer = config["inverse"]["regularizer"]
        torch.manual_seed(config["seed"])
        model = build_model(config).to(device)
        nodes, fit_loss = fit_partitions(model, 10_000)
        weights = interpolation_weights(model.normalized_time(torch.as_tensor(centers)).numpy(), nodes.cpu().numpy())
        with torch.no_grad():
            model.coefficients.copy_(torch.as_tensor(images.T @ weights, dtype=torch.float32))
        optimizer = build_optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["steps"])
        log = []

        def record_row(step):
            row = dict(step=step, **psi_stats(model, all_times), image_temporal_spread=temporal_spread(model, sample),
                       **problem.evaluate(model, mon),
                       coef_min=float(model.coefficients.min()), coef_max=float(model.coefficients.max()))
            log.append(row)
            print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

        print(f"{name}: hat-fit loss {fit_loss:.3g}", flush=True)
        record_row(0)
        model.train()
        for step in range(1, steps + 1):
            frames = torch.randint(len(problem), (4,)).numpy()
            index = torch.as_tensor(frames, device=device)
            images_k = problem.render(model, frames)
            losses = compute_measurement_losses(images_k.reshape(4, problem.size, problem.size),
                                                problem.predicted_data(images_k, frames), problem.data[index],
                                                problem.sigma, regularizer)
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            optimizer.step()
            scheduler.step()
            if step in (100, 300) or step % 1000 == 0:
                model.eval(); record_row(step); model.train()
        results[name] = dict(change=change, hat_fit_loss=fit_loss, log=log)
        json.dump(results, open(f"{out}/replay_{'+'.join(names)}.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
