"""Time the parts of a reconstruction run on this machine (stage 5 of the 2D inverse plan).

For one inverse configuration (its model, measurements and device), without training to
convergence and without the monitor:
- render: model frames on the whole grid, no gradient (ms per frame);
- direct step: K frames rendered, P_k applied, backward and an Adam update (ms per step);
- gradient step: proximal splitting's forward step over all frames (ms per frame);
- inner step: one proximal Adam step on `batch_size` points, with the configured penalty;
- CG: POUnet's coefficient solve over all frames (ms per call).
Medians of `repeats` timings after one warm-up. Writes a JSON next to the results.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..run import build_model, build_optimizer, parse_configuration
from ..losses import compute_measurement_losses
from .problem import InverseProblem
from . import proximal


def synchronizer(device):
    if device.type == "mps":
        return torch.mps.synchronize
    if device.type == "cuda":
        return torch.cuda.synchronize
    return lambda: None


def median_ms(function, repeats, sync):
    function()  # warm-up (kernel compilation, allocations)
    sync()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        sync()
        times.append(time.perf_counter() - start)
    return 1e3 * float(np.median(times))


def benchmark(config, repeats=5):
    device = torch.device(config.get("device", "cpu"))
    sync = synchronizer(device)
    torch.manual_seed(config["seed"])
    problem = InverseProblem(config, device, adjoint=True)
    model = build_model(config).to(device)
    if config["model"] == "pou_net":
        torch.nn.init.normal_(model.coefficients, std=0.01)  # nonzero frames, no warm start
    settings, record = config["inverse"], dict(model=config["model"], device=str(device), frames=len(problem),
                                                parameters=sum(p.numel() for p in model.parameters()))
    memory = (lambda: torch.mps.driver_allocated_memory() / 2 ** 30) if device.type == "mps" else (lambda: float("nan"))

    frames = np.arange(min(8, len(problem)))
    with torch.no_grad():
        record["render_ms_per_frame"] = median_ms(lambda: problem.render(model, frames), repeats, sync) / len(frames)

    per_step = settings.get("frames_per_step", 4)
    optimizer = build_optimizer(model, config)
    regularizer = settings["regularizer"]

    def direct_step():
        chosen = torch.randint(len(problem), (per_step,)).numpy()
        images = problem.render(model, chosen)
        losses = compute_measurement_losses(images.reshape(per_step, problem.size, problem.size),
                                            problem.predicted_data(images, chosen),
                                            problem.data[torch.as_tensor(chosen, device=device)], problem.sigma,
                                            regularizer)
        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        optimizer.step()

    record["direct_frames_per_step"] = per_step
    record["direct_step_ms"] = median_ms(direct_step, repeats, sync)

    targets = None

    def forward_step():
        nonlocal targets
        targets, _ = proximal.gradient_step(model, problem)

    record["gradient_step_ms_per_frame"] = median_ms(forward_step, max(1, repeats // 2), sync) / len(problem)
    record["gradient_step_memory_gib"] = memory()
    coefficient = 1. if regularizer["type"] != "none" else 0.
    record["inner_batch_points"] = config["batch_size"]
    record["inner_step_ms"] = median_ms(
        lambda: proximal.inner_fit(model, problem, targets, optimizer, 1, config["batch_size"], regularizer,
                                   coefficient), repeats, sync)
    if config["model"] == "pou_net":
        cg = settings.get("coefficient_cg") or dict(max_iterations=100, tolerance=1e-6)
        record["cg_ms"] = median_ms(lambda: proximal.pou_coefficient_cg(model, problem, targets, 1e-3, cg),
                                    max(1, repeats // 2), sync)
    record["mps_driver_memory_gib"] = memory()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", required=True, help="JSON file for the timings")
    args = parser.parse_args(argv)
    _, config, _ = parse_configuration(["--config", args.config, "--out", str(Path(args.out).with_suffix(".unused"))])
    record = dict(config=args.config, **benchmark(config, args.repeats))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
