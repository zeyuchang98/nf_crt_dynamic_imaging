"""Compare checkpoint forward/backward cost without optimizer updates or training.

Run after training jobs finish. Alternating measurement order reduces order bias.
This excludes Adam, data loading, monitoring and evaluation, so it is not a
complete reconstruction-time benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .compare_training import load_model


def query_backward(model, coordinates):
    model.zero_grad(set_to_none=True)
    model(coordinates).square().mean().backward()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs=2)
    parser.add_argument("--labels", nargs=2, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=25)
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.warmup, args.rounds, args.iterations) < 1:
        parser.error("Benchmark counts must be positive")
    if len(set(args.labels)) != 2:
        parser.error("Labels must be distinct")
    torch.set_num_threads(args.threads)
    loaded = [load_model(path) for path in args.checkpoints]
    models = [entry[0] for entry in loaded]
    bounds = torch.tensor(loaded[0][1]["config"]["coordinate_bounds"], dtype=torch.float32)
    if loaded[0][1]["config"]["coordinate_bounds"] != loaded[1][1]["config"]["coordinate_bounds"]:
        raise ValueError("Matched queries require identical coordinate bounds")
    generator = torch.Generator().manual_seed(2026)
    coordinates = torch.rand(args.batch_size, 3, generator=generator)
    coordinates = bounds[:, 0] + coordinates * (bounds[:, 1] - bounds[:, 0])
    for model in models:
        for _ in range(args.warmup):
            query_backward(model, coordinates)
    measurements = [[] for _ in models]
    for round_index in range(args.rounds):
        for index in ((0, 1) if round_index % 2 == 0 else (1, 0)):
            start = time.perf_counter()
            for _ in range(args.iterations):
                query_backward(models[index], coordinates)
            measurements[index].append((time.perf_counter() - start) / args.iterations)
    report = dict(device="cpu", threads=args.threads, batch_size=args.batch_size,
                  torch_version=str(torch.__version__), platform=platform.platform(),
                  warmup=args.warmup, rounds=args.rounds, iterations=args.iterations,
                  query_seed=2026, optimizer_updates=0,
                  scope="Full nested model forward, squared-output scalar reduction, backward, and zero_grad. "
                        "No coordinate derivatives, optimizer step, data loading, monitoring or evaluation.",
                  analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), models={})
    for label, path, model, samples in zip(args.labels, args.checkpoints, models, measurements):
        report["models"][label] = dict(checkpoint=str(path.resolve()),
                                      checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                      parameters=sum(p.numel() for p in model.parameters()),
                                      parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                                      seconds_per_batch_by_round=samples,
                                      median_seconds_per_batch=float(np.median(samples)))
    report["second_over_first_time_ratio"] = float(np.median(measurements[1]) / np.median(measurements[0]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
