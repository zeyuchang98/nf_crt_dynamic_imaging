"""psi's time input on the whole series (2026-11-25): why one unit per revolution fails and
whether -1..1 (`pou.time_half_range: 1`) fixes it. Diagnosis only; the truth is not read.

psi change per frame = mean over frames of sum_j |psi_j(t_{k+1}) - psi_j(t_k)| (at most 2),
the measure of the 2026-11-20 diagnosis; the ideal hats of the warm start give 0.0026 on the
whole series (20 partitions, 14,401 frames). "Switches" counts frames where the largest
partition changes (19 for ideal hats in order).

1. initialization: psi before any fitting, for several time ranges (3 seeds); plus the mean
   number of oscillations of the first-layer features sin(2 pi (w t + b)) over the series;
2. warm-start hat fit (inverse/warm_start.fit_partitions, 10^4 Adam steps as in the runs) for
   the configs full_proximal_pou (per revolution) and full_proximal_pou_time1 (-1..1);
3. the finished 2026-11-19 run inverse_full_20261119_proximal_pou (model.pt).

Run from anywhere: python motion_static/results/inverse_pou_time_scale_20261125/psi_time_scale.py
"""
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from motion_static.inverse.warm_start import fit_partitions, hat_functions  # noqa: E402
from motion_static.models.pou_net import POUNet  # noqa: E402
from motion_static.run import build_model, parse_configuration  # noqa: E402

CONFIGS = ROOT / "motion_static" / "experiments" / "inverse"
RUN = ROOT / "motion_static" / "results" / "inverse_full_20261119_proximal_pou"


def psi_scores(model, times):
    with torch.no_grad():
        psi = model.partition(times).numpy()
    return dict(psi_change_per_frame=float(np.abs(np.diff(psi, axis=0)).sum(1).mean()),
                switches=int((np.diff(psi.argmax(1)) != 0).sum()),
                mean_largest_psi=float(psi.max(1).mean()))


def hat_target_scores(model, times):
    low, high = (float(model.normalized_time(t)) for t in model.time_bounds)
    nodes = torch.linspace(low, high, model.coefficients.shape[1])
    hats = hat_functions(model.normalized_time(times), nodes).numpy()
    return dict(psi_change_per_frame=float(np.abs(np.diff(hats, axis=0)).sum(1).mean()),
                switches=int((np.diff(hats.argmax(1)) != 0).sum()))


def resolved(name):
    with tempfile.TemporaryDirectory() as folder:
        _, config, _ = parse_configuration(["--config", str(CONFIGS / name), "--out", str(Path(folder) / "unused")])
    return config


def main():
    report = {}
    reference = resolved("full_proximal_pou.json")
    t0, t1 = reference["coordinate_bounds"][-1]
    times = torch.linspace(t0, t1, reference["frames"])
    pou = reference["pou"]

    # 1. Initialization.
    init = {}
    for half_range in (100., 18., 9., 1.):
        rows = []
        for seed in range(3):
            torch.manual_seed(seed)
            model = POUNet(reference["size"], reference["coordinate_bounds"], (t1 - t0) / (2 * half_range),
                           partitions=pou["partitions"], width=pou["width"], depth=pou["depth"])
            with torch.no_grad():
                first = torch.sin(2 * np.pi * model.psi.input(model.normalized_time(times).reshape(-1, 1))).numpy()
            cycles = float((np.diff(np.sign(first), axis=0) != 0).sum(0).mean() / 2)
            rows.append(dict(psi_scores(model, times), first_layer_oscillations=cycles))
        init[f"-{half_range:g}..{half_range:g}"] = {key: [row[key] for row in rows] for key in rows[0]}
    report["initialization_3_seeds"] = init

    # 2. Warm-start hat fit with the run configs (seed 0 just before building the model).
    fits = {}
    for name in ("full_proximal_pou.json", "full_proximal_pou_time1.json"):
        config = resolved(name)
        torch.manual_seed(config["seed"])
        model = build_model(config)
        start = time.perf_counter()
        _, loss = fit_partitions(model, config["inverse"]["warm_start"]["partition_steps"])
        fits[name] = dict(time_unit_s=config["pou"]["time_unit_s"],
                          range_units=[float(model.normalized_time(torch.tensor(t))) for t in (t0, t1)],
                          fit_loss=loss, after_fit=psi_scores(model, times),
                          ideal_hats=hat_target_scores(model, times), seconds=round(time.perf_counter() - start, 1))
    report["warm_start_hat_fit"] = fits

    # 3. The finished per-revolution proximal POUnet run.
    saved = torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)
    model = build_model(saved["config"])
    model.load_state_dict(saved["state_dict"])
    report["finished_run_inverse_full_20261119_proximal_pou"] = psi_scores(model, times)

    text = json.dumps(report, indent=1)
    print(text)
    (HERE / "psi_time_scale.json").write_text(text + "\n")


if __name__ == "__main__":
    main()
