"""psi of the finished whole-series proximal POUnet runs, by the measures of psi_time_scale.py
(psi change per frame, switches of the largest partition; ideal hats 0.0026 and 19).

The 2026-11-19 run (time per revolution, -100..100) repeats psi_time_scale.py's number as a
check; the 2026-11-25 run uses time -1..1 (`pou.time_half_range: 1`). Diagnosis only, the
truth is not read. CPU. Run from anywhere:
    python motion_static/results/inverse_pou_time_scale_20261125/psi_finished_runs.py
"""
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("psi_time_scale", HERE / "psi_time_scale.py")
psi_time_scale = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psi_time_scale)  # adds the repository root to sys.path
from motion_static.run import build_model  # noqa: E402

RESULTS = psi_time_scale.ROOT / "motion_static" / "results"
RUNS = ("inverse_full_20261119_proximal_pou", "inverse_full_20261125_proximal_pou_time1")


def main():
    report = {}
    for name in RUNS:
        saved = torch.load(RESULTS / name / "model.pt", map_location="cpu", weights_only=False)
        config = saved["config"]
        model = build_model(config)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        t0, t1 = config["coordinate_bounds"][-1]
        times = torch.linspace(t0, t1, config["frames"])
        report[name] = dict(time_unit_s=config["pou"]["time_unit_s"], finished=psi_time_scale.psi_scores(model, times),
                            ideal_hats=psi_time_scale.hat_target_scores(model, times))
    text = json.dumps(report, indent=1)
    print(text)
    (HERE / "psi_finished_runs.json").write_text(text + "\n")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
