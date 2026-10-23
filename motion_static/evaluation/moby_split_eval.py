"""Re-score saved dce_moby runs on the alternate validation/test split of the midpoints.

Loads each run's `model.pt` and writes `metrics_alternate_split.json` next to its
`metrics.json`, which is left unchanged. This makes runs trained with all midpoints
held out comparable with plateau-schedule runs on the same test frames.

python -m motion_static.evaluation.moby_split_eval motion_static/results/RUN [RUN ...]
"""
import argparse
import json
from pathlib import Path

import torch

from .evaluate import image_only_report
from .metrics import save_metrics
from ..phantoms import moby
from ..run import build_model, prepare_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    for run in args.runs:
        config = json.loads((run / "config.json").read_text())
        if config.get("phantom") != moby.NAME or config.get("status") != "complete":
            parser.error(f"{run} is not a complete dce_moby run")
        if config.get("held_out_split", "midpoints") != "midpoints":
            parser.error(f"{run} already uses the alternate split; see its metrics.json")
        checkpoint = torch.load(run / "model.pt", weights_only=True)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        split_config = dict(config, held_out_split="alternate_validation_test")
        data, _ = prepare_data(split_config)
        seconds = json.loads((run / "metrics.json").read_text())["training_seconds"]
        report, _ = image_only_report(model, data, split_config, seconds)
        report["note"] = ("Re-scored from model.pt; this run was trained with all midpoints held out, "
                          "so no validation frame steered its training")
        report["model_step"] = checkpoint["step"]
        save_metrics(run / "metrics_alternate_split.json", report)
        test = report["image"]["test_midpoints"]
        print(f"{run.name}: test-frame RRMSE={test['rrmse']:.4%}, PSNR={test['psnr_db']:.2f} dB, SSIM={test['ssim']}")


if __name__ == "__main__":
    main()
