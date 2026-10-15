"""Render historical batch-loss logs without rerunning or inferring missing data."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .plots import pyplot


def plot_saved_losses(run):
    run = Path(run)
    with (run / "loss.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "fixed_training_mse" in rows[0]:
        raise ValueError("Use this renderer only for historical batch-only logs.")
    config = json.loads((run / "config.json").read_text())
    steps = np.asarray([int(row["step"]) for row in rows])
    series = [("Batch image MSE", [float(row["batch_mse"]) for row in rows])]
    weight = config.get("temporal_weight", 0.)
    if weight > 0:
        series.extend([
            ("Weighted temporal penalty", [weight * float(row["temporal_l2"]) for row in rows]),
            ("Batch total objective", [float(row["total_loss"]) for row in rows]),
        ])
    if not np.all(np.diff(steps) > 0):
        raise ValueError("Logged steps must be strictly increasing.")
    for label, values in series:
        if not np.isfinite(values).all() or not np.all(np.asarray(values) > 0):
            raise ValueError(f"Expected finite positive values for {label}.")
    out = run / "plots"
    out.mkdir(exist_ok=True)
    plt = pyplot(out)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for label, values in series:
        ax.semilogy(steps, values, marker="o", markersize=3, linewidth=1.4, label=label)
    name = config["model"].replace("_", " ")
    if "rank" in config:
        name += f" (R={config['rank']})"
    ax.set(xlabel="Optimizer step", ylabel="Logged batch loss",
           title=f"{config['phantom'].title()} / {name}\n"
                 f"{len(rows)} saved observations; no smoothing")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.supxlabel("Retrospective plot from loss.csv. Fixed-sample losses and LR history were not recorded.",
                  fontsize=8)
    path = out / "logged_loss_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    for run in parser.parse_args().runs:
        plot_saved_losses(run)


if __name__ == "__main__":
    main()
