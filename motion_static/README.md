# Motion and intensity neural fields

The target is a continuous representation of real dynamic objects for PACT
reconstruction. The current runner fits toy images directly; acoustic measurement
reconstruction and real-object validation are not implemented here.

## Start here

- [experiments/template.json](experiments/template.json): the reusable experiment input.
- [experiments/README.md](experiments/README.md): parameter meanings, overrides and commands.
- [PROGRESS.md](PROGRESS.md): retained research conclusions and current status.
- [METRICS.md](METRICS.md): formulas, current conventions and unimplemented proposals.
- [BASELINE.json](BASELINE.json): hashes of the current validated code and template.

The default phantom is **complex, 64x64, 3 cycles, 379 training frames**. Time spans
[-0.5, 2.5], with 378 unseen midpoint frames. The template retains the existing
rank-3/8k generic configuration; use explicit parameters for a different experiment.
This organization update does not select new hyperparameters or change physics,
losses, coordinate conventions, model architectures or metric definitions.

## Implementation map

| Responsibility | Location |
|---|---|
| Shared model implementations | [models/README.md](models/README.md) |
| Single training entry point | [run.py](run.py) |
| Fixed losses, learning rates and curves | [training_monitor.py](training_monitor.py) |
| Run log, metadata and summary index | [run_records.py](run_records.py) |
| Data and sampling | [phantoms/data.py](phantoms/data.py), [phantoms/complex.py](phantoms/complex.py) |
| Metrics and optional visualization | [evaluation/README.md](evaluation/README.md) |
| Focused validation | [tests/](tests/) |

The nested model is `I(r,t) = phi_intensity(r + phi_motion(r,t), t)`. Motion is a
backward displacement to reference coordinates and is zero at its reference time.
Low-rank intensity is a spatial hash basis times temporal MLP coefficients:
`phi_intensity(q,t) = sum_k b_k(q) a_k(t)`. Total rank includes static intensity.
R=2 is a working toy prototype, not an established bound for real objects.

## Run organization

```text
experiments/
  template.json
  README.md
results/
  summary.csv                  # Generated after runs; can be regenerated
  <UTC_timestamp>_<run_name>/
    config.json                # Effective settings, environment, provenance, status
    metrics.json               # Full existing image/intensity/motion reports
    loss.csv                   # Numeric losses and learning-rate history
    loss.png                   # Loss and learning-rate curves
    model.pt                   # Final weights; not an optimizer resume checkpoint
    monitor_samples.npz        # Exact fixed monitoring indices
    source.zip                 # Actual Python sources, including local edits
    run.log                    # Terminal output and failure details
```

The results directory starts empty except for `.gitkeep`. Old experiment presets,
results and their file links were removed. Historical conclusions remain in
`PROGRESS.md`. New run directories are never overwritten. `config.json` records
running/complete/failed status; failures preserve partial outputs for inspection.
The automatically named directory uses a UTC timestamp with microseconds. `--out`
can explicitly choose another new directory.

Optional artifacts are created only when requested:

- `visuals/*.gif`: `--gifs` or the later `evaluation.render` command.
- `predictions.npz`: `--save-predictions`; one archive, without duplicate per-field copies.
- `checkpoints/step_*.pt`: explicitly selected `--checkpoint-steps`.
- `split.npz`: only for the separate block-holdout protocol, to preserve exact masks.

The final model and source snapshot allow later inference without retraining.
Metrics still evaluate the complete configured grid, independent of the GIF and
array export options. Motion accuracy compares displacement vectors directly;
warping true intensity is a visualization only. Proposed metric changes remain
unimplemented unless explicitly adopted.

## Commands

From the repository root in the existing PyTorch environment:

```bash
python -m motion_static.run --dry-run
python -m unittest discover -s motion_static/tests -v
```

After selecting the parameters and training budget:

```bash
python -m motion_static.run --config /path/to/chosen.json --name selected_run
python -m motion_static.evaluation.render /path/to/run --views intensity motion
python -m motion_static.evaluation.summarize --results motion_static/results
```

`summary.csv` is derived from direct child run folders and includes evaluation
protocol, data settings, parameters and principal metrics. RRMSE values are ratios,
not percentages. Block and midpoint scores are explicitly labeled and must not be
pooled as equivalent evaluations. Optional old comparison utilities are invoked
separately; training never calls them to create extra reports or GIFs.

## Retention and synchronization

Generated results are ignored by Git. Deliberately select compact records for
publication; do not force-add entire run directories. Models and optional arrays
can remain local. Source code, the template and documentation are versioned.
The separate `nfCRT/` and `applications/` work is outside this cleanup.

The earlier history reset replaced the old `motion-static` branch with a clean
root. Old clones from before that reset require a fresh clone after preserving
local changes. Current clones on the reset branch use normal fast-forward pulls.

```bash
git clone --depth 1 --single-branch --branch motion-static \
  https://github.com/zeyuchang98/nf_crt_dynamic_imaging.git nf_crt_dynamic_imaging_clean
```
