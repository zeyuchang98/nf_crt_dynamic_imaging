# Motion and intensity neural fields

This directory is the implementation baseline for fresh motion/intensity experiments.
The target is a continuous representation of real dynamic objects for PACT reconstruction.
The current runner fits toy images directly; it does not implement PACT measurement reconstruction.

## Start here

- [PROGRESS.md](PROGRESS.md): research decisions, historical comparisons, and limitations.
- [METRICS.md](METRICS.md): formulas and implementation status. Proposed metrics remain unimplemented.
- [BASELINE.json](BASELINE.json): SHA-256 hashes of the 52 active Python files and 48 configuration files.
- [Historical experiment records](results/README.md): configurations, metrics, numeric losses, and loss curves.

The retained Python implementation was verified against pre-reset commit `1d4265e`.
That identifier is provenance only; validate the current files with `BASELINE.json`.
The cleanup removes old outputs and duplicate legacy code; it does not change models,
coordinate conventions, losses, metric definitions, or experiment settings.

## Implementation map

| Responsibility | Location |
|---|---|
| Hash interpolation and coordinate normalization | [models/hash_grid.py](models/hash_grid.py), [models/coordinates.py](models/coordinates.py) |
| Original nested hash model | [models/nested_hash_nf.py](models/nested_hash_nf.py) |
| Low-rank intensity and nested composition | [models/low_rank_intensity.py](models/low_rank_intensity.py), [models/nested_low_rank_hash_nf.py](models/nested_low_rank_hash_nf.py) |
| Direct hash and nested MLP baselines | [models/hash_nf.py](models/hash_nf.py), [models/nested_mlp.py](models/nested_mlp.py) |
| Single training entry point | [run.py](run.py) |
| Numeric losses, learning rates, and loss curves | [training_monitor.py](training_monitor.py) |
| Toy data | [phantoms/](phantoms/) |
| Metrics, branch evaluation, and visualizations | [evaluation/README.md](evaluation/README.md) |
| Reusable experiment settings | [experiments/](experiments/) |
| Existing 36-configuration queue | [hash_sweep.py](hash_sweep.py), [sweep configuration guide](experiments/complex/hash_sweep_24k/README.md) |
| Focused validation | [tests/](tests/) |

The model is `I(r,t) = phi_intensity(r + phi_motion(r,t), t)`.
Motion is a backward displacement to reference coordinates and is exactly zero at
its configured reference time. Low-rank intensity uses a spatial hash basis and a
small temporal MLP: `phi_intensity(q,t) = sum_k b_k(q) a_k(t)`.
The total rank includes any static component. R=2 is the working toy prototype,
not an established rank limit for real objects. See the [model guide](models/README.md).

## Run a fresh experiment

From the repository root, with PyTorch, NumPy, Matplotlib, SciPy, and scikit-image
available in the Python environment:

```bash
python -m unittest discover -s motion_static/tests -v
python -m motion_static.run --help
python -m motion_static.run --phantom complex --model nested_low_rank_hash_nf \
  --config motion_static/experiments/complex/nested_low_rank_hash_nf.json \
  --out motion_static/results/rerun/rank_2
```

This is a usage example, not authorization to launch a new research run. Choose the
configuration and training budget before running. The existing configuration is
unchanged. For R=3, use `--rank 3` and a different output directory. Run the original
nested model with `--model nested_hash_nf` and its corresponding configuration.

Always use a **new output directory**. Historical record directories are incomplete
by design after cleanup and must not be passed to resume/skip validation as if they
still contain trained models and predictions. The trainer already refuses to
silently overwrite an existing output directory.

Every new run still writes final weights, source/configuration provenance, numeric
loss/LR history, fixed training/validation losses, loss curves, predictions, metrics,
and GIFs. This cleanup changes what is versioned, not what training computes or saves.

## Output retention

`results/` is ignored for new untracked files. Previously tracked historical text
records and loss curves remain versioned. Select compact records explicitly for
publication; do not add entire output directories with `git add -f`.
The legacy sweep `--publish` option stages full run artifacts and is incompatible
with this compact retention policy. Use the queue **without `--publish`** when a
new sweep is approved; publication needs an explicit selection of lightweight records.

For each run, retain its configuration, source provenance, metrics, numeric loss/LR
history, and loss curves. Keep model weights, raw arrays, and other figures outside
Git when needed; selected visual references can be added deliberately. Existing large outputs and the old `archive/before_reorganization` code were
removed. The user subsequently requested removal of the old motion-static Git
history as well; those artifacts no longer have a supported recovery path.

## Evaluation conventions

- Image and reference-intensity metrics compare scalar values on their stated grids.
- Motion accuracy compares predicted and true displacement directly. A GIF that
  applies motion to true intensity is a visualization, not the motion error metric.
- `reference_intensity.gif` varies time at fixed reference coordinates.
- `motion_only.gif` freezes learned intensity at the motion reference time.
- `motion_effect.gif` applies true/predicted motion to the same true time-varying intensity.
- Held-out labels are used for evaluation and monitoring, never for optimization.

Historical midpoint and block-split evaluation protocols remain distinct. The
standardization proposals in `METRICS.md` are not silently applied to old results.

The separate `nfCRT/` and `applications/` implementations belong to the original
reconstruction project and are outside this cleanup's implementation scope.

## Synchronize after the history reset

The canonical `motion-static` history was replaced with a single clean root commit.
Old clones cannot update across this boundary with `git pull --ff-only`.
Preserve any uncommitted files from an old clone, then use a fresh clone:

```bash
git clone --depth 1 --single-branch --branch motion-static \
  https://github.com/zeyuchang98/nf_crt_dynamic_imaging.git \
  nf_crt_dynamic_imaging_clean
```

Do not merge or push an old experimental branch back into the clean baseline.
The `main` branch and the original upstream project are separate from this reset.
