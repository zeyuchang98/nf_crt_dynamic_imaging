# Reusable experiment template

`template.json` is the only maintained experiment input. Copy it to a scratch
location, edit explicit settings, and run the shared trainer. Generated sweep
configurations do not belong in this directory. Every run saves its resolved
configuration in its own output folder.

## Data and model settings

The default benchmark is **complex, 64x64, 3 complete cycles, 379 training frames**:
time [-0.5, 2.5], 126 intervals per cycle and 378 unseen midpoint evaluation frames.
Phantom shapes, motion and brightness laws are unchanged.

The template retains the previous generic low-rank configuration's **rank 3,
8,000 steps and eight hash levels from 8 to 128**. These are editable starting
values, not a newly selected best configuration. Motion settings are now explicit;
they equal the previously inherited settings. The earlier rank-2 prototype can be
selected with `--rank 2`. Training remains observed-image MSE with no additional
regularization or branch supervision.

| Setting | Meaning |
|---|---|
| `model` | `nested_low_rank_hash_nf`, `low_rank_hash_nf`, `nested_hash_nf`, `nested_mlp`, or `hash_nf` |
| `rank` | Total rank including static intensity; reference intensity for nested, full image for motion-free |
| `encoding` | Intensity hash levels, features, table size, minimum and maximum resolution |
| `motion_encoding` | Independent overrides for `nested_low_rank_hash_nf`; scalar or `[x,y,t]` resolutions |
| `head_width`, `temporal_width` | Hash head and low-rank temporal MLP widths |
| `steps`, `batch_size`, `seed`, `threads` | Optimizer updates, samples per update, random seed and CPU threads |
| `hash_lr`, `mlp_lr` | Initial Adam rates; cosine schedules end at the selected step budget |
| `monitor_samples`, `monitor_seed`, `log_interval` | Fixed training/validation monitoring pools and recording interval |
| `evaluation_split` | `midpoints` by default; `blocks` remains an explicit separate protocol |
| `checkpoint_steps` | Optional intermediate saves; empty by default |
| `output.gifs` | Save GIFs under `visuals/`; false by default |
| `output.predictions` | Save one `predictions.npz`; false by default |

For `nested_mlp`, explicit `intensity_width` and `motion_width` may be added;
the retained defaults are 128 and 32. The direct `hash_nf` model retains its
built-in architecture and does not consume the nested model's encoding settings.
Model implementations and metric formulas are not changed by this template.

`low_rank_hash_nf` uses the existing low-rank module directly at observation
coordinates, without motion. `--rank` works for both low-rank models. The runner
removes inapplicable `motion_encoding` and `reference_time` fields for the
motion-free model and rejects the nested-only temporal penalty for it.

## Usage

From the repository root, inspect the resolved configuration without training or
creating output files:

```bash
python -m motion_static.run --dry-run
python -m motion_static.run --config /path/to/chosen.json --rank 2 --steps 24000 --dry-run
```

After choosing an experiment, run it explicitly:

```bash
python -m motion_static.run --config /path/to/chosen.json --name comparison_r2
```

CLI values override corresponding JSON fields. `--phantom simple` selects its
original one-cycle/64-frame sampling unless explicit `--cycles`/`--frames` are also
provided. A single-cycle complex comparison must explicitly use `--cycles 1
--frames 64`. The model, rank, step budget and hash settings should be reviewed
before each new comparison; the directory name is not a configuration.

Block holdouts use `block_split: {"block_shape": [8, 8, 2], "test_fraction": 0.2,
"seed": 0}` unless supplied explicitly. The existing block algorithm requires
exact divisibility; 379 frames are incompatible with a temporal block size of 2.
The runner rejects that combination rather than modifying the protocol. Choose
compatible settings only as part of an explicitly agreed block experiment.

## Output and later visualization

Normal runs save `config.json`, `metrics.json`, `loss.csv`, `loss.png`, `model.pt`,
`monitor_samples.npz`, `source.zip` and `run.log`. See the [main guide](../README.md).
`loss.csv` includes batch losses, fixed training/validation losses, elapsed time
and each optimizer group's next learning rate. Monitoring indices are retained
for exact loss rechecks; they never replace the full evaluation set.

```bash
python -m motion_static.evaluation.render /path/to/run --views intensity motion
python -m motion_static.evaluation.render /path/to/run --views image motion_only
```

Rendering uses the saved checkpoint and evaluation times without training. It
checks model/config consistency and numerical source hashes. If code has changed,
use the run's saved `source.zip` in a separate checkout. Existing model, metrics
and loss records are not rewritten by rendering. `motion` means a visualization
with true intensity; direct displacement errors remain in `metrics.json`.

The optional historical 36-setting generator in `hash_sweep.py` remains available
for explicitly requested comparisons. It requires a fresh `--config-dir` outside
`experiments/` and `results/`, records source/config hashes and uses the same flat
outputs. It no longer publishes artifacts or generates GIF comparisons
automatically. No batch experiment was launched during this cleanup.

## Approved baseline comparison (2026-10-14)

The user selected the earlier visually preferred 8-level, 8-to-128 configuration,
with 8,000 updates per run. The five jobs are nested hash, low-rank + motion R=2,
and motion-free low rank at R=8,16,24. All share the complex 64x64, 3-cycle/379-frame
benchmark, seed 0, batch size 4096, CPU with four threads, Adam hash/MLP learning
rates 0.01/0.001, a cosine schedule, head width 64 and temporal width 32 where
applicable. Only observed image MSE is optimized. Monitoring uses seed 2026 and
65,536 fixed samples per split; logs are saved every 500 steps and at step 1.

The shared [baseline script](../baselines.py) prepares explicit inputs outside
this directory and runs serially through the same trainer:

```bash
python -m motion_static.baselines prepare --config-dir /tmp/baselines_20261014_8k
python -m motion_static.baselines run --config-dir /tmp/baselines_20261014_8k
```

Preparation freezes input/source hashes. Execution pulls with `--ff-only` before
each run and stops if code changes or an existing run is incomplete. Completed
matching runs can be reused. Run directories remain flat under `results/`.
GIFs, prediction arrays and intermediate checkpoints stay disabled.

Select the best motion-free rank using the final fixed midpoint validation image
MSE; exact ties choose the smaller rank. There is no early stopping or checkpoint
selection. All candidate records and models are retained, and
`results/baselines_20261014_8k_selection.json` identifies the selected model and
reports full-grid midpoint RRMSE/PSNR/SSIM for all three candidates.
The same midpoints are used for selection and reporting, so these are validation
comparisons, not an independent test. Equal seeds do not imply identical initial
parameters or training batches when model sizes differ; monitoring indices are
identical. This is a single-seed comparison, with no claim of convergence.
