# Motion and Intensity Separation: Current Progress

Last updated: 2026-10-28.

## Current status (2026-10-28)

Summary for continuing on another machine; details and links are in the dated sections
at the end of this file.

- **Data in use:** DCE-MOBY z = 100 slice, window frames 1250-1400 (26.5-34.0 s, three
  breaths, 248 x 248), file `~/Data/dce-moby/moby_z100.h5` (sha256 `89958d21...59da`; the
  Vista copy is in `$WORK/moby_slice/`).
- **Best setting (held-out protocol B):** J5 structure with intensity hash 8-512 (table
  2^19), batch 16,384, rates 0.01/0.001 and the plateau schedule: 11.96% +- 0.28
  test-frame RRMSE, PSNR 51.73 dB, SSIM 0.9985 (776,062 parameters, 3.1 MB checkpoint).
  Cosine 12k steps gives 12.18% +- 0.32 at a third of the time.
- **SS(r) comparison:** on the held-out test frames the NF is slightly better than every
  SS(r) (floor 12.12%, frame interpolation). Under the embedding protocol A (all frames
  fitted and scored) SS(r) is more parameter-efficient: SS(13) 5.18% versus NF 8.86%.
- **Branches:** the learned motion matches the size and direction of the small in-plane
  outline motion (about 1.5 px) and halves the outline mismatch; the intensity branch
  still carries organ-edge changes, possibly from out-of-plane motion.
- **Code:** `device` option (cpu, cuda, mps; CUDA verified on Vista 2026-10-26), `held_out_split`
  (midpoints, alternate_validation_test, none), plateau schedule, SS(r) baseline,
  embedding figures, branch GIFs (with frame subsets), whole-series data checks,
  time-chunked SSIM, Vista job scripts. 137 tests pass in this machine's `nf-crt` conda
  environment (Python 3.11, torch 2.14.0, numpy 1.26.4, h5py 3.16.0); 136 passed earlier
  in the second machine's `.venv-hash-sweep` and Vista's `$WORK/envs/nf-crt`.
- **Local only on the second machine (not on GitHub):** `model.pt` of the DCE-MOBY runs
  (33 files, 116 MB) and `predictions.npz` (14 files, 454 MB). Re-scoring, branch GIFs
  and the embedding figures need `model.pt`.
- **Full-series run on Vista (planning, 2026-10-26):** agreed so far: protocol A, no
  model change, rank from a data-only check; a breathing-phase motion input (M2) is
  recorded as an idea only. Data-only checks done: the contrast change has rank 3,
  the kinetics are slow (fastest rise 23 s) and breathing repeats only approximately
  every 50 frames. Adopted: R = 3 and an intensity time grid of at most 288 cells, on
  all 14,401 frames. Time-chunked SSIM is implemented and the GPU works on Vista
  (`$WORK/envs/nf-crt`). A 10k-step trial of the draft design (job 1102639, 18 min on
  one GH200) gave 10.61% all-frame RRMSE, 54.4 ms per step at batch 65,536 and a
  67.7 GB host-memory peak; the per-frame error depends strongly on the breathing
  phase. SS(r) on the full series (job 1104733) is done: 4.82% RRMSE at r = 17, the
  rank closest to the NF's parameter count; the trial lies between SS(6) and SS(7).
  The formal run (288k steps, job 1104729) was still queued on 2026-10-28, with the
  branch GIFs and the embedding figures (1104764-1104766) waiting for it.
  The source data include `motion_vectors.h5`. See the last section.
- **Open, to decide with the user:** more spatial compression
  for 3D (Umberto, 2026-10-25: e.g. hash-table size); the NF budget under protocol A;
  POUnet as a later baseline.

## Goal and current baseline

Represent real dynamic objects for PACT reconstruction, including both motion and
genuine brightness changes. Current experiments fit toy images directly; no PACT
acoustic forward operator or real measurements have been evaluated here.

The default data are the **complex 64x64 phantom, 3 complete cycles, 379 training
frames**, time [-0.5, 2.5], 126 intervals per cycle, and 378 unseen midpoint frames.
The phantom has an elliptical body, two spots, x/y translation, unequal x/y
expansion, and global/local brightness changes. The multicycle extension retains
these dynamics and doubles the original temporal sampling density.

The maintained input is [experiments/template.json](experiments/template.json).
It retains the previous generic configuration's rank 3, 8k steps and eight-level
8-128 hash settings. These are editable starting values, not a new best-setting
claim. R=2 remains the working research prototype; select it explicitly when
planning a run. Rank, budget and architecture changes require an agreed experiment.

## Current best results (2026-10-15)

Best joint models (both branches learned, no ground-truth branch), with the earlier
joint baseline for reference. Details are in the dated sections below.

| Metric | Joint baseline | J1 | J3 | **J5 (best)** |
|---|---:|---:|---:|---:|
| Motion cells space / time; time network; steps | 8-32 / 8-32; Tanh MLP; 8k | 2-8 / 8-32; grid 96; 8k | 2-8 / 4-16; grid 48; 8k | **2-8 / 4-16; grid 48; 24k** |
| Image RRMSE / PSNR (dB) / SSIM | 5.36% / 46.11 / 0.998119 | 1.51% / 57.12 / 0.999881 | 1.63% / 56.43 / 0.999839 | **0.92% / 61.42 / 0.999939** |
| Ref. intensity RRMSE / PSNR (dB) / SSIM | 52.03% / 27.87 / 0.880784 | 40.61% / 30.02 / 0.921483 | 22.51% / 35.15 / 0.971153 | **9.61% / 42.54 / 0.994251** |
| Ref. center / spot curve RRMSE | 79.45% / 22.37% | 1.20% / 0.83% | 0.81% / 0.74% | **0.36% / 0.36%** |
| Motion EPE RMSE / mean / max (px) | 4.860 / 4.188 / 15.21 | 1.076 / 0.968 / 3.22 | 0.636 / 0.581 / 2.04 | **0.336 / 0.291 / 2.04** |
| Motion vector RRMSE / folding | 80.07% / 31.39% | 17.72% / 0 | 10.48% / 0 | **5.53% / 0** |
| Fixed training / validation MSE | 2.560e-05 / 2.613e-05 | 1.976e-06 / 1.783e-06 | 2.282e-06 / 2.128e-06 | **7.412e-07 / 7.272e-07** |
| Parameters / training seconds | 224,372 / 265 | 85,102 / 223 | 79,336 / 264 | 79,336 / 876 |

Links: [baseline](results/baselines_20261015_8k_low_rank_motion_r2_m32/metrics.json),
[J1](results/joint_tuning_20261015_8k_j1_mt8-32_grid96/metrics.json),
[J3](results/joint_tuning_20261015_8k_j3_mt4-16_grid48/metrics.json),
[J5](results/joint_tuning_20261015_24k_j5_mt4-16_grid48/config.json) (config) /
[metrics](results/joint_tuning_20261015_24k_j5_mt4-16_grid48/metrics.json).
Zero-motion foreground EPE RMSE is 6.070 px.

Single-branch diagnostic upper bounds use one ground-truth branch, so they are not
reconstruction methods:
[E4](results/intensity_fit_20261015_8k_e4_grid_c3-96/metrics.json) (true motion,
time grid 3-96) gives image 0.42% and reference intensity 0.35%;
[M1c](results/motion_fit_20261015_8k_m1c_s2-8_t8-96/metrics.json) (true
intensity, motion cells [2,2,8]-[8,8,96]) gives image 0.42%, EPE 0.129 px and no
folding.

**J5 run name** `joint_tuning_20261015_24k_j5_mt4-16_grid48`:
- `joint`: both branches learned.
- `tuning`: part of the config-only reference-frame tuning series.
- `24k`: 24,000 updates of 4,096 sampled space-time points; the cosine rate ends
  at step 24,000.
- `j5`: fifth run in that series.
- `mt4-16`: motion hash time axis, 4 to 16 cells over [-0.5, 2.5] in eight levels
  (about 5.3 cells, or 24 frames per cell, per cycle at the finest level).
- `grid48`: the intensity time network is a six-level 1D grid with 3 to 48 cells
  (16 cells per cycle at the finest level).
- Not in the name: complex 64x64, 379 frames, R=2, seed 0, motion space 2-8
  cells, intensity spatial hash 8-128, hash/MLP rates 0.01/0.001, image MSE only,
  t0=-0.5. The `config.json` is authoritative; the name is only a label.

**Metric domains:**
- All RRMSE, PSNR, SSIM, curve and motion values here are from the 378 unseen
  midpoint frames (`unseen_time_midpoints`, `motion.midpoint`).
- Image and reference intensity cover the full 64x64 grid (ROI values are
  nearly identical). Motion EPE, vector RRMSE and folding use the true
  foreground mask.
- Fixed training/validation MSE are on fixed 65,536-point subsamples of the
  training frames and midpoint frames.
- The ad hoc first-frame error quoted below for the jump uses training frame t0.

**Limits:** single seed. J5's reference intensity is just under 10%, and a small
reference-frame jump at t0 remains (first-frame image error 8.77%).

## Representation and conventions

$$
q=r+\phi_{\mathrm{motion}}(r,t),\qquad
\hat I(r,t)=\phi_{\mathrm{intensity}}(q,t),\qquad
\phi_{\mathrm{intensity}}(q,t)=\sum_{k=1}^{R} b_k(q)a_k(t).
$$

The spatial hash produces R bases; a small temporal MLP produces R coefficients.
R is the total rank, including static intensity. There is no extra residual or
nonlinearity after the inner product. Motion is a backward displacement to the
reference frame, anchored by subtracting its output at t0=-0.5, so it is exactly
zero at the first frame. Reference intensity is evaluated at fixed q coordinates.

Only observed image values supervise training. True motion, reference intensity
and held-out labels are for evaluation/monitoring. A step is one optimizer update
on a sampled batch, normally 4,096 space-time points. The main objective remains
image MSE, without branch supervision or an added regularizer.

[METRICS.md](METRICS.md) contains formulas and implementation status. Motion accuracy
compares displacement vectors directly; applying motion to true intensity is for
visualization. Existing pixel endpoint RMSE, mean EPE, vector RRMSE and scalar
RRMSE/PSNR/SSIM remain implemented. Proposed dynamic-intensity RRMSE, physical/FOV
units and unified spatial-frame SSIM remain unimplemented. Historical midpoint
volume-SSIM and masked block spatial-SSIM are different protocols.

## Historical findings retained after artifact deletion

The following numbers summarize previously completed experiments. Old run folders,
configs, metrics files and loss curves were deleted at the user's request during
the current cleanup. These are historical conclusions, not newly reproduced
results or links to available models. Local and second-machine runs remain distinct.

### Nested models and low-rank intensity

Single cycle, 64 training frames, 63 unseen midpoints, seed 0, 8k steps:

| Model | Image RRMSE | Intensity RRMSE | Foreground motion endpoint RMSE |
|---|---:|---:|---:|
| Direct Hash NF | 4.03% | — | — |
| Nested MLP | 13.53% | 98.72% | 6.553 px |
| Nested Hash NF | 3.40% | 98.99% | 6.421 px |
| Low-rank R=2 | 5.16% | 46.65% | 4.775 px |
| Low-rank R=3 | 5.31% | 52.04% | 4.817 px |

The zero-motion baseline was 6.066 px. Original nested models fit images while
intensity retains geometry changes. Low rank improves branch errors but does not
establish correct separation. R=2 was slightly better in this single-seed comparison.

The user preferred the early R=2 intensity GIF (the fourth reviewed example).
That run used eight 8-128 levels in both encoders, two features per level, capacity
2^15 per level, head width 64, temporal width 32, Adam hash/MLP rates 0.01/0.001 and
an 8k cosine schedule. This visual preference does not verify the true temporal
waveform. The GIF and model are no longer retained in the current worktree.

### More cycles, denser frames and longer training

Three cycles, 379 training frames, seed 0, R=2:

| Independent budget | Image RRMSE | Intensity RRMSE | Motion RMSE | Fixed train / validation MSE |
|---|---:|---:|---:|---|
| 8k | 2.46% | 63.87% | 5.473 px | 4.02896e-6 / 4.96784e-6 |
| 32k | 1.88% | 68.66% | 5.484 px | 1.09396e-6 / 2.83704e-6 |

Each run had its own cosine schedule; the 32k run did not continue the 8k model.
Image fitting improved without recovering true branches. No sustained validation
loss increase was observed; the late plateau with decaying learning rate does not
prove complete convergence. Cycle count and density changed together, and the
normalized temporal grid changed relative to physical cycles, so their effects
were not isolated. Single- and multicycle aggregate errors cover different domains.

### Hash-grid tuning

At 32k steps, reducing motion spatial resolution from 8-128 to 2-32 while retaining
its time resolution improved motion RMSE from 5.484 to 4.339 px and intensity RRMSE
from 68.66% to 55.06%, while image RRMSE increased from 1.88% to 3.14%.

With this motion grid, the eight-level intensity max-64 run had the lowest image
and intensity RRMSE among max 128/64/32: 3.13% / 35.10%. Max 32 gave 3.60% / 42.04%
and motion RMSE 3.29 px. Lower maximum resolution was not uniformly better.

The 24k sweep varied motion levels 4/8/12 and max 16/32, and intensity levels
4/8/12 and max 32/64, for 36 combinations. All used the three-cycle data and R=2.
The local machine completed 10; the second machine completed all 36.

| Source and distinction | Motion levels/max | Intensity levels/max | Image RRMSE | Intensity RRMSE | Motion RMSE |
|---|---|---|---:|---:|---:|
| Local: lowest intensity and motion errors | 4/32 | 8/64 | 5.1053% | 28.5814% | 2.7565 px |
| Second machine: lowest image error | 12/32 | 8/64 | 2.5825% | 33.8151% | 3.4273 px |
| Second machine: lowest intensity/motion errors | 4/32 | 8/64 | 5.0667% | 28.8133% | 2.8088 px |

The local low-error intensity prediction was nearly static: center brightness
0.3315-0.3386 versus truth 0.21-0.35, and spot brightness 0.7879-0.8300 versus truth
0.4056-0.9744. Lower aggregate error alone does not establish better intensity
dynamics. Reliable separation remains unresolved. These are single-seed toy
comparisons, not real-object validation or an independent generalization test.

### Earlier auxiliary experiments

The simple translating disk also showed accurate image fitting with poor branch
recovery. Its 8k Nested MLP result had 2.55% image RRMSE, 97.71% intensity RRMSE and
7.874 px motion RMSE.

The separate 64-frame block-holdout study used 8x8x2 blocks and 20.02% held-out
pixels. Nested Hash NF had 11.09% test image RRMSE, 98.20% reference-intensity RRMSE
and 6.475 px motion RMSE; the zero-motion control was 5.974 px. This is missing-region
completion, not the unseen-frame midpoint protocol.

An earlier non-low-rank, single-cycle temporal-L2 experiment used weight 1e-4 and
time difference 1/63 at fixed, detached warped q. Image RRMSE changed 3.40% to
5.79%, intensity RRMSE 98.99% to 85.72%, and motion RMSE 6.421 to 5.925 px. Estimated
true brightness variation was suppressed. This loss is not adopted for the default
low-rank setup.

## Decisions and unresolved directions

- Preserve both geometry and true brightness dynamics; a low image loss alone is insufficient.
- R=2 is a useful toy prototype, not a proven rank bound for real PACT objects.
- No Jacobian training penalty was adopted; existing auxiliary evaluation diagnostics remain.
- New motion/intensity separation losses and alternative intensity coordinates remain discussion only.
- Proposed standardized metrics in `METRICS.md` require review before implementation.
- Future training must retain numeric losses, learning rates, fixed training/held-out monitoring and a loss curve.
- Oracle diagnostics (2026-10-15, below): with true motion the R=2 intensity still learns nearly static
  brightness, and with true intensity the motion branch still misses most displacement.
- With true motion, an optional 1D multi-resolution time grid (R=2 hash space x time network kept)
  fits the intensity to 0.35% RRMSE (below). It has not been tried with learned motion.
- With true intensity, a spatially coarse motion hash (x,y 2-8, t 8-96; same single 3D network)
  recovers motion to 0.13 px foreground endpoint RMSE with no folding (below).
- Joint training with both fixes (below) removes folding and recovers the brightness waveforms, but
  selects a different reference frame (about 1.4x / 1.5x larger) everywhere except at t0, via a jump
  right after t0.
- Config-only joint tuning (below): coarser motion time resolution (t 4-16) removes most of that
  reference-frame shift. At 24k steps all raw criteria are met (reference intensity 9.61%, motion
  0.34 px, no folding, image 0.92%); a small residual jump at t0 remains. The anchor definition and
  evaluation protocol are unchanged.

## Current organization cleanup

The user approved removal of old experiment inputs and results, one reusable
template, flat run folders, retained monitoring indices and opt-in GIFs. Removed
50 old experiment files and 593 old result files (about 66.6 MiB of results).
Research conclusions above replace obsolete links; no old results were rerun.

Training now records config/environment/source/status, one complete metric JSON,
loss CSV/PNG, final model, monitoring indices, actual source ZIP and console log.
Optional exports are one prediction archive, selected intermediate checkpoints
and GIFs in `visuals/`. Block experiments additionally preserve their exact split.
`summary.csv` is derived from flat run records with explicit protocol labels.
A dry-run command previews settings without creating files or training.

The old sweep helper now requires explicitly generated scratch inputs and uses
flat runs. It no longer publishes results or creates plots/GIF comparisons
automatically. Shared models, phantom dynamics and numerical metric definitions
are retained. Unrelated `nfCRT/` and `applications/` edits are preserved.

**Validation completed:** all 79 unit tests passed. Default and opt-in exports
produce identical metrics; later rendering leaves the model, metrics, loss and
monitoring samples unchanged. Inference-only comparisons against the pre-cleanup
evaluation code reproduced every existing midpoint and block metric, including
ux/uy reports, exactly on small fixtures. The template preserves the previous
generic model's parameter layout and seeded initialization exactly. Dry-run and
failure recording checks passed; local documentation links resolve. Refreshed
`BASELINE.json` and confirmed unrelated files are byte-identical.

No research experiment or parameter search was launched. New integration fixtures
used initialized models without optimizer updates; existing numerical unit tests
retain their small synthetic optimizer checks. These checks establish software
consistency, not model quality on the new benchmark.

## Training readability refactor

`run.train()` now reads as sampling, objective calculation, optimizer update and
recording. Optimizer groups are built separately. `losses.compute_training_losses()`
assembles named scalar terms and their weighted total; the monitor receives names
instead of relying on tuple order. Run metadata, source snapshots and checkpoints
are in `run_records.py`, whose summary extraction explicitly separates midpoint
and block report fields. The README documents where to edit an agreed future loss.
Current formulas, weights, defaults, logging intervals and output schemas are unchanged.

**Validation completed:** 81 tests passed, including analytic objective/gradient
checks and verification that zero temporal weight skips that penalty's queries.
Compared with commit `208b8ffc4ec0c1b2232a0b174ee506ed822fe541`, seven four-step
synthetic checks covered all four model families and both zero/nonzero temporal
weight for nested models. Loss history, weights, RNG, optimizer/scheduler checkpoint
state, monitoring indices, console values and loss CSV values matched exactly
(elapsed time excluded). Sixteen summary-row comparisons also matched. Refreshed
`BASELINE.json`; results remain empty except for `.gitkeep`. No research experiment
or parameter search was launched, and no new scientific conclusion is implied.

## Approved baseline comparison: completed

The user approved local CPU runs with the earlier visually preferred 8-level
8-128 grids and 8k updates: nested hash, low-rank + motion R=2, and motion-free
low rank at R=8,16,24. All use the complex 64x64, 3-cycle/379-frame benchmark,
seed 0, batch 4096, Adam hash/MLP rates 0.01/0.001 and the existing cosine schedule.
The three motion-free candidates are ranked by final fixed midpoint validation
image MSE. Full midpoint image RRMSE/PSNR/SSIM are reported for all; these are
validation results after rank selection. No extra seeds or rank expansion is approved.

Added `low_rank_hash_nf` by directly reusing `LowRankIntensity` at observation
coordinates. The runner, optimizer and checkpoint loader now support it; its
metrics report full-image rank and image quality, without invented motion or
reference-intensity branch scores. Existing nested models and loss formulas are
unchanged. `baselines.py` prepares the exact five inputs in scratch storage,
checks code/config hashes and runs serially into flat result folders, retaining
all records and identifying the selected rank. GIFs remain opt-in.

**Preflight verified:** 87 tests passed, including all five actual model settings,
rank bounds, coordinate gradients, reference anchors where applicable, optimizer
parameter coverage, three-step synthetic motion-free updates, checkpoint reload,
image-only evaluation and validation-based selection. A fresh agent executed the
documented seeded runtime witness successfully. Existing local Python 3.11.16 /
PyTorch 2.14.0 environment is reused on CPU; no dependencies changed.

All five runs completed on 2026-10-14 local. The first used commit
`4c74bf9a527860e251b1027b42914a63e772cb67`; the remaining runs used `75343b6`,
which only updated this progress record. Every run's Python source hashes are
identical. Scratch inputs/queue log are under `/tmp/baselines_20261014_8k`;
resolved configurations, numerical results and curves are retained below.

### Complete midpoint image comparison

All rows use the same 378 midpoint frames, data range 1 and existing sequence
SSIM convention. RRMSE is displayed as a percentage. Training time includes
fixed monitoring and plotting, and excludes the final full-grid evaluation.

| Model and full metrics | Image RRMSE | PSNR (dB) | SSIM | Fixed validation MSE | Train (s) | Loss/LR curve |
|---|---:|---:|---:|---:|---:|---|
| [Nested hash](results/baselines_20261014_8k_nested_hash/metrics.json) | 1.1988% | 59.1210 | 0.999882 | 1.2356603e-06 | 281.23 | [PNG](results/baselines_20261014_8k_nested_hash/loss.png), [CSV](results/baselines_20261014_8k_nested_hash/loss.csv) |
| [Low rank + motion, R=2](results/baselines_20261014_8k_low_rank_motion_r2/metrics.json) | 2.5181% | 52.6742 | 0.999630 | 5.3106635e-06 | 277.42 | [PNG](results/baselines_20261014_8k_low_rank_motion_r2/loss.png), [CSV](results/baselines_20261014_8k_low_rank_motion_r2/loss.csv) |
| [Low rank only, R=8](results/baselines_20261014_8k_low_rank_only_r8/metrics.json) | 61.4310% | 24.9278 | 0.761793 | 0.0031772972 | 67.80 | [PNG](results/baselines_20261014_8k_low_rank_only_r8/loss.png), [CSV](results/baselines_20261014_8k_low_rank_only_r8/loss.csv) |
| [Low rank only, R=16](results/baselines_20261014_8k_low_rank_only_r16/metrics.json) | 65.4166% | 24.3818 | 0.750756 | 0.0035703279 | 68.07 | [PNG](results/baselines_20261014_8k_low_rank_only_r16/loss.png), [CSV](results/baselines_20261014_8k_low_rank_only_r16/loss.csv) |
| [Low rank only, R=24 (selected)](results/baselines_20261014_8k_low_rank_only_r24/metrics.json) | 59.7455% | 25.1695 | 0.772001 | 0.0029911024 | 67.66 | [PNG](results/baselines_20261014_8k_low_rank_only_r24/loss.png), [CSV](results/baselines_20261014_8k_low_rank_only_r24/loss.csv) |

### Branch comparison

Only the two nested models have independently evaluated motion and reference
intensity. Motion errors compare displacement vectors directly on the existing
true foreground mask; the zero-motion endpoint RMSE is 6.0701 px.

| Model | Reference intensity RRMSE | Motion vector RRMSE | Motion endpoint RMSE (px) |
|---|---:|---:|---:|
| Nested hash | 94.2346% | 101.7730% | 6.1777 |
| Low rank + motion, R=2 | 59.0838% | 85.9171% | 5.2153 |

**Selected motion-free baseline: R=24**, by the lowest final fixed validation MSE
among R=8,16,24. See the [selection record](results/baselines_20261014_8k_selection.json)
and [all-run summary](results/summary.csv). All candidate records and local models
are retained. Model weights remain local in each run's `model.pt`; lightweight
records (config, metrics, losses, curve, monitoring indices, source and log) are
published. GIFs were subsequently generated on request (links below); prediction
arrays and intermediate checkpoints remain absent.

Nested hash fits the image best but does not recover the true branches. Low-rank
+ motion reduces both branch errors while retaining substantial decomposition
error. Motion-free low rank fits poorly at this budget, even for the selected
R=24. These runs do not distinguish insufficient capacity from optimization or
training-budget limitations; they do not establish a fundamental limitation of
motion-free low-rank representations. There is no extra rank search or training.

**Post-run verification:** reloaded all five saved models and reproduced their
final fixed training and validation MSE exactly. All five runs use identical
monitoring indices. Checked the 8k final step, complete loss/LR rows, zero final
learning rates, eight-file core output layout, source archive hashes, motion anchors
and absence of fictitious branch metrics for motion-free models. Rank tails were
below 6.1e-8 for the trained low-rank predictions. Recomputed the rank selection
and matched the saved selection record. These checks validate saved results,
not convergence or physical separation.

### GIFs generated on request

Rendered all five saved models with the existing `evaluation.render` command;
no optimization, model/configuration changes or metric recomputation. GIFs remain
opt-in for future runs. Each animation uses all 378 unseen midpoint frames over
three cycles, 90 ms per frame and infinite looping.

| Model | Full image | Reference intensity | Motion effect | Motion only |
|---|---|---|---|---|
| Nested hash | [GIF](results/baselines_20261014_8k_nested_hash/visuals/image.gif) | [GIF](results/baselines_20261014_8k_nested_hash/visuals/reference_intensity.gif) | [GIF](results/baselines_20261014_8k_nested_hash/visuals/motion_effect.gif) | [GIF](results/baselines_20261014_8k_nested_hash/visuals/motion_only.gif) |
| Low rank + motion, R=2 | [GIF](results/baselines_20261014_8k_low_rank_motion_r2/visuals/image.gif) | [GIF](results/baselines_20261014_8k_low_rank_motion_r2/visuals/reference_intensity.gif) | [GIF](results/baselines_20261014_8k_low_rank_motion_r2/visuals/motion_effect.gif) | [GIF](results/baselines_20261014_8k_low_rank_motion_r2/visuals/motion_only.gif) |
| Low rank only, R=8 | [GIF](results/baselines_20261014_8k_low_rank_only_r8/visuals/image.gif) | N/A | N/A | N/A |
| Low rank only, R=16 | [GIF](results/baselines_20261014_8k_low_rank_only_r16/visuals/image.gif) | N/A | N/A | N/A |
| Low rank only, R=24 (selected) | [GIF](results/baselines_20261014_8k_low_rank_only_r24/visuals/image.gif) | N/A | N/A | N/A |

Full image compares truth, prediction and absolute error. Reference intensity is
queried at fixed coordinates, without applying motion. As updated on 2026-10-15,
the linked motion-effect GIFs apply true and predicted motion to the same true
intensity frozen at t0; their right panel is absolute image error. The original
exports used time-varying true intensity. Motion only freezes the learned intensity at the motion reference time:
the left panel is the fixed learned image, the middle applies predicted motion,
and the right shows absolute change, not error against ground truth. Motion-free
models have only the full-image view.

Truth/prediction display ranges are fixed at 0..1. Each GIF's error/change range
is fixed over time but scaled independently and printed above its right panel;
compare the numerical metrics rather than error-panel brightness across GIFs.

**Rendering verification:** all 11 GIFs decode to 378 frames at 864x408, with
90 ms duration and looping. All prediction panels vary over time; both motion-only
left panels remain identical across all frames as intended. Inspected labeled
frames for every GIF. SHA-256 checks confirmed all 42 original result artifacts
(including models, metrics and losses) and 49 unrelated local files are unchanged.
The new GIFs occupy 36.22 MiB and are published with this index.

## Motion maximum resolution 32 baseline (2026-10-15): completed

Approved one additional local CPU run of low-rank + motion R=2, changing only
`motion_encoding.max_resolution` from 128 to scalar 32 and enabling requested GIF
exports. The scalar setting applies to x, y and t; motion resolutions are
`[8, 9, 11, 14, 17, 21, 26, 32]`. Intensity remains eight levels from 8 to 128.
The 8k budget, seed 0, complex 64x64/3-cycle/379-frame benchmark, batch size,
optimizer, learning rates, loss, monitoring samples and evaluation protocol match
the original R=2 baseline. No loss or implementation changes are made.

The unchanged local `nf-crt` runtime is reused. Preflight checked exact setting
and source parity, identical intensity initialization, optimizer coverage,
coordinate gradients, motion reference anchoring and rank bounds. Three tiny
synthetic updates verified gradient flow through both branches. The initial
scratch check incorrectly demanded nonzero motion-table gradients before the
zero-initialized output head updated; the corrected check verifies the expected
initial zero and subsequent nonzero gradients. Model parameters: 224,372 versus
390,934 for max=128. Scratch inputs/checks are in `/tmp/baseline_motion32_20261015`.
The run completed all 8,000 updates using commit `fab901e1e02a71953a9f301393ff6df81408b937`.
Its Python source hashes match the original baseline exactly. The resolved
[configuration](results/baselines_20261015_8k_low_rank_motion_r2_m32/config.json),
[full metrics](results/baselines_20261015_8k_low_rank_motion_r2_m32/metrics.json),
[loss/LR curve](results/baselines_20261015_8k_low_rank_motion_r2_m32/loss.png),
[numeric losses](results/baselines_20261015_8k_low_rank_motion_r2_m32/loss.csv),
monitoring indices, source archive and run log are retained. Weights remain local
in `model.pt`, consistent with the other baselines.

All image/intensity scores below use the complete 378 midpoint frames. Motion
scores compare displacement vectors directly on the existing true foreground mask.

| Metric | Motion max 128 | Motion max 32 |
|---|---:|---:|
| Image RRMSE | 2.5181% | 5.3585% |
| Image PSNR (dB) | 52.6742 | 46.1147 |
| Image SSIM | 0.999630 | 0.998119 |
| Reference intensity RRMSE | 59.0838% | 52.0252% |
| Reference intensity PSNR (dB) | 26.7674 | 27.8725 |
| Reference intensity SSIM | 0.874101 | 0.880784 |
| Motion vector RRMSE | 85.9171% | 80.0716% |
| Motion endpoint RMSE (px) | 5.2153 | 4.8604 |
| Fixed training MSE | 4.0248556e-06 | 2.5598572e-05 |
| Fixed validation MSE | 5.3106635e-06 | 2.6125811e-05 |
| Parameters | 390,934 | 224,372 |
| Training time (s; excludes final evaluation/GIFs) | 277.42 | 264.65 |

At this budget, reducing motion resolution improves both branch errors but
worsens complete-image fit. Substantial branch errors remain. This single-seed
comparison changes spatial and temporal resolution together; it does not establish
an optimal grid or convergence. Monitoring samples and intensity initialization
match exactly, while different motion table sizes alter subsequent random-number
consumption, so equal seeds do not imply matched training batches.

Requested GIFs use the existing definitions and display conventions described above:
[full image](results/baselines_20261015_8k_low_rank_motion_r2_m32/visuals/image.gif),
[reference intensity](results/baselines_20261015_8k_low_rank_motion_r2_m32/visuals/reference_intensity.gif),
[motion effect with true intensity fixed at t0](results/baselines_20261015_8k_low_rank_motion_r2_m32/visuals/motion_effect.gif),
and [motion only with frozen learned intensity](results/baselines_20261015_8k_low_rank_motion_r2_m32/visuals/motion_only.gif).
Motion-only still uses the learned intensity at t0. The motion-effect link was
subsequently regenerated with frozen true intensity in the visualization update
below; that change was not part of this configuration-only training run.
Default GIF export remains disabled in the reusable template.

**Post-run verification:** the saved model reproduces final fixed training and
validation MSE exactly; monitoring indices match the max=128 run. Checked all
18 loss rows, finite values, cosine rates ending at zero, source archive hashes,
the zero-motion reference anchor and rank-2 tails below 3.7e-8. All four GIFs
decode to 378 frames at 864x408 and 90 ms per frame; inspected the labels and
sampled frames, and verified the frozen left panel in motion-only. The loss curve
was visually checked. SHA-256 checks preserve all 53 old result artifacts and
49 unrelated local files; all existing summary rows are unchanged.

## Motion max 32: 24k-step follow-up (2026-10-15): completed

Approved one additional run with the exact R=2, motion max=32 setup above,
changing only `steps` from 8,000 to 24,000. Training starts from the same seeded
initialization; the existing cosine scheduler now reaches zero at step 24,000.
This is a new run with a longer schedule, not a continuation of the finished 8k
checkpoint. All data, model, optimizer, loss, monitoring and evaluation settings
remain identical, including the four requested GIF views. The reusable template
is unchanged.

Preflight confirmed the input differs only in `steps`, all numerical source
hashes and the runtime match, and the complete initial model state is identical
to a fresh model under the 8k configuration. The model still has 224,372 parameters.
Scratch inputs and checks are in `/tmp/baseline_motion32_24k_20261015`.
Completed all 24,000 updates using commit `5774f44cfcbaf739fbffd9a2dccd0c582c699cd5`.
The saved [configuration](results/baselines_20261015_24k_low_rank_motion_r2_m32/config.json),
[full metrics](results/baselines_20261015_24k_low_rank_motion_r2_m32/metrics.json),
[loss/LR curve](results/baselines_20261015_24k_low_rank_motion_r2_m32/loss.png),
[numeric losses](results/baselines_20261015_24k_low_rank_motion_r2_m32/loss.csv),
monitoring indices, source archive and log are retained. The final `model.pt`
remains local, as for the other baselines.

### 8k versus 24k at motion max 32

Full-image and reference-intensity scores use all 378 midpoint frames; motion
scores compare displacement vectors directly on the same true foreground mask.

| Metric | 8k steps | 24k steps |
|---|---:|---:|
| Image RRMSE | 5.3585% | 3.7109% |
| Image PSNR (dB) | 46.1147 | 49.3060 |
| Image SSIM | 0.998119 | 0.999122 |
| Reference intensity RRMSE | 52.0252% | 59.3448% |
| Reference intensity PSNR (dB) | 27.8725 | 26.7291 |
| Reference intensity SSIM | 0.880784 | 0.855292 |
| Motion vector RRMSE | 80.0716% | 83.5368% |
| Motion endpoint RMSE (px) | 4.8604 | 5.0708 |
| Fixed training MSE | 2.5598572e-05 | 1.0711806e-05 |
| Fixed validation MSE | 2.6125811e-05 | 1.2019928e-05 |
| Training time (s; excludes final evaluation/GIFs) | 264.65 | 658.16 |

The [loss comparison](results/baselines_20261015_24k_low_rank_motion_r2_m32/visuals/loss_comparison.png)
plots both numeric histories without smoothing. Longer training with the longer
cosine schedule reduces fixed validation MSE by about 54% and improves image
RRMSE/PSNR/SSIM, but both branch errors become worse. In this comparison, better
image fitting does not produce better motion/intensity separation. The late loss
curve flattens as the learning rate approaches zero; this does not prove full
convergence. These are two single-seed runs with different schedule horizons,
not an isolated test of extra updates at an unchanged learning-rate schedule.

All four requested animations are saved:
[full image](results/baselines_20261015_24k_low_rank_motion_r2_m32/visuals/image.gif),
[reference intensity](results/baselines_20261015_24k_low_rank_motion_r2_m32/visuals/reference_intensity.gif),
[motion effect with true intensity fixed at t0](results/baselines_20261015_24k_low_rank_motion_r2_m32/visuals/motion_effect.gif),
and [motion only with frozen learned intensity](results/baselines_20261015_24k_low_rank_motion_r2_m32/visuals/motion_only.gif).
At training completion the GIF definitions, clipping and individual fixed error
scales were unchanged. The linked motion-effect GIF was subsequently regenerated
with frozen true intensity in the visualization update below.

**Post-run verification:** reloading the final model reproduces fixed training
and validation MSE exactly. Monitoring indices match the 8k run. Checked the
24k final step, all 50 finite loss/LR rows, the cosine schedule ending at zero,
source archive hashes, zero reference motion and rank-2 tails below 3.6e-8.
Initial monitoring and the first pre-update batch loss match exactly. An extra
bitwise check of first post-update monitoring failed on absolute differences of
3.59e-13 (training) and 4.29e-12 (validation); the original values were preserved
and these checks pass at 1e-8 relative tolerance. This check did not alter any
training, metrics or saved loss values; its initial failure log is retained in
the scratch folder. All four GIFs decode to 378 frames at 864x408 and 90 ms, and
motion-only has the intended fixed left panel. Sampled GIF frames and both loss
plots were visually checked. All 65 old result artifacts, 49 unrelated local
files and prior summary rows are unchanged.

## Standard motion GIF: true intensity frozen at t0 (2026-10-15)

The user approved replacing the motion-effect visualization with a shared true
reference frozen at the model's motion reference time (t0=-0.5 for these runs):
left `I_true(r + u_true(r,t), t0)`, middle
`I_true(r + phi_motion(r,t), t0)`, right the absolute image difference.
Both motion fields retain their current observation time. Neither panel uses
the learned intensity network. The true reference is evaluated continuously at
the displaced coordinates, rather than warped from a rendered reference bitmap.

`evaluation/motion_visuals.py` implements the shared definition for the standard
renderer, optional training exports, branch diagnostics and multi-run GIFs.
This replaces the former time-varying true-intensity motion view. The separately
named `motion_only.gif` retains its learned-reference diagnostic definition.
The template still disables automatic GIF generation. Numerical evaluation,
including direct motion errors and time-varying branch-replacement metrics,
is unchanged; no model was trained or metric record recomputed.

Regenerated all four existing motion-effect GIFs from the saved models:

| Model | Updated motion GIF |
|---|---|
| Low rank + motion R=2, motion max 32, 24k | [GIF](results/baselines_20261015_24k_low_rank_motion_r2_m32/visuals/motion_effect.gif) |
| Low rank + motion R=2, motion max 32, 8k | [GIF](results/baselines_20261015_8k_low_rank_motion_r2_m32/visuals/motion_effect.gif) |
| Low rank + motion R=2, motion max 128, 8k | [GIF](results/baselines_20261014_8k_low_rank_motion_r2/visuals/motion_effect.gif) |
| Nested hash, 8k | [GIF](results/baselines_20261014_8k_nested_hash/visuals/motion_effect.gif) |

**Validation:** 91 tests passed. Controls verify zero image error when predicted
motion equals true motion, a time-constant image for zero predicted motion,
custom t0 handling and unchanged inputs. Checked every panel of all 378 frames
in each regenerated GIF against the frozen-reference formula: decoded grayscale
values match exactly. All four left-panel sequences are identical and move over
time. Dimensions (864x408), 90 ms duration, looping, fixed 0..1 image display and
per-GIF fixed error scales are preserved; labeled frames were visually inspected.
Source syntax comparisons confirm that midpoint/block numerical modules changed
only at the GIF export call. Rendering old checkpoints verifies unchanged model,
phantom and data-source hashes; the revised evaluation export call is accepted
only after comparing all remaining module syntax with the hash-verified source
archive. Changed inference functions still fail the provenance check.

Checksums preserve all 75 other result files (including models, configurations,
metrics, losses, source archives and other GIFs) and 37 unrelated working files.
Validation scripts/logs are in `/tmp/frozen_true_motion_20261015`. The updated
images visualize existing motion errors; this is not a new experiment or evidence
of improved model performance.

## Oracle-branch diagnostics A1/A2 (2026-10-15): completed

Approved two 8k runs that each replace one branch of the R=2, motion-max-32 baseline
with the complex phantom's true field and train only the other branch from image MSE:
**A1** uses true backward motion `u = (r - T(t))/S(t) - r` and learns the low-rank
intensity; **A2** uses the true reference intensity `C(q,t)` (exactly rank 2) and
learns the motion hash field. These runs use ground-truth branch labels and are
diagnostics, not reconstruction methods. All other settings match the baseline.

`models/oracle_branches.py` holds torch copies of the phantom formulas (A2 needs
coordinate gradients through `C`); `phantoms/complex.py` is unchanged. The option
`oracle_branch` is accepted only for `nested_low_rank_hash_nf`, complex, midpoints
and t0=-0.5; runs set `true_motion_used_for_training` or
`reference_intensity_labels_used_for_training` and a new `oracle_branch` summary
column. Both branches are built before replacement, so the learned branch's
initialization and every training batch equal the baseline's.

**Preflight:** 98 tests passed (7 new: torch/NumPy formulas agree to 1e-12, anchors,
both oracles reproduce observed images, initialization/RNG parity, optimizer scope,
gradient flow through the true intensity, CLI rejection, complete run and reload).
At full size, dry-run inputs differ from the baseline only in `oracle_branch`; RNG
state, learned-branch initialization and the first two batches matched exactly, and
the unchanged baseline path reproduced its saved step-1 batch MSE exactly.
Implementation commit `703aa29`; both runs used it. Scratch inputs and checks are in
the session scratch folder `oracle_20261015`.

| Metric (378 midpoints) | Baseline (joint) | A1: true motion | A2: true intensity |
|---|---:|---:|---:|
| Image RRMSE | 5.3585% | 17.7370% | 10.3651% |
| Image PSNR (dB) / SSIM | 46.1147 / 0.998119 | 35.7180 / 0.991679 | 40.3841 / 0.996908 |
| Reference intensity RRMSE | 52.0252% | 19.0426% | (true) |
| Reference intensity center / spot curve RRMSE | 79.45% / 22.37% | 17.65% / 21.76% | (true) |
| Motion vector RRMSE | 80.0716% | (true) | 64.6749% |
| Motion endpoint RMSE / mean (px) | 4.8604 / 4.1877 | (true) | 3.9258 / 3.0206 |
| Foreground nonpositive-Jacobian fraction | 31.39% | 0 | 15.82% |
| Fixed training / validation MSE | 2.5599e-05 / 2.6126e-05 | 2.7660e-04 / 2.5991e-04 | 9.6457e-05 / 9.0028e-05 |
| Trainable parameters / training seconds | 224,372 / 264.65 | 67,468 / 47.87 | 156,904 / 134.85 |

Records: A1 [config](results/oracle_20261015_8k_true_motion_r2/config.json),
[metrics](results/oracle_20261015_8k_true_motion_r2/metrics.json),
[loss curve](results/oracle_20261015_8k_true_motion_r2/loss.png),
[losses](results/oracle_20261015_8k_true_motion_r2/loss.csv),
GIFs: [image](results/oracle_20261015_8k_true_motion_r2/visuals/image.gif),
[reference intensity](results/oracle_20261015_8k_true_motion_r2/visuals/reference_intensity.gif).
A2 [config](results/oracle_20261015_8k_true_intensity_m32/config.json),
[metrics](results/oracle_20261015_8k_true_intensity_m32/metrics.json),
[loss curve](results/oracle_20261015_8k_true_intensity_m32/loss.png),
[losses](results/oracle_20261015_8k_true_intensity_m32/loss.csv),
GIFs: [image](results/oracle_20261015_8k_true_intensity_m32/visuals/image.gif),
[motion effect](results/oracle_20261015_8k_true_intensity_m32/visuals/motion_effect.gif).
Weights remain local in `model.pt`.

**A1, intensity branch:** even with exact motion, the image fit is worse than the
joint baseline, and the learned intensity is nearly static. Its reference-space
singular values are 94.81 and 0.54, versus 96.46 and 5.60 for the truth. At the
body center, learned intensity has standard deviation 0.0063 over time versus
0.0495 for truth. In a least-squares fit, span(a_1, a_2, 1) of the learned temporal
coefficients explains R^2=0.021 of g(t) and 0.063 of g(t)s(t) (baseline: 0.017 and
0.053). The reference-intensity GIF also shows vertical streaks in the body. At
this architecture and budget, the temporal coefficients do not learn the true
brightness waveforms, which oscillate 9 and up to 15 times over [-0.5, 2.5]. The
joint baseline's better image fit therefore relies on incorrect, folded motion.
These runs do not distinguish temporal-MLP capacity from optimization (learning
rate 0.001, 8k steps).

*Rank versus time (analysis of the saved A1 model, no training):* R=2 is not the
limit, because the truth is exactly rank 2 and even its best rank-1 approximation
has 5.80% RRMSE. A1's 19.04% reference-intensity error comes from the temporal
coefficients. Keeping the learned a(t) and fitting optimal spatial bases by least
squares only lowers it to 18.36%; keeping the learned b(q) and fitting optimal
time coefficients lowers it to 5.60%. Of the learned coefficients' variance,
78-82% lies below 1.5 cycles per period, whereas 99-100% of the true g(t) and
g(t)s(t) variance lies above it. This is consistent with a low-frequency bias of
the temporal MLP; capacity versus optimization remains untested.

**A2, motion branch:** even with exact intensity, the learned motion recovers only
part of the displacement and still folds 15.8% of the foreground. Per-frame
foreground endpoint RMSE correlates with true displacement magnitude (r=0.60);
frames whose true foreground RMS displacement exceeds 6 px average 4.19 px error
(234 frames), versus 2.16 px below 2 px (30 frames). The median ratio of error to
true displacement is 0.70. Flat-interior motion is unobservable, so part of the
endpoint error is unidentifiable; the 10.37% image error with true intensity shows
that edge correspondence is also wrong, mainly around spot1 and the body edge.
This is consistent with a limited capture range for large displacements of sharp
edges, but that cause has not been isolated.

The waveform R^2, center standard deviation and per-frame correlations above are
ad hoc analyses of the saved models and metrics, computed outside the pipeline;
they are not adopted metrics. Single seed, single configuration.

**Post-run verification:** reloading both models reproduces final fixed training
and validation MSE exactly; monitoring indices equal the baseline's. Checked all
18 finite loss/LR rows, cosine rates ending at zero, the optimizer groups (A1:
intensity hash/MLP/time MLP; A2: motion hash/MLP), zero motion at t0, the A1 rank-2
tail (3.9e-8), source archive hashes, and sampled GIF frames.

**Provenance note:** `run.py` changed, so `evaluation.render` now refuses earlier
checkpoints by design. Re-render them from their `source.zip` or the original
commit. For this analysis, the baseline was loaded with `build_model` plus its
`state_dict`; this reproduced its saved fixed validation MSE exactly.

## Intensity fitting with true motion: time-network options (2026-10-15): completed

The user asked to keep the architecture (R=2, hash spatial bases x time network)
and to make the intensity fit with true motion, by widening the time MLP, using a
1D time grid, or tuning. All runs use the A1 setting above (true motion, learned
R=2 intensity, 8k steps, seed 0, identical data, batches and monitoring indices);
only the time network or its learning rate changes. These are oracle diagnostics.

Implementation commit `d0733e2` adds two optional keys; omitting both leaves models,
parameter groups and checkpoints unchanged:
- `temporal_encoding`: dense 1D multi-resolution grid with linear interpolation,
  then a ReLU MLP (width `temporal_width`) to R coefficients. Its table uses `hash_lr`.
- `temporal_mlp_lr`: the time MLP's own rate, defaulting to `mlp_lr`.

E4 uses six levels with 3, 6, 12, 24, 48 and 96 cells over the three-cycle span
(1 to 32 cells per cycle; 126 training intervals per cycle). It has no periodicity
assumption. **Validation:** 105 tests passed (7 new: interpolation against
`np.interp`, level rule shared with the hash grid, locality, rank bound,
optimizer groups and rates, gradient flow, CLI rejection, complete run and reload).
Replaying the saved A1 and baseline configurations reproduced their step-0 fixed
MSE and step-1 batch MSE bitwise. Post-update fixed MSE differed by about 1e-10
relative and varied between identical replays, consistent with the CPU
nondeterminism noted in the 24k follow-up.

| Run (true motion, R=2, 8k) | Time network | Image RRMSE | Ref. intensity RRMSE | Center / spot curve | Singular values (truth 96.46 / 5.60) | Fixed val. MSE | Params / s |
|---|---|---:|---:|---:|---|---:|---:|
| A1 | Tanh MLP w32, lr 1e-3 | 17.74% | 19.04% | 17.65% / 21.76% | 94.81 / 0.54 | 2.599e-04 | 67,468 / 48 |
| [E1](results/intensity_fit_20261015_8k_e1_mlp_w32_tlr1e-2/metrics.json) | Tanh MLP w32, time lr 1e-2 | 14.51% | 15.51% | 13.75% / 18.62% | 94.88 / 2.84 | 1.682e-04 | 67,468 / 50 |
| [E2](results/intensity_fit_20261015_8k_e2_mlp_w128/metrics.json) | Tanh MLP w128, lr 1e-3 | 17.13% | 18.43% | 16.97% / 20.99% | 94.46 / 0.99 | 2.467e-04 | 83,308 / 81 |
| [E3](results/intensity_fit_20261015_8k_e3_mlp_w128_tlr1e-2/metrics.json) | Tanh MLP w128, time lr 1e-2 | 7.41% | 7.49% | 7.51% / 6.74% | 96.20 / 4.77 | 4.706e-05 | 83,308 / 95 |
| [E4](results/intensity_fit_20261015_8k_e4_grid_c3-96/metrics.json) | 1D grid 3-96 + ReLU MLP w32 | **0.42%** | **0.35%** | **0.14% / 0.25%** | 96.51 / 5.69 | **1.532e-07** | 68,210 / 83 |

Each run has its config, loss CSV/curve, monitoring indices, source and log beside
the linked metrics. E4 GIFs:
[image](results/intensity_fit_20261015_8k_e4_grid_c3-96/visuals/image.gif) and
[reference intensity](results/intensity_fit_20261015_8k_e4_grid_c3-96/visuals/reference_intensity.gif).
E4 reference-intensity PSNR/SSIM are 71.23 dB / 0.999995, with fixed training
MSE 1.125e-07 versus validation 1.532e-07 on unseen midpoints.

**Findings:**
- The 1D grid meets the proposed success criteria (reference intensity < 5%,
  point curves < 10%): both temporal waveforms are recovered. In the ad hoc
  span(a_1, a_2, 1) fit, R^2 is 1.000 for both g(t) and g(t)s(t), versus 0.021 and
  0.064 in A1. The A1 streak artifacts are absent; a sampled frame shows maximum
  error 0.016.
- For the Tanh MLP, width alone does nothing (E2) and a higher time rate alone
  helps a little (E1). Together they reach 7.49% (E3), but a sampled waveform plot
  shows E3 missing about one oscillation near t=1.0-1.4. This supports an
  optimization/low-frequency limit of the Tanh MLP rather than rank; whether E3
  converges with longer training was not tested.
- The time grid did not need a separate learning rate; its table used the
  existing hash rate.

**Post-run verification:** all four reloads reproduce final fixed training and
validation MSE exactly; monitoring indices equal the baseline's; 18 finite loss/LR
rows each with the expected optimizer groups and zero final rates; zero motion at
t0; rank-2 tails at most 4.0e-8; source archive hashes match commit `d0733e2`.
The E4 GIFs were rendered afterwards with `evaluation.render` (no training or
metric change), and sampled frames were inspected.

**Not established:** single seed, one grid setting, toy phantom only. The grid
has not been tested with learned motion, where a more flexible time network may
also make crossfade-like explanations of motion easier. The grid resolution is a
bandwidth assumption that would need a physical choice for real PACT data.

## Motion fitting with true intensity: per-axis motion hash resolution (2026-10-15): completed

Approved: tune the existing motion network in the A2 setting (true reference
intensity, learned motion, 8k steps, seed 0; identical data, monitoring indices and
evaluation). The motion network stays one 3D (x,y,t) hash field plus MLP, anchored at
t0; only the existing per-axis `motion_encoding` resolutions change. No code change;
runs used commit `bb82ad0`. Separable motion was explicitly excluded by the user.
All runs use eight levels, two features and 2^15 table entries per level.

| Run (true intensity) | Motion cells (x,y,t) min -> max | Image RRMSE | Foreground EPE RMSE / mean / max (px) | Vector RRMSE | Folding | Fixed val. MSE | Params / s |
|---|---|---:|---:|---:|---:|---:|---:|
| A2 | 8 -> 32 (all axes) | 10.37% | 3.926 / 3.021 / 12.83 | 64.67% | 15.82% | 9.003e-05 | 156,904 / 135 |
| [M1a](results/motion_fit_20261015_8k_m1a_s8-32_t8-96/metrics.json) | [8,8,8] -> [32,32,96] | 4.37% | 3.645 / 2.894 / 10.55 | 60.05% | 13.30% | 1.712e-05 | 224,500 / 154 |
| [M1b](results/motion_fit_20261015_8k_m1b_s2-8_t8-32/metrics.json) | [2,2,8] -> [8,8,32] | 0.64% | 0.156 / 0.107 / 0.87 | 2.57% | 0 | 3.336e-07 | 16,892 / 176 |
| [M1c](results/motion_fit_20261015_8k_m1c_s2-8_t8-96/metrics.json) | [2,2,8] -> [8,8,96] | **0.42%** | **0.129 / 0.087 / 0.69** | **2.12%** | 0 | **1.525e-07** | 34,364 / 196 |
| [M1d](results/motion_fit_20261015_8k_m1d_s2-16_t8-96/metrics.json) | [2,2,8] -> [16,16,96] | 0.48% | 0.257 / 0.183 / 1.34 | 4.24% | 0 | 1.947e-07 | 94,258 / 181 |

The zero-motion foreground EPE RMSE is 6.070 px. In M1a, the two finest levels
exceed the table and are hashed; all M1b-M1d levels are dense. Each run has its
config, loss CSV/curve, monitoring indices, source and log beside the linked metrics.
M1c GIFs: [image](results/motion_fit_20261015_8k_m1c_s2-8_t8-96/visuals/image.gif) and
[motion effect](results/motion_fit_20261015_8k_m1c_s2-8_t8-96/visuals/motion_effect.gif).

**Findings:**
- Spatial motion resolution was the limiting factor. Coarsening space to 2-8 cells
  recovers motion (M1b-M1d meet the proposed criteria: EPE < 1 px, folding < 1%,
  image < 2%). Refining only time while keeping space at 8-32 (M1a) leaves 3.6 px
  error and 13% folding. A finer time axis gives a smaller further gain (M1b to M1c).
- Ad hoc split (computed outside the pipeline, not an adopted metric): on
  foreground pixels where the true frame has spatial gradient (63% of the
  foreground) versus flat interior pixels, endpoint RMSE is 3.856 / 4.042 px for A2
  and 0.147 / 0.089 px for M1c. A2 was also wrong at edges, and the coarse grid
  interpolates the unobservable flat interior correctly for this globally affine
  motion.
- Whole-FOV EPE remains 2.650 px for M1c (zero motion: 6.722 px). Background motion
  has no image signal there and is not constrained by the data.
- A sampled M1c motion-effect frame matches the true panel; its fixed error scale
  is 0-0.018, versus 0-0.38 for A2.

**Post-run verification:** all four reloads reproduce final fixed training and
validation MSE exactly; monitoring indices equal the baseline's; 18 finite loss/LR
rows each with motion hash/MLP groups and zero final rates; zero motion at t0; source
archive hashes match `bb82ad0`. M1c GIFs were rendered afterwards with
`evaluation.render`. The optional learning-rate variants were not run, because the
criteria were met.

**Not established:** single seed, toy phantom with globally affine motion. A
2-8 spatial grid is a strong smoothness prior that may be too coarse for real
local deformations. True intensity was supplied, and joint training with both
fixes has not been run.

## Joint training with both fixes (2026-10-15): completed

Approved: one joint run (no oracle) combining the E4 time grid and the M1c motion
resolution. Everything else matches the R=2, motion-max-32, 8k joint baseline
(complex 64x64, 3 cycles, seed 0, same optimizer, image-MSE loss and evaluation).
No code change; commit `40b3875`. Records:
[config](results/joint_20261015_8k_r2_timegrid_motion_s2-8_t8-96/config.json),
[metrics](results/joint_20261015_8k_r2_timegrid_motion_s2-8_t8-96/metrics.json),
[loss curve](results/joint_20261015_8k_r2_timegrid_motion_s2-8_t8-96/loss.png),
[losses](results/joint_20261015_8k_r2_timegrid_motion_s2-8_t8-96/loss.csv); GIFs:
[image](results/joint_20261015_8k_r2_timegrid_motion_s2-8_t8-96/visuals/image.gif),
[reference intensity](results/joint_20261015_8k_r2_timegrid_motion_s2-8_t8-96/visuals/reference_intensity.gif),
[motion effect](results/joint_20261015_8k_r2_timegrid_motion_s2-8_t8-96/visuals/motion_effect.gif),
[motion only](results/joint_20261015_8k_r2_timegrid_motion_s2-8_t8-96/visuals/motion_only.gif).

| Metric (378 midpoints) | Joint baseline (motion 8-32, Tanh MLP) | Joint, time grid + motion [2,2,8]->[8,8,96] |
|---|---:|---:|
| Image RRMSE / PSNR / SSIM | 5.3585% / 46.11 / 0.998119 | 2.3155% / 53.40 / 0.999792 |
| Reference intensity RRMSE / PSNR / SSIM | 52.03% / 27.87 / 0.880784 | 91.01% / 23.01 / 0.789538 |
| Reference center / spot curve RRMSE | 79.45% / 22.37% | 10.45% / 1.93% |
| Motion foreground EPE RMSE / mean / max (px) | 4.860 / 4.188 / 15.21 | 3.190 / 2.960 / 5.90 |
| Motion vector RRMSE / folding | 80.07% / 31.39% | 52.55% / 0 |
| Learned singular values (truth 96.46 / 5.60) | 114.70 / 8.25 | 140.85 / 7.43 |
| Fixed training / validation MSE | 2.560e-05 / 2.613e-05 | 2.611e-06 / 4.152e-06 |
| Parameters / training seconds | 224,372 / 265 | 102,574 / 232 |

**Reference-frame (gauge) finding (ad hoc analysis of the saved model; not adopted
metrics):**
- The learned motion is nearly affine at every time, with a per-frame affine fit
  residual of 0.14-0.50 px, and periodic. Relative to the truth it differs by one
  fixed map: q_hat ~ W q_true + c, with W = [[1.382, -0.039], [-0.037, 1.516]] and
  c = (0.79, -0.69) px. That is close to the phantom's maximum expansion (1.30, 1.45).
- The anchor u(r,t0)=0 holds only at the instant t0. Within about 0.01 time units
  after t0 (about one frame interval), the learned affine scale jumps from (1.00,
  1.00) to about (1.36, 1.58), and the temporal coefficients jump with it.
- Training-frame image RRMSE is 22.83% at t0 and 8.00%, 4.89%, 2.90% for the next
  three frames, versus a 1.33% median.
- The learned reference intensity is the same object enlarged by about 1.4x / 1.5x
  (reference-intensity GIF), which explains the 91% raw reference-intensity error.
- After removing the fitted fixed map W (fit with the true motion, t > -0.48),
  motion error drops from 3.19 px to 0.52 px RMSE (max 2.63 px), and
  reference-intensity RRMSE drops from 91.0% to 18.7%.
- Interpretation: image MSE, rank 2 and a smooth motion field are unchanged by
  any fixed reparameterization of the reference frame. A single-instant anchor
  fixes that frame only at t0, so a flexible time representation can switch
  frames immediately after t0 at a small image cost. The old baseline did not show
  this behaviour (its motion slope relative to truth was 0.57, with large affine
  residuals), but its branches were wrong for other reasons.

**Post-run verification:** reloading the model reproduces final fixed training and
validation MSE exactly; monitoring indices equal the baseline's; 18 finite loss/LR
rows with all six optimizer groups and zero final rates; zero motion at t0; rank-2
tail 3.4e-8; source archive hashes match `40b3875`. All four GIFs decode to 378
frames; a sampled reference-intensity frame was inspected.

**Open decision (not implemented):** how to pin the reference frame (for example
limiting how fast motion may change after t0, or a different anchor definition),
or whether to evaluate up to a fixed reference-frame map. These change the model
meaning or the evaluation protocol and require the user's choice. Single seed.

## Joint tuning against the reference-frame jump (2026-10-15): completed

The user asked to keep tuning parameters (config only: anchor definition, loss and
evaluation unchanged). Every run starts from the joint configuration above (R=2,
time grid, motion space 2-8) and changes only the motion time resolution, the
time-grid maximum resolution, `hash_lr` or the step budget. Seed 0, no GIFs during
tuning, commit `a416cd6`; at most six runs were planned and five were used.
Proposed success criteria on raw metrics: reference intensity < 10%, motion EPE <
1 px, folding < 1%, image < 2%. `W` and the aligned EPE are ad hoc diagnostics
(a fixed affine map fit with the true motion for t > -0.48), not adopted metrics.

| Run | Motion t cells | Time grid max | Other | Image | Ref. intensity | Center / spot | Motion EPE RMSE (px) | Folding | W_xx / W_yy | Aligned EPE (px) | First-frame image error |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| Joint above | 8-96 | 96 | 8k | 2.32% | 91.01% | 10.5% / 1.9% | 3.190 | 0 | 1.382 / 1.516 | 0.520 | 22.83% |
| [J1](results/joint_tuning_20261015_8k_j1_mt8-32_grid96/metrics.json) | 8-32 | 96 | 8k | 1.51% | 40.61% | 1.2% / 0.8% | 1.076 | 0 | 1.115 / 1.169 | 0.391 | 26.51% |
| [J2](results/joint_tuning_20261015_8k_j2_mt8-32_grid48/metrics.json) | 8-32 | 48 | 8k | 1.62% | 45.64% | 1.2% / 0.9% | 1.202 | 0 | 1.139 / 1.190 | 0.424 | 32.43% |
| [J3](results/joint_tuning_20261015_8k_j3_mt4-16_grid48/metrics.json) | 4-16 | 48 | 8k | 1.63% | 22.51% | 0.8% / 0.7% | 0.636 | 0 | 1.040 / 1.083 | 0.387 | 19.60% |
| [J4](results/joint_tuning_20261015_8k_j4_mt4-16_grid48_hashlr3e-3/metrics.json) | 4-16 | 48 | hash lr 0.003 | 2.18% | 25.36% | 0.9% / 0.8% | 0.772 | 0 | 1.054 / 1.100 | 0.464 | 21.95% |
| [J5](results/joint_tuning_20261015_24k_j5_mt4-16_grid48/metrics.json) | 4-16 | 48 | **24k steps** | **0.92%** | **9.61%** | **0.4% / 0.4%** | **0.336** | 0 | **1.017 / 1.023** | 0.247 | 8.77% |

All tuning runs use motion space 2-8 cells; the time grid has six levels from 3
cells. J5 full metrics: image PSNR/SSIM 61.42 dB / 0.999939; reference intensity
42.54 dB / 0.994251; motion mean/max EPE 0.291 / 2.04 px and vector RRMSE 5.53%;
fixed training/validation MSE 7.412e-07 / 7.272e-07; 79,336 parameters; 876 s.
For the temporal coefficients, the ad hoc span R^2 is 1.000 for both waveforms.
J5 GIFs: [image](results/joint_tuning_20261015_24k_j5_mt4-16_grid48/visuals/image.gif),
[reference intensity](results/joint_tuning_20261015_24k_j5_mt4-16_grid48/visuals/reference_intensity.gif),
[motion effect](results/joint_tuning_20261015_24k_j5_mt4-16_grid48/visuals/motion_effect.gif),
[motion only](results/joint_tuning_20261015_24k_j5_mt4-16_grid48/visuals/motion_only.gif).

**Findings:**
- The motion's time resolution controls the reference-frame shift. Going from
  8-96 to 8-32 to 4-16 cells moves W from 1.38/1.52 to 1.12/1.17 to 1.04/1.08.
  Coarser motion time cells make an immediate frame switch after t0 more costly.
- Halving the time-grid maximum (96 to 48) changed little. A lower shared hash
  rate (J4) was slightly worse than J3.
- With the J3 settings, the 24k schedule (J5) reaches every proposed criterion on
  raw metrics, with W = 1.017/1.023. The reference-intensity GIF shows the
  correct object size, with a thin edge-ring error (maximum about 0.14 in the
  sampled frame).
- A smaller residual jump remains: first-frame image error is 8.77% versus a
  0.92% midpoint average. Motion t 4-16 means about 5.3 cells per cycle for motion
  whose fastest component (y translation) has 2 oscillations per cycle. This is a
  bandwidth assumption that suits this phantom but needs a physical basis for
  real data.

**Post-run verification:** all five reloads reproduce final fixed training and
validation MSE exactly; monitoring indices equal the baseline's; finite loss/LR
rows (18, or 50 for J5) with all six optimizer groups and zero final rates; zero
motion at t0; rank-2 tails at most 3.6e-8; source archive hashes match
`a416cd6`. J5 GIFs (378 frames) were rendered afterwards with `evaluation.render`.

**Not established:** single seed. J5 differs from J3 in step budget and cosine
horizon, not only extra updates. Reference intensity (9.61%) is only just under
the proposed threshold. The single-instant anchor is still weak in principle;
tuning made the frame switch costly rather than impossible.

## DCE-MOBY z-slice extraction (2026-10-20/21): completed for z = 100

Goal: test the toy-validated model on the DCE-MOBY p0 data. The full set is too large
(269 GB of p0 on Vista at `/work/03453/villa13/vista/proj/moby-data`), so the plan is
to extract one z plane of every respiratory phase on Vista and copy only that back.

Verified source facts (`p0_00000/00001`, `indicator_00000/00001`, copied to `~/Data/dce-moby/`):
- `p0` is (x, y, z, frame) = (248, 248, 200, 73 or 72) float32, 0.15 mm voxels,
  fov x,y 0-37.2 mm and z 26.25-56.25 mm, wavelength 800. Frame count is 73 for
  phase 0 and 72 for phase 1 (the README says 72).
- Frames are 10 s apart within a phase; phase p is offset by 0.05 p s
  (indicator attribute `dt` = 0.05, 200 phases), so the interlaced series has 0.05 s
  steps. Time 0 is injection; the series starts at -36 s.
- Indicator files hold `artery`, `liver`, `tumor`, `tumor_core` partial volumes on
  the same (248, 248, 200) grid with the same geometry attributes.

[phantoms/moby_slice.py](phantoms/moby_slice.py) (numpy + h5py only, runs standalone)
reads one z plane from every selected phase, interlaces all frames by `time`, and
writes `p0` (x, y, frame) with `time`, `resp_phase`, `frame_in_phase`, the indicator
planes (x, y, phase), source geometry, `z_index` and the code commit. Axis order is
unchanged; nothing is resampled. It rejects inconsistent geometry and duplicate times.

Validated: [tests/test_moby_slice.py](tests/test_moby_slice.py) (synthetic files, 3
tests) passes. On the two local phases at z = 100 (test index only): 145 frames,
every frame equals the source plane, indicator planes match, 2.3 s and 0.18 GB peak
memory. Estimated full run: about 14,500 frames, 3.6 GB in memory.

**Full extraction (2026-10-21):** user chose the z midpoint, `z_index = 100`
(200 planes; 100 = 200 // 2, one of the two central planes). Ran as Slurm job 1087728
on Vista (`gg` queue, 1 node, allocation ASC25089) in 55 min 46 s, about 16 s per
phase; a first 30-min `idev` attempt timed out at phase 84 and wrote nothing. Job
script, logs and output are in `$WORK/moby_slice/` on Vista (`/work/11803/zc7665/vista`);
the script there was copied from commit `948d7b9`, so the file's `code_commit` reads
`unknown`. Python on Vista: `module load gcc python3` plus a venv at `$WORK/envs/moby`
(h5py, numpy), following the Vista user guide.

Output `~/Data/dce-moby/moby_z100.h5` (1.46 GB, lzf): `p0` (248, 248, 14401), times
-36 to 684 s, strictly increasing with 0.05 s steps; phases cycle 0..199; phase 0 has
73 frames, the other 199 phases 72. Values finite, range 0 to 0.450. Indicator planes
(248, 248, 200) for all four tissues.

Verified: every frame of phases 0 and 1 and their indicator planes equal the local
source files; frames (57, 30), (150, 0), (199, 71) have the same SHA-1 and time as the
source files on Vista. Other phases were not compared frame by frame.

Open decisions (user): whether the out-of-plane respiratory motion of a fixed z slice
is acceptable for the 2D model, and the observation geometry and evaluation for this
data.

## DCE-MOBY z = 100 GIFs (2026-10-22): completed

[evaluation/moby_gifs.py](evaluation/moby_gifs.py) writes two GIFs from the slice file
so each effect is seen on its own: one respiratory cycle of consecutive frames (cycle 0,
-36 to -26 s, before injection, so almost pure motion) and every frame of one phase
(phase 0, 73 frames, -36 to 684 s, so only the contrast changes). Both use one fixed
display range, x to the right and y upward like the toy GIFs, 90 ms per frame, a still
PNG and a `gifs.json` with source hash, frame range and code commit.
[tests/test_moby_gifs.py](tests/test_moby_gifs.py) (frame selection, orientation) passes.

With the default range 0..0.450 (global maximum) the GIFs are 0.94 MB and 0.32 MB but
almost black: the maximum comes from a few vessel voxels (99th / 99.9th percentile of
nonzero values 0.029 / 0.108; tumor mean 0.008 to 0.048 over time). `--display-max`
sets another fixed range. The user chose 0..0.05 (a display normalization only; data
and metrics are unchanged), which clips 0.31% of nonzero pixels over all frames.
Published: [results/dce_moby_z100/](results/dce_moby_z100/) `respiratory_cycle0.gif`
(2.3 MB) and `contrast_phase0.gif` (2.0 MB), stills and `gifs.json` (code `34fd088`).

Observed in the slice (verified numerically):
- Motion repeats every 50 phases (2.5 s): phases 0, 50, 100 and 150 give identical
  images, so the 200 phases (10 s) hold four breaths, not one. Up to about 6,800
  pixels change between phases. *Correction (2026-10-26):* the repetition is exact for
  these phases but only approximate in general; 19 of 150 phase pairs p, p + 50 differ
  by more than 1%, up to 33% (see the full-series data checks at the end).
- In-plane motion is small: the tumor centroid moves by one pixel (0.15 mm) in x.
- The z = 100 plane contains no liver (its indicator is zero); tumor and artery are
  present.

## J5 structure on the DCE-MOBY z = 100 plane (2026-10-22): completed, poor fit

Question (user): can the existing structure fit this measured-style plane? Evaluation is
image only, as agreed: there is no true motion or reference intensity for these data.

Setup ([phantoms/moby.py](phantoms/moby.py), `dce_moby` path in [run.py](run.py),
`evaluate_image_only` in [evaluation/evaluate.py](evaluation/evaluate.py)):
- Window: frames 1250-1400 of `moby_z100.h5`, t = 26.5-34.0 s, 3 breaths (motion repeats
  every 50 frames). Chosen where tumor enhancement rises fastest: tumor mean rises about
  27% within the window, so both motion and intensity change.
- Grid 248x248, no resampling. Physical coordinates (mm, s) with
  `coordinate_bounds` [[0, 37.2], [0, 37.2], [26.5, 34.0]], normalized inside the model
  by the existing `CoordinateNormalizer`; pixel centers at (i + 0.5) * 0.15 mm (assumed
  from fov = 248 * 0.15). Reference time is the first frame, 26.5 s.
- Intensity divided by the global maximum of the slice, 0.4504, so data lie in [0, 1].
- Even frames observed (76), odd frames held out (75), as in the toy midpoint protocol.
- Model and optimizer exactly J5 (R=2, motion hash 2-8 space / 4-16 time, intensity hash
  8-128, time grid 3-48, rates 0.01/0.001), 8,000 steps of 4,096 points, seed 0.
  Because 3 breaths map to [0, 1] in time like J5's 3 cycles, the time resolutions keep
  their per-cycle meaning.

Result [results/dce_moby_20261022_8k_j5_z100](results/dce_moby_20261022_8k_j5_z100/metrics.json)
(188 s training, 79,336 parameters, code `24889b7`):

| Image metric | Training frames | Held-out midpoints |
|---|---:|---:|
| RRMSE | 16.42% | 18.03% |
| PSNR (dB, range 1) | 48.99 | 48.16 |
| SSIM | 0.99700 | 0.99661 |
| Fixed-sample MSE (final) | 1.83e-05 | 1.49e-05 |

For comparison J3 (same structure, 8k steps, toy) reached 1.63% held-out image RRMSE.
Per-frame held-out RRMSE ranges from 8.0% to 36.5% (median 16.6%).

Where the error is (held-out frames, checked numerically and on a still frame):
- The 0.10% of pixels brighter than raw 0.05 (small vessels, 1-2 px) carry 60% of the
  squared error; pixels on sharp edges (3.85%) carry 66% (the two overlap). Away from
  both, RRMSE is 7.55%. The prediction is visually close; errors line the organ and
  body boundaries and the vessels.
- Training and held-out errors are equal, so this is underfitting, not overfitting.

Not established (untested explanations): the intensity hash maximum 128 is coarser than
the 248-pixel grid, which would limit edges and 1-pixel vessels; each training point is
sampled about 7 times in 8k steps versus about 21 in the toy; the loss was still
decreasing when the cosine rate reached zero. Motion and reference intensity are not
evaluated, so whether the branches separate correctly is unknown. Single seed.

Visual check (2026-10-22): [evaluation/moby_fit_gif.py](evaluation/moby_fit_gif.py) draws
whole-image truth (left) and prediction (right) from a dce_moby run's saved predictions,
in raw p0 units with the fixed 0..0.05 display range of the data GIFs.
[visuals/midpoint_truth_vs_prediction.gif](results/dce_moby_20261022_8k_j5_z100/visuals/midpoint_truth_vs_prediction.gif)
shows the 75 held-out frames (6.5 MB); `predictions.npz` itself stays local (34 MB).

## DCE-MOBY whole-image tuning (2026-10-22/23): completed

Goal (user, 2026-10-22): before defining dynamic or branch metrics, tune the existing
structure on the same z = 100 window for the three whole-image metrics (RRMSE, PSNR,
SSIM). Approved plan: about 20 config-only runs in stages (intensity hash, motion hash,
learning rates, step budget and batch, extra seeds). Held-out midpoint image RRMSE
selects; PSNR ranks identically on the same data, SSIM is reported. Window, even/odd
split, loss, R=2, anchor and evaluation are unchanged. Selecting on the held-out
midpoints makes them a validation set: no independent test frames remain.

Setup: start from the J5 structure of `dce_moby_20261022_8k_j5_z100` (8k steps), run
on the second machine (Apple M6, CPU, torch 2.14.0). Inputs are generated from that
configuration; each run's `config.json` is authoritative.

**Result (24 runs, S0-S5):** the best setting is J5 with an intensity hash of 8-512
(table 2^19), unchanged motion hash and learning rates, and 12k steps of 16,384
points (`threads=1`). Over seeds 0-2 (t18, t20, t21) it gives held-out image RRMSE
**12.63% +- 0.12**, PSNR **51.25 +- 0.08 dB** and SSIM **0.9985**. The J5
configuration itself gave 18.03-19.35% (three single runs). The remaining error is
mainly the gap from training to held-out frames: training RRMSE is 7.15% +- 0.41.
More intensity capacity (8-1024) lowers training error further but not held-out
error. Single window, one data set, held-out frames used for selection.

| Run | Change from J5 | Train RRMSE | Held-out RRMSE | PSNR (dB) | SSIM | Params |
|---|---|---:|---:|---:|---:|---:|
| [t00](results/dce_moby_tuning_20261022_8k_t00_base/metrics.json) | none | 17.23% | 18.68% | 47.85 | 0.99652 | 79,336 |
| [t00b](results/dce_moby_tuning_20261022_8k_t00b_base_repeat/metrics.json) | none (repeat) | 17.91% | 19.35% | 47.54 | 0.99632 | 79,336 |
| [t01](results/dce_moby_tuning_20261022_8k_t01_ih256/metrics.json) | intensity hash 8-256, table 2^17 | 14.16% | 16.28% | 49.04 | 0.99718 | 229,102 |
| [t02](results/dce_moby_tuning_20261022_8k_t02_ih512/metrics.json) | intensity hash 8-512, table 2^19 | 12.19% | 14.48% | 50.06 | 0.99809 | 776,062 |
| [t03](results/dce_moby_tuning_20261022_8k_t03_ih512_l12/metrics.json) | as t02, 12 levels | 12.95% | 15.31% | 49.57 | 0.99780 | 1,010,706 |
| [t08](results/dce_moby_tuning_20261022_8k_t08_ih1024/metrics.json) | intensity hash 8-1024, table 2^21 | 10.93% | 14.09% | 50.29 | 0.99840 | 2,822,768 |
| [t04](results/dce_moby_tuning_20261022_8k_t04_ih512_mt32/metrics.json) | t02 + motion t max 32 | 11.48% | 14.25% | 50.20 | 0.99813 | 780,506 |
| [t05](results/dce_moby_tuning_20261022_8k_t05_ih512_mt64/metrics.json) | t02 + motion t max 64 | 10.32% | 14.38% | 50.12 | 0.99814 | 788,888 |
| [t06](results/dce_moby_tuning_20261022_8k_t06_ih512_ms16/metrics.json) | t02 + motion x,y max 16 | 14.42% | 16.54% | 48.90 | 0.99761 | 787,444 |
| [t07](results/dce_moby_tuning_20261022_8k_t07_ih512_ms32/metrics.json) | t02 + motion x,y max 32 | 14.30% | 16.63% | 48.86 | 0.99770 | 828,742 |

Table sizes keep every level dense (vertex count at most the table size). Motion tables
stay 2^15, dense for all motion settings above. All runs: seed 0, `threads=4`.

**Run-to-run variation:** with `threads=4` the training is not bitwise reproducible.
The same J5 configuration gave 18.03% (first machine), 18.68% and 19.35% (this machine);
loss histories agree at step 1 and differ by step 500, with identical code, torch,
numpy and Python versions. In a 600-step check, two `threads=1` runs had identical
loss histories while two `threads=4` runs did not; `threads=1` was about 1.45 times
slower per run. From S3 on, all runs use `threads=1` (a compute setting only).
Single-run differences below about 1 point are not resolved.

**Findings so far (single runs):**
- Intensity hash maximum 128 -> 256 -> 512 lowers held-out RRMSE 18.7% -> 16.3% ->
  14.5%, with training RRMSE falling alike: the J5 run was underfitting spatially.
  Twelve levels at 512 were not better; 1024 (14.1%) is within the variation of 512.
- A finer spatial motion hash (x,y maximum 16 or 32) is worse by about 2 points.
  A finer motion time axis (32 or 64) lowers training RRMSE (11.5%, 10.3%) but not
  held-out RRMSE, and raises the worst held-out frame (36.0%, 37.6% versus 30.6%).
- SSIM follows RRMSE but stays within 0.9963-0.9984.
- t02 (intensity 8-512, motion unchanged) is carried forward as the simplest of the
  tied best settings.

**S3: seeds and learning rates** (t02 configuration, `threads=1`, 8k steps):

| Run | Change from t02 | Train RRMSE | Held-out RRMSE | PSNR (dB) | SSIM |
|---|---|---:|---:|---:|---:|
| [t09](results/dce_moby_tuning_20261022_8k_t09_ih512_th1_s0/metrics.json) | seed 0 | 12.00% | 14.69% | 49.93 | 0.99810 |
| [t10](results/dce_moby_tuning_20261022_8k_t10_ih512_th1_s1/metrics.json) | seed 1 | 12.65% | 15.13% | 49.68 | 0.99793 |
| [t11](results/dce_moby_tuning_20261022_8k_t11_ih512_th1_s2/metrics.json) | seed 2 | 12.42% | 14.99% | 49.76 | 0.99800 |
| [t12](results/dce_moby_tuning_20261022_8k_t12_ih512_th1_hlr3e-3/metrics.json) | hash rate 0.003 | 14.37% | 16.31% | 49.02 | 0.99715 |
| [t13](results/dce_moby_tuning_20261022_8k_t13_ih512_th1_hlr3e-2/metrics.json) | hash rate 0.03 | 12.30% | 14.87% | 49.83 | 0.99781 |
| [t14](results/dce_moby_tuning_20261022_8k_t14_ih512_th1_mlr3e-4/metrics.json) | MLP rate 0.0003 | 15.69% | 18.03% | 48.16 | 0.99579 |
| [t15](results/dce_moby_tuning_20261022_8k_t15_ih512_th1_mlr3e-3/metrics.json) | MLP rate 0.003 | 12.29% | 14.92% | 49.80 | 0.99801 |

- Seeds 0-2 give 14.94% +- 0.22 (sample standard deviation): the yardstick for
  single-run differences at 8k.
- Lower rates are worse (hash 0.003: +1.4 points; MLP 0.0003: +3.1 points). Higher
  rates (hash 0.03, MLP 0.003) tie the defaults, so 0.01/0.001 are kept.

**S4: step budget and batch** (t02 configuration, `threads=1`, seed 0, predictions
saved locally; t19 uses the intensity 8-1024 hash of t08):

| Run | Steps x batch | Train RRMSE | Held-out RRMSE | PSNR (dB) | SSIM | Worst held-out frame | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| [t09](results/dce_moby_tuning_20261022_8k_t09_ih512_th1_s0/metrics.json) | 8k x 4,096 | 12.00% | 14.69% | 49.93 | 0.99810 | 31.40% | 399 |
| [t16](results/dce_moby_tuning_20261022_24k_t16_ih512_th1/metrics.json) | 24k x 4,096 | 8.36% | 13.23% | 50.84 | 0.99842 | 31.50% | 1,012 |
| [t17](results/dce_moby_tuning_20261022_48k_t17_ih512_th1/metrics.json) | 48k x 4,096 | 7.66% | 12.81% | 51.12 | 0.99847 | 29.21% | 1,773 |
| [t18](results/dce_moby_tuning_20261022_12k_t18_ih512_th1_b16k/metrics.json) | 12k x 16,384 | 7.01% | 12.68% | 51.21 | 0.99846 | 25.74% | 1,667 |
| [t19](results/dce_moby_tuning_20261022_24k_t19_ih1024_th1/metrics.json) | 24k x 4,096, intensity 8-1024 | 5.55% | 12.89% | 51.07 | 0.99853 | 27.67% | 1,076 |

Seconds are training time with four runs in parallel on one thread each.

- More samples help with diminishing returns: 14.7% -> 13.2% -> 12.8% held-out
  RRMSE for 8k -> 24k -> 48k steps. At the same sample count as t17, the larger batch
  (t18) gives 12.68% with similar wall time.
- Training error keeps falling (to 5.6-8.4%) while held-out error levels off near
  12.7-12.9%. In t18 the fixed held-out MSE stays at about 8.5e-6 from step 5,000
  while the fixed training MSE keeps decreasing. The remaining error is mostly the
  gap between observed and held-out frames, not fitting capacity.
- The top runs (t17, t18, t19) lie within about one 8k seed standard deviation.
- The fixed-sample training MSE is occasionally above the held-out value (t17: 1.5e-5
  versus 8.5e-6) although the full-grid training RRMSE is lower. Probably a few large
  vessel errors in the 65,536 fixed samples; not investigated further.

**S5: seeds of the best setting and a combination** (12k x 16,384, `threads=1`):

| Run | Setting | Train RRMSE | Held-out RRMSE | PSNR (dB) | SSIM | Worst held-out frame |
|---|---|---:|---:|---:|---:|---:|
| [t18](results/dce_moby_tuning_20261022_12k_t18_ih512_th1_b16k/metrics.json) | intensity 8-512, seed 0 | 7.01% | 12.68% | 51.21 | 0.99846 | 25.74% |
| [t20](results/dce_moby_tuning_20261022_12k_t20_ih512_th1_b16k_s1/metrics.json) | intensity 8-512, seed 1 | 7.61% | 12.71% | 51.19 | 0.99842 | 20.56% |
| [t21](results/dce_moby_tuning_20261022_12k_t21_ih512_th1_b16k_s2/metrics.json) | intensity 8-512, seed 2 | 6.83% | 12.49% | 51.34 | 0.99851 | 25.90% |
| [t22](results/dce_moby_tuning_20261022_12k_t22_ih1024_th1_b16k/metrics.json) | intensity 8-1024, seed 0 | 4.11% | 13.24% | 50.84 | 0.99844 | 30.60% |

- The t18 setting is stable over seeds (held-out RRMSE sd 0.12 points).
- Combining the 8-1024 hash with the larger batch is worse on held-out frames (13.24%)
  while its training RRMSE drops to 4.11%.
- `predictions.npz` and `model.pt` of t16-t22 are kept locally only, as for earlier
  runs. No GIFs were made for these runs; branch, dynamic and tissue-curve metrics
  were not evaluated. The tuning used 24 runs, four more than the planned ~20,
  because of the added repeat, 1024-hash and seed runs.
- Not established: whether the setting transfers to other windows or a longer window.
- Data-only check (2026-10-23, no model): averaging the two true training frames
  around each held-out frame gives 12.76% held-out RRMSE (previous frame alone:
  17.65%), close to the best model's 12.63%. Per-frame model errors correlate with
  this interpolation error (r = 0.85 for t18). The 15 hardest frames lie at fast
  breathing transitions and are intermediate states, not copies of a neighbour (1 of
  15); there t18 averages 18.1% versus 21.2% for the truth average. The held-out
  plateau therefore mostly reflects how much the image changes between 0.1 s training
  frames. At 8k steps a finer motion time axis lowered the easier frames slightly
  (11.7% -> 11.0%) but not the hardest ones; the user decided not to test it further.

## Plateau schedule and early stop (2026-10-23): implemented

Approved by the user (2026-10-23) so that the step budget no longer needs tuning.
Optional `"schedule": "plateau"` for dce_moby ([run.py](run.py), documented in
[experiments/README.md](experiments/README.md)) with
`"held_out_split": "alternate_validation_test"`
([phantoms/moby.py](phantoms/moby.py) `split_midpoints`): the 75 midpoints become 38
validation frames (window frames 1, 5, 9, ...) and 37 test frames (3, 7, 11, ...).
Every 500 steps the MSE on all validation frames sets `ReduceLROnPlateau` (factor
0.5, patience 3, relative threshold 1e-3, all rate groups together); the plateau
after 4 reductions stops the run, `steps` is an upper limit, and the weights with the
lowest validation-frame MSE are kept and evaluated. Validation labels never enter the
loss; test frames are not seen in training or monitoring. `metrics.json` reports
training, validation and test frames; the summary row uses the test frames.
[evaluation/moby_split_eval.py](evaluation/moby_split_eval.py) re-scores saved runs
from `model.pt` on the same test frames (`metrics_alternate_split.json`).

**Validation completed:** 122 unit tests pass (5 new in
[tests/test_plateau_schedule.py](tests/test_plateau_schedule.py): split, rate
reductions and stop on a synthetic plateau, kept weights equal the best logged
validation MSE, configuration rules, end-to-end outputs and re-scoring). Without
`schedule`, a 400-step `threads=1` dce_moby run reproduces the pre-change loss
history, metrics and predictions exactly. The reloaded t18 model reproduces its saved
`metrics.json` exactly. A 3000-step smoke run (patience 1, at most 2 reductions,
every 100 steps) halved all rates once after two non-improving evaluations, stopped
at the step limit and kept step 2,900; one full validation evaluation costs about
5 s. The smoke run was not kept.

Cosine runs re-scored on the 37 test frames
([t16](results/dce_moby_tuning_20261022_24k_t16_ih512_th1/metrics_alternate_split.json),
[t17](results/dce_moby_tuning_20261022_48k_t17_ih512_th1/metrics_alternate_split.json),
[t18](results/dce_moby_tuning_20261022_12k_t18_ih512_th1_b16k/metrics_alternate_split.json),
[t19](results/dce_moby_tuning_20261022_24k_t19_ih1024_th1/metrics_alternate_split.json),
[t20](results/dce_moby_tuning_20261022_12k_t20_ih512_th1_b16k_s1/metrics_alternate_split.json),
[t21](results/dce_moby_tuning_20261022_12k_t21_ih512_th1_b16k_s2/metrics_alternate_split.json),
[t22](results/dce_moby_tuning_20261022_12k_t22_ih1024_th1_b16k/metrics_alternate_split.json)):
t16 12.31%, t17 12.20%, t18 11.99%, t19 12.49%, t20 12.55%, t21 12.00%, t22 12.51%
test-frame RRMSE. The best setting (t18, t20, t21) gives 12.18% +- 0.32 on these
frames, PSNR 51.57 dB, SSIM 0.9985.

**Plateau runs at R=2 and R=3 (2026-10-23): completed.** The best setting (intensity
hash 8-512, batch 16,384, rates 0.01/0.001, `threads=1`) with default plateau settings,
step limit 100k, seeds 0-2, six runs in parallel. Test frames: the same 37 as above.

| Run | Rank, seed | Kept / stop step | Reduction steps | Train RRMSE | Validation RRMSE | Test RRMSE | Test PSNR (dB) | Test SSIM | Minutes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| [p01](results/dce_moby_plateau_20261023_p01_ih512_b16k_r2_s0/metrics.json) | R=2, 0 | 22,500 / 24,500 | 12.5k, 17k, 19k, 21k | 4.53% | 12.73% | 11.64% | 51.96 | 0.99863 | 88 |
| [p02](results/dce_moby_plateau_20261023_p02_ih512_b16k_r2_s1/metrics.json) | R=2, 1 | 12,000 / 18,000 | 7.5k, 10.5k, 14k, 16k | 7.69% | 13.42% | 12.18% | 51.57 | 0.99841 | 68 |
| [p03](results/dce_moby_plateau_20261023_p03_ih512_b16k_r2_s2/metrics.json) | R=2, 2 | 22,000 / 26,000 | 9k, 15k, 20k, 24k | 5.14% | 12.37% | 12.07% | 51.65 | 0.99856 | 91 |
| [p04](results/dce_moby_plateau_20261023_p04_ih512_b16k_r3_s0/metrics.json) | R=3, 0 | 15,000 / 23,000 | 11.5k, 17k, 19k, 21k | 6.50% | 13.12% | 12.21% | 51.55 | 0.99850 | 85 |
| [p05](results/dce_moby_plateau_20261023_p05_ih512_b16k_r3_s1/metrics.json) | R=3, 1 | 18,000 / 22,000 | 8.5k, 13k, 16k, 20k | 5.42% | 13.41% | 12.38% | 51.43 | 0.99860 | 82 |
| [p06](results/dce_moby_plateau_20261023_p06_ih512_b16k_r3_s2/metrics.json) | R=3, 2 | 19,500 / 21,500 | 10.5k, 13k, 16k, 18k | 4.27% | 12.87% | 12.37% | 51.44 | 0.99860 | 80 |

Minutes are training time with six runs in parallel on one thread each.

- All six runs stopped at the plateau after four reductions (18k-26k steps); none
  reached the step limit. The kept step varies by seed (12k-22.5k).
- R=2: test RRMSE 11.96% +- 0.28, PSNR 51.73 dB, SSIM 0.9985, versus 12.18% +- 0.32
  for the cosine 12k runs (t18, t20, t21) on the same frames: within seed variation,
  with no step budget to choose, at about three times the training time.
- R=3: test RRMSE 12.32% +- 0.10, PSNR 51.47 dB, SSIM 0.9986. It adds 98 parameters
  and is not better than R=2 on the whole-image metrics; training RRMSE is similar
  (5.40% versus 5.79%).
- Training RRMSE falls to 4-8% while validation and test stay near 12-13%: the same
  observed-to-held-out gap as before.
- Not established: branch, dynamic or tissue-curve behaviour at R=3; other windows.
  `predictions.npz` and `model.pt` stay local.

## Branch GIFs of the best R=2 run (2026-10-23): motion partly captured, not cleanly separated

On request, [evaluation/moby_branch_gifs.py](evaluation/moby_branch_gifs.py) draws three
views of p01 (R=2, seed 0, kept step 22,500) over all 151 window frames in time order,
from `model.pt`; there is no true motion or reference intensity for these data:
[image](results/dce_moby_plateau_20261023_p01_ih512_b16k_r2_s0/visuals/window_image.gif)
(truth | prediction | absolute error),
[intensity](results/dce_moby_plateau_20261023_p01_ih512_b16k_r2_s0/visuals/window_intensity.gif)
(J(q, t) at fixed pixels, no motion | J(q, t) - J(q, t0)) and
[motion](results/dce_moby_plateau_20261023_p01_ih512_b16k_r2_s0/visuals/window_motion.gif)
(J(r + d(r, t), t0), learned intensity frozen at t0 = 26.5 s and moved by the learned
motion | |d(r, t)| in mm). These are the toy `reference_intensity` and `motion_only`
definitions. Fixed raw ranges: images 0..0.05, error 0..0.02, change +-0.01,
displacement 0..0.55 mm. One shared undithered palette keeps the GIFs at 5.4, 4.5 and
10.9 MB. [branch_gifs.json](results/dce_moby_plateau_20261023_p01_ih512_b16k_r2_s0/visuals/branch_gifs.json)
stores the settings and, per frame, ||V(t) - V(t0)|| / ||V(t0)|| for each view.
[tests/test_moby_branch_gifs.py](tests/test_moby_branch_gifs.py) (3 tests) passes; all
125 tests pass.

Observed (relative change from t0; frames 50, 100, 150 share t0's breathing geometry,
verified before injection, so the truth changes there by contrast only):

| View | Max | At frames 50 / 100 / 150 | Mean over other frames | Profile correlation with truth |
|---|---:|---:|---:|---:|
| Truth | 79.5% | 3.1 / 5.7 / 7.9% | 55.4% | 1 |
| Prediction | 80.1% | 9.1 / 10.3 / 11.3% | 55.3% | 0.99 |
| Intensity J(q, t) | 68.5% | 14.5 / 14.6 / 15.2% | 46.8% | 0.95 |
| Motion only J(r + d, t0) | 72.0% | 19.8 / 19.8 / 20.1% | 52.8% | 0.88 |

- The intensity branch carries most of the image change from breathing: J at fixed
  coordinates changes by up to 68.5%, following the truth's breathing profile, where
  contrast alone would change it by about 3-8%. The change view shows red/blue dipoles
  along internal organ boundaries, i.e. moving edges. These norms are dominated by
  high-contrast edges, so shifts of 1-2 pixels already give large relative changes.
- The true in-plane motion is small, and the learned motion matches its size and
  direction at the body outline (ad hoc checks on the truth, from `model.pt`). Over the
  window the body area varies by 0.84%; the body outline moves at most by a mean of
  1.52 px (0.23 mm; 95th percentile 3 px, maximum 4 px) from frame 0, at frame 27. On
  that outline the learned |d| is 1.46 px on average (95th percentile 2.63 px).
  Moving frame 0 by the learned motion lowers the symmetric outline distance to frame k
  from about 1.5 to 0.7 px at mid-breath frames (22, 27, 75, 125) and raises the tumor
  IoU from 0.65 to 0.72-0.74.
- The learned motion is smooth (maximum 0.50 mm): the body-mean |d| is about 0.19 mm
  at frames 25 and 75 and about 0.07 mm at frames 50, 100 and 150. It follows the
  breathing period but does not return to zero at t0's geometry; at frame 40, close
  to that geometry, it lowers the tumor IoU from 0.81 to 0.71.
- Contrast is not cleanly in the intensity branch either. Ad hoc tumor means at frames
  50/100/150 relative to t0 (each frame's own indicator; J with t0's indicator): truth
  9.3/18.5/27.6%, prediction 11.3/17.9/20.7%, J at fixed coordinates 2.4/4.6/6.2%; tumor
  core truth 19.1/38.8/58.7%, prediction 20.7/30.2/34.7%, J 11.9/14.2/16.2%.
- Not established: how much of the breathing change an in-plane displacement can
  represent at all. The z = 100 slice has out-of-plane motion (tumor area per phase
  245-284 pixels), so part of it must appear as intensity change in a 2D model.
  Visual and ad hoc diagnostics only; no branch metric is defined for these data.

## Semiseparable SS(r) baseline on the DCE-MOBY window (2026-10-25): completed

Approved by the user as the first baseline (2026-10-25), following the SS(r) comparisons
in Lozenski et al. 2022 and 2024. [evaluation/moby_svd_baseline.py](evaluation/moby_svd_baseline.py)
takes the uncentered truncated SVD of the 76 training frames (61,504 x 76), so
f(r, t) ~ sum_i u_i(r) v_i(t) with r (61,504 + 76) parameters. Validation and test frames
never enter the SVD; they use v_i(t) linearly interpolated in time. Scores come from the
same image report and the same 37 test frames as the neural fields.
[tests/test_moby_svd_baseline.py](tests/test_moby_svd_baseline.py) (2 tests: exact
recovery of rank-2 data with linear curves, outputs) passes; 127 tests pass. Results:
[svd_ranks.csv](results/dce_moby_svd_20261025_z100_w1250/svd_ranks.csv) (r = 1-76),
[svd_baseline.json](results/dce_moby_svd_20261025_z100_w1250/svd_baseline.json), full
reports for r = 1, 12, 13, 46 and
[rrmse_vs_parameters.png](results/dce_moby_svd_20261025_z100_w1250/rrmse_vs_parameters.png).
The J5 runs t00/t00b were re-scored on the test frames for this comparison.

| Representation | Parameters | Test RRMSE | Test PSNR (dB) | Test SSIM |
|---|---:|---:|---:|---:|
| SS(1) | 61,580 | 37.70% | | |
| NF, J5 structure, 8k (t00 / t00b) | 79,336 | 18.25% / 18.77% | 48.06 / 47.81 | 0.9967 / 0.9964 |
| SS(2) | 123,160 | 22.50% | | |
| SS(5) | 307,900 | 13.11% | | |
| SS(8) | 492,640 | 12.20% | | |
| SS(12) | 738,960 | 12.19% | 51.56 | 0.99869 |
| NF best, plateau (p01-p03) | 776,062 | 11.96% +- 0.28 | 51.73 | 0.9985 |
| NF best, cosine 12k (t18, t20, t21) | 776,062 | 12.18% +- 0.32 | 51.57 | 0.9985 |
| SS(13) | 800,540 | 12.28% | | |
| NF hash 8-1024 (t19 / t22) | 2,822,768 | 12.49% / 12.51% | | |
| SS(r), r >= 30 | >= 1.85 M | 12.14% | | |

- SS(r) reaches a floor of 12.14% (minimum 12.12% at r = 25) once it reproduces the
  training frames (training RRMSE 0.36% at r = 30, 0 at r = 76). At full rank it equals
  linear interpolation of the true training frames, the data-only check above.
- At a small budget the NF is much better: 79k parameters give 18.3-18.8% versus 37.7%
  for SS(1) and 22.5% for SS(2) with 1.6 times the parameters.
- At the tuned budget the two are comparable: 11.96% +- 0.28 (plateau) versus 12.19% for
  SS(12) and 12.28% for SS(13). The best NF seed (11.64%) is below the SS floor, but
  SS(8) already reaches 12.20% with 493k parameters, and SS(12) has the higher SSIM.
- Both are limited by the change between 0.1 s training frames; this window cannot
  separate them further. Consistent with Lozenski et al. 2024, where the NF matched a
  semiseparable approximation of similar size in embedding.

## Embedding study in the format of Lozenski et al. 2022 (2026-10-25): completed

Approved by the user (2026-10-25): compare only SS(r) and the current architecture, under
both protocols, and add a compute-device option for later full-frame runs on Vista.

Code (commits `13f0cb8`, `e648a69`):
- Optional `"device"` (`cpu` default, `cuda`, `mps`) in [run.py](run.py); batches are still
  drawn with the CPU generator and `model.pt` is saved on the CPU. Without it, 400-step
  cosine and plateau runs reproduce the pre-change loss histories, metrics and
  predictions exactly. On this machine's MPS a 400-step run matched the CPU at steps 0-1
  and then drifted slightly (held-out RRMSE 30.08% versus 30.30%), 2.4 times faster.
  **CUDA is not verified yet** (no NVIDIA GPU here).
- Protocol A, `"held_out_split": "none"` ([phantoms/moby.py](phantoms/moby.py)): all 151
  window frames are fitted and scored (`all_window_frames`). Protocol B is the held-out
  setting above (37 test frames).
- [evaluation/moby_svd_baseline.py](evaluation/moby_svd_baseline.py) `--protocol embedding`;
  [evaluation/moby_embedding_figures.py](evaluation/moby_embedding_figures.py) draws the
  paper's Figs. 4-6 and Table IV for our data, and in place of the POUnet partition maps
  (Fig. 7) the NF's intensity change and displacement at six times. SS ranks follow the
  paper's rules (closest parameter count, first to reach the NF's SSIM, first to reach its
  RRMSE). Frames: the scored frame nearest to and farthest from window frame 0. Points:
  tumor centroid at frame 0 and the body pixel with the largest temporal std.
- [tests/test_moby_embedding.py](tests/test_moby_embedding.py) (3 tests); 130 tests pass.

Protocol A runs: the best setting (intensity hash 8-512, 12k cosine steps of 16,384
points, `threads=1`) on all frames, seeds 0-2
([e01](results/dce_moby_embedding_20261025_12k_e01_ih512_b16k_r2_s0/metrics.json),
[e02](results/dce_moby_embedding_20261025_12k_e02_ih512_b16k_r2_s1/metrics.json),
[e03](results/dce_moby_embedding_20261025_12k_e03_ih512_b16k_r2_s2/metrics.json)):
8.64%, 9.14%, 8.81% all-frame RRMSE, 26 minutes each with three in parallel.
SS(r) embedding: [svd_ranks.csv](results/dce_moby_svd_20261025_z100_w1250_embedding/svd_ranks.csv).

| Protocol | Representation | Parameters | RRMSE | PSNR (dB) | SSIM |
|---|---|---:|---:|---:|---:|
| A (all 151 frames) | SS(6) | 369,930 | 9.07% | 54.14 | 0.9991 |
| A | SS(7) | 431,585 | 7.99% | 55.23 | 0.9993 |
| A | SS(13) | 801,515 | 5.18% | 59.00 | 0.9996 |
| A | NF (3 runs) | 776,062 | 8.86% +- 0.25 | 54.34 +- 0.25 | 0.9991 |
| B (37 test frames) | SS(6) | 369,480 | 12.75% | 51.17 | 0.9985 |
| B | SS(13) | 800,540 | 12.28% | 51.50 | 0.9987 |
| B | SS(25) | 1,539,500 | 12.12% | 51.61 | 0.9988 |
| B | NF (plateau, 3 runs) | 776,062 | 11.96% +- 0.28 | 51.73 +- 0.21 | 0.9985 |

Figures and tables: [A](results/dce_moby_embedding_20261025_A_all_frames/table.md),
[B](results/dce_moby_embedding_20261025_B_test_frames/table.md) (each folder holds
`fig_singular_values`, `fig_frames`, `fig_activity_curves`, `fig_nf_decomposition` and
`table` PNGs, plus `embedding_figures.json` with every choice).

- Under the paper's embedding protocol, SS(r) is more parameter-efficient than the NF on
  this window: at about the same size SS(13) reaches 5.18% versus 8.86%, and SS(7)
  already beats the NF with 56% of its parameters. Under the held-out protocol the NF is
  slightly better than every SS(r), whose floor is frame interpolation.
- At the tumor centroid no representation follows the slow enhancement closely in either
  protocol; the breathing pixel is reproduced by all of them.
- Not established: whether the NF's embedding fit is limited by its 12k-step budget,
  which was tuned on the held-out protocol with half as many fitted frames (protocol A
  has no validation frames for the plateau schedule). Single window and slice.

## Full-series fit on Vista (2026-10-26/28): data checks, a 10k-step trial and SS(r) done

Agreed so far (user, 2026-10-26):
- Protocol A (embedding, `held_out_split: "none"`): every frame is fitted and scored.
- No model change for now. Time stays in the existing motion time hash and intensity
  time grid; only their configured resolutions change.
- Rank is chosen from a data-only check (below): the SVD of frames of one respiratory
  phase, i.e. the rank of the contrast change alone.

Scale (checked against the code and the slice file):
- The whole series is 14,401 frames = 288 breaths x 50 frames + 1 (-36 to 684 s), which
  the existing window definition covers exactly (`first_frame` 0, `frames_per_cycle` 50,
  `cycles` 288). That is 61,504 x 14,401 = 885.7 M points; the data are 3.5 GB as float32.
- `run.py` materializes the full coordinate grid (float64 in numpy, 21 GB, plus the
  meshgrid copies; 10.6 GB as float32 on the device). Coordinates must be computed from
  the sampled indices instead. No resume from a checkpoint exists yet.
- Time is normalized over the window, so the current resolutions lose their per-breath
  meaning: the motion time hash (4-16 cells over 3 breaths, 5.3 per breath) would need
  about 1,536 cells for 288 breaths, and the intensity time grid (3-48, 16 per breath)
  4,608 to keep its per-breath resolution.
- Reference effort: e01 sampled each of its 9.3 M points about 21 times (12k x 16,384).
  The same density on the full series would be 18.8 B samples; the frames are highly
  redundant, so the actual budget is to be set from measured throughput and a pilot.

Recorded idea, not to be implemented without approval (M2): drive the motion branch by
the breathing phase psi(t) = (t mod T) / T with T = 2.5 s (50 frames), d(r, psi), so
all breaths share one displacement field. Even this phantom is only approximately
periodic (data checks below: most phases repeat to 0.1%, a few transition phases differ
by up to 33%), and real data would need a respiratory signal or a learned phase; it is
a physical assumption and a model change.

**Data-only checks (2026-10-26): completed.**
[evaluation/moby_series_checks.py](evaluation/moby_series_checks.py) (no model, code
`006286d`, 28 s, 1.2 GB peak memory) on all 14,401 frames of `moby_z100.h5`:
[series_checks.json](results/dce_moby_series_checks_20261026_z100/series_checks.json),
[tissue_curves.png](results/dce_moby_series_checks_20261026_z100/tissue_curves.png),
[time_resolution.png](results/dce_moby_series_checks_20261026_z100/time_resolution.png),
[same_phase_svd.png](results/dce_moby_series_checks_20261026_z100/same_phase_svd.png) and
`tissue_curves.csv`. [tests/test_moby_series_checks.py](tests/test_moby_series_checks.py)
(4 tests) passes; 134 tests pass.

- *Breathing periodicity.* Before injection, frames of the same respiratory phase are
  identical (10 s apart). Phase p and p + 50 differ by a median of 0.10%
  (||f(p) - f(p + 50)|| / ||f(p)||; consecutive phases differ by 13.3%), but 19 of 150
  pairs differ by more than 1% and 12 by more than 5%, up to 33% (phases 31, 36, 81, 86,
  132, 137, ...: fast breathing transitions). Phases 0, 50, 100, 150 are identical, so
  the branch-GIF reasoning above, which used them, stands.
- *Tissue kinetics* (partial-volume weighted mean with each frame's phase indicator;
  timing from the mean over each 50-frame breath). Liver is absent at z = 100.

  | Tissue | Pre-injection | Peak (time) | Rise 10-90% | Breathing modulation, RMS / enhancement range |
  |---|---:|---:|---:|---:|
  | artery | 0.121 | 0.154 (42.7 s) | 23.0 s | 32% |
  | tumor | 0.0078 | 0.044 (97.7 s) | 52.7 s | 2.0% |
  | tumor core | 0.0010 | 0.039 (102.7 s) | 54.0 s | 0.9% |
  | whole image | 0.0030 | 0.0048 (45.3 s) | 25.4 s | 0.6% |

  The kinetics are slow: the fastest rise (artery) takes 23 s, about nine breaths.
  The modulation is what remains after the mask follows the motion; for the artery it
  is large (pre-injection peak-to-peak 128% of its enhancement range). Its cause is not
  established (partial volume of a thin vessel, or out-of-plane motion).
- *Time resolution* (least-squares piecewise-linear fit on N equal cells over the 720 s,
  error relative to the enhancement range). The breath-mean trend needs few cells:
  N = 72 (10 s per cell) gives at most 0.45% (artery), N = 288 (2.5 s, one cell per
  breath) at most 0.09%. Every-frame curves keep the modulation error (artery 32%, tumor
  2.0%) up to N = 288 and only begin to follow breathing from N = 576 (1.25 s). By
  construction the trend is linear between breath means, so its error drop beyond
  N = 576 says nothing about faster kinetics.
- *Rank of the contrast change* (uncentered SVD of the 72-73 frames of respiratory
  phases 0, 10, 20, 30, 40; the five agree within 0.4 points): rank 2 leaves 13.5-13.7%
  of the change from the first frame unexplained (image RRMSE 4.1-4.3%), rank 3 leaves
  0.8% (RRMSE 0.25%), rank 4 0.36% (0.11%).

Adopted from these checks (user, 2026-10-26):
- Rank R = 3: the smallest rank that explains the contrast change at fixed geometry
  (0.8% unexplained versus 13.6% at R = 2). It is a lower bound for the NF, whose
  intensity also has to carry what the motion branch does not.
- Intensity time grid: finest level of at most 288 cells over the series (2.5 s). This
  resolves the kinetics to 0.1% of their range and cannot follow the 2.5 s breathing,
  which then has to go to the motion branch or remain as error (for the artery about
  32% of its enhancement range).

**Draft run design (2026-10-26): input config and job script added (commit `9352257`),
tested by the 10k-step trial below; the formal step budget awaits approval.**
- Input: all 14,401 frames (`first_frame` 0, `cycles` 288, `frames_per_cycle` 50),
  protocol A, R = 3, seed 0, `device` cuda, reference time t0 = -36 s (frame 0).
- Changed from e01: motion time hash 384-1536 cells (same cells per breath as e01:
  1.33-5.33) with table 2^17 so every level stays dense; intensity time grid 3-288 cells;
  batch 65,536 (S4: a larger batch at the same sample count tied or helped); steps set
  after a throughput measurement (e01's density, 21 samples per point, would be about
  288k steps at this batch).
- Unchanged from e01: intensity hash 8-512 (2^19), motion space 2-8, rates 0.01/0.001,
  widths 64/32, image MSE only, cosine schedule, monitoring on fixed training samples.
- Model size 1,308,330 parameters (motion 542,720, intensity spatial 757,630), close to
  SS(17) with 1,290,385 on this series (75,905 per rank).
- Memory: training as coded should peak at an estimated 45 GB host memory (float64
  coordinate grid and its meshgrid copies; not measured) and 10.6 GB on the GPU. The
  whole-array SSIM would need about 114 GB for 14,401 frames (16 float64 copies of the
  volume, traced), so it is now computed in time chunks (below). With that, in-process
  evaluation is estimated at 50-70 GB host memory, dominated by the float64 coordinate
  grids of training and prediction (estimate, not measured).
- Code changes: chunked SSIM (done, below); Slurm scripts and the input config under
  `experiments/`; resume from a checkpoint only if the step budget exceeds the queue
  time limit; coordinates computed per frame block only if the host memory is too small.
  The deferred evaluation proposed first is no longer needed. SS(r) needs no change
  (`--protocol embedding --max-rank 60`, on a CPU node).
- Vista steps: environment (aarch64 torch with CUDA) and data hash check; CUDA check on
  the e01 window (400 steps, CPU versus CUDA); throughput at batch 16k/65k/262k; seed 0
  main run with in-process evaluation; SS(r) on a CPU node; seeds 1-2 if seed 0 is sound.
- Confirmed on Vista (GPU test below): 212 GB host memory, 2-day `gh`/`gg` limits, and
  the source data's `motion_vectors.h5` (displacement vectors per respiratory phase,
  per its README) exists.

**Time-chunked SSIM (2026-10-26): implemented, approved by the user.** skimage's SSIM
holds 16 float64 arrays of the input's size (traced with tracemalloc: exactly 16.0 at
500 and 1,000 frames of 248 x 248). [evaluation/metrics.py](evaluation/metrics.py)
`_ssim` now scores space-time arrays above 2^24 voxels in chunks of the last (time)
axis: each chunk's kept frames are computed with 3-frame halos and only the interior of
the SSIM map is summed, which is the whole-array mean up to rounding. Arrays at or
below 2^24 voxels (every earlier run: the DCE-MOBY window is 9.3 M) use the unchanged
single call. Validated:
- Two new tests in [tests/test_metrics.py](tests/test_metrics.py): 2D+t and 3D+t random
  arrays at four chunk sizes agree with the whole-array value to 1e-13 relative; below
  the threshold the result is bit-identical; `settings.ssim.computation` is written only
  when chunks are used. 136 tests pass.
- Real data: 1,000 DCE-MOBY frames (prediction = next frame) give whole-array
  0.998404721799573 and chunked 0.998404721799575 (relative difference 1.8e-15).
- Re-scoring `dce_moby_20261022_8k_j5_z100` from its `model.pt` reproduces the image
  section of its `metrics.json` exactly.
- Memory (tracemalloc, above the float32 inputs): 2.37 GB at 500 and 2.99 GB at 1,000
  frames, about 1.2 MB per frame, so about 20 GB at 14,401 frames instead of 114 GB.
  About 8 s per 1,000 frames on this machine. macOS RSS readings understate these
  values (compressed memory), so tracemalloc is used.

**Vista GPU test (2026-10-26): completed, CUDA verified.** The user asked for a simple
check that the GPU can be used; nothing else was run.
- Environment: load `gcc/15.1.0` and then `python3/3.11.8` as two separate
  `module load` calls (on one line python3 did not load). Venv `$WORK/envs/nf-crt`
  (5.6 GB) with torch 2.14.0+cu130, numpy 1.26.4, h5py 3.16.0, scipy 1.13.1,
  scikit-image 0.26.0, matplotlib 3.9.4, the local versions except the CUDA build. The
  venv's python needs the module loaded (libpython). Code: a clone of `motion-static`
  at `$WORK/nf_crt_dynamic_imaging` (commit `4123848`); 136 tests pass on the login
  node. `$WORK/moby_slice/moby_z100.h5` has the local file's sha256.
- Test: e01's window setting (protocol A, R = 2, batch 16,384, seed 0) for 400 steps
  with `device` cuda on a `gh-dev` node (job 1102585, `srun`, allocation ASC25089;
  NVIDIA GH200 120GB, 97,871 MiB GPU memory, driver 590.48.01). The step-1 batch MSE
  equals the local CPU run (3.98318e-4); afterwards the runs drift as on MPS:
  all-frame RRMSE 24.21% versus 23.76% on the CPU, SSIM 0.99161 versus 0.99177.
  Training took 14.4 s versus 61.3 s on this machine's CPU (one thread). Outputs stay in
  `$WORK/gpu_test/cuda_400`; nothing was committed from the test.
- Vista facts: the GH200 node has 212 GB host memory, so the estimated 45 GB training
  and 50-70 GB in-process evaluation fit and per-block coordinates are not needed.
  Queue limits: `gh` and `gg` 2 days, `gh-dev` 2 h, `gb` 12 h; resume is needed only for
  runs over 48 h. ASC25089 has 14,947 SUs (expires 2027-10-31).
- The source folder `/work/03453/villa13/vista/proj/moby-data` does contain
  `motion_vectors.h5` (16.5 GB, readable), `phantom_anatomy_0.h5` and `moby_mesh.h5`.
  True motion may therefore exist for these data; the files were not opened.

**10k-step trial on the full series (2026-10-26): completed.** Agreed with the user as a
short job to measure throughput and memory before the formal run; no other change.
- Input [experiments/dce_moby_full_series.json](experiments/dce_moby_full_series.json)
  (the draft design above with 10,000 cosine steps) and the job script
  [experiments/vista/train.sbatch](experiments/vista/train.sbatch), both commit
  `9352257`, which the run's `config.json` records. Job 1102639, `sbatch -p gh-dev -t
  02:00:00`, one node, allocation ASC25089. It started at 18:27:35, the same evening,
  far earlier than Slurm's estimate, and ended after 18 min 04 s with exit code 0
  (about 0.3 SU).
- Records: [dce_moby_full_20261026_10k_f00_ih512_b64k_r3_s0](results/dce_moby_full_20261026_10k_f00_ih512_b64k_r3_s0/)
  (`config.json`, `metrics.json`, `loss.csv`, `loss.png`, `run.log`,
  `monitor_samples.npz`, `source.zip`, the Slurm log and `gpu-1102639.csv`; checksums
  match Vista's). `model.pt` (5.3 MB) stays on Vista in the same folder.
- Throughput: training took 549 s, 54.4 ms per step at batch 65,536 (1.2 M samples/s);
  the 10k steps drew 655 M samples, 0.74 per point. The GPU ran at 97% utilization
  with 14,940 MiB in use (nvidia-smi, includes PyTorch's cache) of 97,871 MiB.
  Evaluation of all 14,401 frames took about 8 min (GPU 25%, then idle during SSIM).
- Host memory peak 67.7 GB (`/usr/bin/time -v` maximum RSS), within the 50-70 GB
  estimate and a third of the node's 212 GB. Per-block coordinates are not needed.
- Fit (protocol A, all frames): RRMSE 10.61%, PSNR 54.67 dB, SSIM 0.99913 (time-chunked);
  activity-curve RRMSE 0.27%. The fixed-sample training MSE fell from 2.65e-4 (step 1) to
  3.34e-6 and was still falling when the cosine schedule reached zero.
- Per-frame RRMSE: median 10.04% (10th-90th percentile 5.45-14.15%, 99th 20.37%). It
  does not grow over the series: the median is 9.9-10.6% in every interval from
  pre-injection to 684 s. It depends strongly on the breathing phase: the median over
  frames of one phase (frame index mod 50) ranges from 3.53% (phase 23) to 16.52%
  (phase 49). The worst frames are the first ones: 55.4% at frame 0 (t = -36 s, the
  motion reference frame) and 38-55% for frames 0-4, against 15.6% at frame 50.
- Not comparable with e01 (8.64% on the 151-frame window): different frames, rank and
  budget. SS(r) on the full series has not been computed. Whether the phase-dependent
  error comes from the step budget, the motion branch or the time resolution is not
  established.
- Projection, not measured: e01's sampling density (21 samples per point) would be
  about 288k steps, 4.4 h of training at this speed plus 8 min of evaluation, about
  4.5 SU on `gh`, well inside its 48 h limit.

**Formal run, SS(r), GIFs and embedding figures (2026-10-27): submitted; SS(r) done, training still queued.**
Approved by the user: e01's sampling density, about 288k steps at batch 65,536, cosine
schedule, seed 0, on `gh`; GIFs of (A) one frame per breath at respiratory phase 0 and
(B) three-breath segments at the start, the artery peak, the tumor peak and the end;
the embedding figures in the format of Lozenski et al. 2022 against SS(r) on the full series.
- Vista's `summary.csv` held the trial's row, byte-identical to the pushed commit; it
  was restored and the clone pulled (commit `b630abe` at submission, `5e52615` later;
  neither changes training or scoring code).
- Jobs (allocation ASC25089; each later job starts only after its inputs succeed):

  | Job | Queue, limit | What | Output in `results/` |
  |---|---|---|---|
  | 1104729 | `gh`, 8 h | `train.sbatch dce_moby_full_series.json --steps 288000` (21.3 samples per point; e01's density is 286,113 steps) | `dce_moby_full_20261027_288k_f01_ih512_b64k_r3_s0` |
  | 1104733 | `gg`, 12 h | `moby_svd_baseline` on the trial run's data settings, `--protocol embedding --max-rank 60`, trial marked | `dce_moby_svd_20261027_z100_full_embedding` |
  | 1104764 | `gg`, 1 h, after 1104729 | branch GIFs A: `--frames 0:14401:50 --name phase0` (289 frames) | the run's `visuals/phase0_*` |
  | 1104765 | `gg`, 1 h, after 1104729 | branch GIFs B: `--frames 0:151 1500:1651 2600:2751 14250:14401 --name segments` (604 frames) | the run's `visuals/segments_*` |
  | 1104766 | `gg`, 6 h, after both | `moby_embedding_figures` with the formal run | `dce_moby_embedding_20261027_full_A_all_frames` |

  Estimates, not measured: training 4.4 h plus 8 min evaluation; SS(r) 4-6 h (about
  5 min of scoring per rank); embedding figures 1-2 h (SVD again and a CPU prediction of
  all 885.7 M points at about 0.3 M points/s); about 7 SU in all.
- New job script [experiments/vista/evaluate.sbatch](experiments/vista/evaluate.sbatch)
  runs one evaluation module on a CPU node. TACC jobs start with `OMP_NUM_THREADS=1`, so
  the first SS(r) submission (1104730) started with one BLAS thread; it was cancelled
  after 1 min, while still loading data, and resubmitted with every core. With 144
  threads, loading, the SVD and the first two ranks took under 13 min.
  The script now gives numpy's BLAS every core and torch 16 threads: CPU inference in
  batches of 8,192 points peaks at about 8 threads (laptop: 0.17, 0.30, 0.33 and 0.28 M
  points/s at 1, 4, 8 and 10 threads).
- [evaluation/moby_branch_gifs.py](evaluation/moby_branch_gifs.py): `--frames`
  (window frame ranges START:STOP[:STEP], which must start at frame 0, the motion
  reference frame from which every change is measured) and `--name` (output prefix;
  the default keeps `window_*.gif` and `branch_gifs.json`). Captions give window frame
  indices; the record adds `frame_ranges` and `window_frames`. Without `--frames` the GIFs
  and stills are byte-identical to the previous code on a tiny run. Two tests added
  ([tests/test_moby_branch_gifs.py](tests/test_moby_branch_gifs.py)); 137 tests pass.
- Fixed a test failure left by commit `9352257`: the experiments layout test allowed
  only `README.md` and `template.json`. It now allows the agreed
  `dce_moby_full_series.json` and `vista/`, and the experiments README says so.
- Preview with the 10k-step trial model, made on this machine (scratch copy of the run
  with the local data path, same sha256; not committed). A: 2.5 min, 6.1 GB, GIFs of
  6.7, 3.5 and 20.6 MB; B: 5 min, 37.3 MB for the largest GIF. What it shows for the trial:
  at the phase-0 frames, where the true geometry repeats, the learned displacement is
  still up to 0.19 mm and the motion-only view differs from t0 by a median 52%. The
  predicted image's change from frame 0 (median 67%) is twice the truth's (32%), because
  frame 0 itself fits worst (55% RRMSE). The intensity-change panel saturates at the
  tumor, whose enhancement (about 0.036 raw) exceeds the default display range of
  +-0.01; the formal GIFs keep the default ranges.
  A further preview of window frames 0-150 (`--frames 0:151`, every frame of the first
  three breaths) shows the breathing, which phase-0 frames cannot show. In-plane motion
  is small (learned |d| at most 0.20 mm, 1.3 px). From the trial's `metrics.json`: the
  truth changes from frame 0 by up to 79% within a breath, while the median per-frame
  RRMSE at each of the 50 breathing phases is 3.5-16.5%, so the prediction follows the
  breathing.

**SS(r) on the full series (job 1104733): completed 2026-10-27.**
[results/dce_moby_svd_20261027_z100_full_embedding/](results/dce_moby_svd_20261027_z100_full_embedding/)
holds `svd_baseline.json`, `svd_ranks.csv`, `rrmse_vs_parameters.png` and the job log
`slurm-1104733.log` (checksums match Vista's).
- 3 h 39 min on one `gg` node (about 1.2 SU; about 3.5 min of scoring per rank), peak
  host memory 40.0 GB, code `b630abe` with 144 threads (from the job log).
  `svd_baseline.json` records `code_commit` `5e52615`, because the module reads the
  commit when it writes the record and the clone was pulled during the run;
  `b630abe..5e52615` changes only the branch GIFs, tests, READMEs and
  `evaluate.sbatch`, not SS(r) or scoring. The record is left as written.
- Embedding protocol, all 14,401 frames fitted and scored; r(61,504 + 14,401) parameters:

  | r | Parameters | RRMSE | PSNR (dB) | SSIM |
  |---:|---:|---:|---:|---:|
  | 1 | 75,905 | 37.72% | 43.65 | 0.9837 |
  | 3 | 227,715 | 18.01% | 50.07 | 0.9969 |
  | 6 | 455,430 | 10.93% | 54.41 | 0.9988 |
  | 7 | 531,335 | 9.94% | 55.23 | 0.9989 |
  | 10 | 759,050 | 7.45% | 57.74 | 0.9995 |
  | 17 | 1,290,385 | 4.82% | 61.52 | 0.9997 |
  | 18 | 1,366,290 | 4.59% | 61.95 | 0.9997 |
  | 30 | 2,277,150 | 2.41% | 67.54 | 0.9999 |
  | 60 | 4,554,300 | 0.55% | 80.40 | 1.0000 |

- The NF has 1,308,330 parameters, between SS(17) and SS(18). The 10k-step trial
  (10.61%) matches SS(6)-SS(7), which use 35-41% of its parameters. The formal run is
  the comparison to make; the trial is marked in the figure only for reference.
- Queue: at 2026-10-28 10:12 CDT the training job 1104729 had been pending for 19.5 h
  (reason Priority; 605 jobs pending on `gh`, 136 of them with higher priority, 572 of
  576 nodes allocated). Slurm gave no start estimate.

Still open:
- Results of the remaining jobs; seeds 1-2 if seed 0 is sound.
- Whether to widen the GIFs' intensity-change display range for the whole series.
- Whether to use `motion_vectors.h5` for motion evaluation (a separate decision).
