# Motion and Intensity Separation: Current Progress

Last updated: 2026-10-15.

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
