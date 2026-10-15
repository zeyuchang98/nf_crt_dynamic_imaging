# Motion and Intensity Separation: Current Progress

Last updated: 2026-10-14.

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
