# Evaluation and optional visualization

[METRICS.md](../METRICS.md) documents formulas and current implementation status.
This layout update preserves numerical definitions, evaluation grids, masks,
reference coordinates and intensity ranges. No proposed metric was implemented.

## Default evaluation

The shared runner writes one `metrics.json` and no automatic visualization files.
It retains the existing full-grid, ROI, per-frame and temporal-curve measurements.

| JSON key | Contents |
|---|---|
| `evaluation_split` | `midpoints` or `blocks`; these are different protocols |
| `image` | Image metrics on the declared train/evaluation selections |
| `reference_intensity` | Midpoint protocol: fixed-reference intensity metrics |
| `reference_intensity_full_grid` | Block protocol: separate full-grid intensity diagnosis |
| `motion` | Direct displacement errors, foreground/FOV selections, framewise errors and existing auxiliary diagnostics |
| `motion_components` | Existing ux/uy scalar metric reports |
| `diagnostics` / `diagnostics_on_test_pixels` | Existing branch replacement checks |
| `intensity_rank` | Uncentered spatial-by-time singular values and rank tail, when applicable |

Scalar computations remain in `metrics.py`; masked spatial-window SSIM remains in
`masked_metrics.py`. Midpoint and block SSIM conventions are still distinct.
Raw predictions are never clipped before metrics. Turning off exports does not
subsample evaluation times or use the monitoring subset as the evaluation set.

## Generate GIFs for selected runs

```bash
python -m motion_static.evaluation.render /path/to/run --views image intensity motion motion_only
```

| View | Meaning |
|---|---|
| `image` | Full image prediction versus truth |
| `intensity` | Intensity at fixed reference coordinates, without motion |
| `motion` | True motion (left), predicted motion (middle), absolute image error (right); both use true intensity frozen at t0 |
| `motion_only` | Learned intensity frozen at the reference time, then warped by learned motion |

Only `visuals/` is created or updated. The command checks the saved configuration,
checkpoint and numerical source hashes, then performs inference without optimizer
updates. It uses all configured evaluation times and the saved block mask where
applicable. Model, phantom and data-construction sources must match their saved
hashes. A change confined to the unused `evaluate()` reporting entrypoint is
accepted only after comparing the remaining module syntax with the hash-verified
`source.zip`. Changed inference functions still require the original sources.
`motion_only`'s fixed first panel is intentional
and its third panel is change from that reference, not motion ground-truth error.

The standard `motion_effect.gif` definition, adopted on 2026-10-15, is

\[
L(r,t)=I_{\mathrm{true}}(r+u_{\mathrm{true}}(r,t),t_0),\qquad
P(r,t)=I_{\mathrm{true}}(r+\phi_{\mathrm{motion}}(r,t),t_0),\qquad
E(r,t)=|P(r,t)-L(r,t)|.
\]

Here t0 is the model's motion reference time, not the first displayed midpoint.
Displacements are evaluated at each current time t. Both panels query the same
continuous true phantom intensity at t0; no learned intensity network enters
this comparison. [motion_visuals.py](motion_visuals.py) shares this definition
across the standard renderer, branch diagnostics and multi-run comparisons.
Display ranges remain fixed over time. This is an image visualization of motion;
direct displacement metrics and the existing time-varying branch-replacement
metrics are unchanged. GIFs remain opt-in.

## Other explicit analysis commands

`evaluation.summarize` regenerates `results/summary.csv`; `summarize_blocks` is a
compatibility alias to the same protocol-labeled index. `diagnose_branches` and
`compare_gifs` can read explicitly saved predictions or infer them from a model.
`compare_training` reads the compact metric record and any explicitly saved
checkpoints. These commands are never invoked automatically by training.
Known-shape geometry diagnosis remains specific to the toy phantom.

`moby_svd_baseline` computes the semiseparable baseline SS(r) for a dce_moby window,
with `--protocol heldout` (fit the training frames, interpolate held-out frames) or
`--protocol embedding` (fit and score every window frame). `moby_embedding_figures`
draws the embedding-study figures and table in the format of Lozenski et al. 2022 for
SS(r) against saved neural-field runs of the same protocol. `moby_branch_gifs` animates
a run's image, fixed-coordinate intensity and motion-only views.

`moby_split_eval` re-scores saved dce_moby runs (from their `model.pt`) on the
alternate validation/test split of the midpoints and writes
`metrics_alternate_split.json`, leaving `metrics.json` unchanged; this compares
cosine runs with plateau-schedule runs on the same test frames. `moby_fit_gif`
draws the test frames by default for runs that use that split.

`monitor_samples.npz` stores fixed training/validation pool indices used by
`loss.csv`. The generator is independent of training batches, and validation has
no gradients. For block experiments, indices address the selected observed and
held-out pools respectively. `split.npz` preserves the corresponding exact masks.
