# Historical experiment records

These directories are **records of completed experiments**, not resumable runs.
The approved cleanup retains configurations, metrics, verification reports,
source snapshots, numeric loss/LR histories, and loss curves. It removes trained
models, prediction arrays, other figures/GIFs, and archived smoke-run bundles.
Scientific numbers and metric definitions have not been modified.

The research narrative and interpretation are in [PROGRESS.md](../PROGRESS.md).

| Experiment group | Retained records |
|---|---|
| Initial simple and complex baselines | [Original comparison](comparison.json), [verification](verification.json) |
| Original nested hash | [Configuration](complex/nested_hash_nf/config.json), [metrics](complex/nested_hash_nf/summary.json), [loss](complex/nested_hash_nf/plots/logged_loss_curve.png) |
| Early low-rank R=2 | [Configuration](complex/low_rank/rank_2/config.json), [metrics](complex/low_rank/rank_2/summary.json), [loss](complex/low_rank/rank_2/plots/logged_loss_curve.png) |
| Early low-rank R=3 | [Configuration](complex/low_rank/rank_3/config.json), [metrics](complex/low_rank/rank_3/summary.json), [loss](complex/low_rank/rank_3/plots/logged_loss_curve.png) |
| Multiple cycles and longer training | [Record](complex/low_rank_multicycle/README.md) |
| Larger motion grid cells | [Record](complex/low_rank_coarse_motion/README.md) |
| Intensity maximum 64 / 32 | [Max-64 comparison](complex/low_rank_intensity_max64/comparison.json), [max-32 comparison](complex/low_rank_intensity_max32/comparison.json) |
| Local 24k sweep, 10 completed | [Table](complex/hash_sweep_24k/README.md), [comparison](complex/hash_sweep_24k/comparison.json) |
| Second-machine 24k sweep, 36 completed | [Table](complex/hash_sweep_24k_machine2/README.md), [comparison](complex/hash_sweep_24k_machine2/comparison.json) |

The second-machine compact copy contains all 36 configurations, verification
reports, numeric loss/LR histories and loss curves, plus the aggregate comparison
and final validation report. Detailed historical prediction-based metric files
were not copied into this compact baseline; the aggregate comparison is retained. Matching local and second-machine
run names identify independent runs; they must not be combined as duplicate measurements.

## Removed artifacts and provenance

The user requested removal of the old motion-static Git history after the file
cleanup. Historical models, raw predictions, non-loss figures/GIFs and legacy code
are no longer retained through a supported repository history reference. Old source
commit IDs in configurations and verification reports are provenance labels only.
Retained source snapshots and `../BASELINE.json` document the implementation.

The early R=2 visual preference remains recorded in `../PROGRESS.md`, but its old
GIF was removed. Generating new model outputs requires a new run. Existing
verification reports describe checks performed before cleanup, not the contents
of these now-compact record directories. The numerical records are unchanged.

## Fresh runs

Follow the [training guide](../README.md#run-a-fresh-experiment) and choose a new
output directory. New outputs are ignored by Git; retain compact records deliberately.
