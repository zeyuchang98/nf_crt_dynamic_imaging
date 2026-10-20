# CLAUDE.md

Concise working rules. Same policy as `AGENTS.md`; if they conflict, follow `AGENTS.md`.

## Project
Neural-field (NF) dynamic imaging for PACT (circular Radon transform).
- `nfCRT/`: original shared library (CRT operator, neural fields, function spaces).
- `applications/`: original paper code (`proximal_splitting/`, `self_supervised_training/`).
- `motion_static/`: current research work (motion/intensity separation on toy phantoms).
  - `run.py` (single training entry), `losses.py`, `training_monitor.py`, `run_records.py`, `splits.py`, `baselines.py`, `hash_sweep.py`
  - `models/` (hash NF, nested, low-rank), `phantoms/`, `evaluation/` (metrics, plots, GIFs), `experiments/template.json`, `tests/`
  - `results/` is local run output (gitignored except selected records); `archive/` is retired material.
- Read `motion_static/PROGRESS.md` (status, conclusions) before work; `METRICS.md` for metric definitions.

## Roles and scope
- The user decides research methods, direction and experiment scope. You implement, debug and validate.
- Ask before resolving ambiguity in math definitions, physical assumptions, model meaning, data processing or evaluation conventions.
- Never change math, physics, evaluation protocols or default parameters/metrics without explicit approval, even if results improve.
- New feature/experiment/substantial task: discuss idea and plan first and get agreement. Exploratory talk is not authorization. After agreement, work independently within scope; ask again only if assumptions or scope change.
- Stop when the task is done. Do not expand experiments. Suggestions are welcome; unrequested implementation is not.

## Code and organization
- Simple, readable code; meaningful names; comments explain the math link and non-obvious choices.
- Reuse shared logic; no frameworks for hypothetical needs; no unrelated refactors.
- Experiment entry points configure shared components. Keep code, config and outputs separate; outputs must be traceable to config and code version.
- No redundant plans/reports; update existing records. Ask before large moves, renames or deletions.
- English for all new/updated project content (code, comments, docs, logs, plots, file names).

## Git
- Start of each task and before launching a run: `git pull --ff-only`; confirm success. If blocked by local changes or divergence, stop and ask. Never stash, reset, discard or force-push.
- Preserve unrelated uncommitted changes; no merge commits.
- After each completed milestone: validate, update `PROGRESS.md`, commit within scope, push (standing authorization). Include code, configs, docs and saved outputs needed to inspect the work; exclude caches. Verify the push before reporting.

## Validation and delivery
- Every training run saves loss curves, numeric loss history and LR schedule; with held-out data, also fixed-sample train/validation losses (validation labels never used for optimization).
- Do focused validation (numerical/mathematical consistency, `motion_static/tests/`). No full experiments or large searches without authorization.
- Report failures and real results as they are; never alter evaluation to hide problems.
- Update `PROGRESS.md` after substantive changes (what was run and verified, results, open issues, agreed next steps). Discussion-only tasks need no update.
- Final report: what changed, what was validated, what remains unresolved; separate verified facts from untested conclusions.
