# Project Working Rules

## Roles and Scope
- The user decides the research methods, research direction, and scope of experiments.
- The agent is responsible for implementing, debugging, and performing necessary validation for assigned tasks.
- Ask the user before resolving ambiguities that affect mathematical definitions, physical assumptions, model meaning, data processing, or evaluation conventions.
- Handle routine implementation details independently, following existing project conventions.
- Within the assigned scope, independently implement, test, debug, and iterate until the acceptance criteria are met. Escalate only decisions that affect scientific meaning or require expanding scope.
- Never modify mathematical definitions, physical assumptions, or evaluation protocols without explicit approval, even if a change appears to improve results.
- Suggestions for improvements are welcome, but do not implement changes outside the assigned scope without the user's approval.
- Stop when the assigned task is complete. Do not independently expand experiments or change the research direction.

## Discuss Before Implementation
- Before starting a new research feature, experiment, or substantial implementation task, first discuss the underlying idea, intended direction, and proposed plan with the user.
- Clarify the goal, relevant mathematical and physical assumptions, expected behavior, scope, and how success will be evaluated. Identify meaningful alternatives and trade-offs when relevant.
- Present a concise implementation plan, including the main steps, files or components likely to change, and validation approach. Obtain the user's agreement on the direction and plan before making substantive changes or launching experiments.
- Do not treat an initial idea or exploratory discussion as authorization to implement it. If the user has already explicitly approved a sufficiently clear plan, proceed without asking again.
- Once the plan is agreed upon, execute routine implementation and debugging independently within its boundaries. Ask again only if a material assumption, direction, scope, or plan needs to change; do not interrupt for minor coding decisions.

## Language
- Use English for all new or updated content in project files, including code identifiers, comments, docstrings, documentation, progress records, configuration descriptions, logs, and plot or report text.
- Use English names for new files and directories. Preserve existing paths unless renaming is part of the approved task.
- Apply this rule within the assigned scope. Preserve mathematical meaning, numerical results, and data values; do not translate unrelated files as part of another task.

## Code Style
- Prefer simple, clear, readable implementations.
- Use meaningful names. Comments should primarily explain connections to the mathematics and the reasoning behind non-obvious design choices.
- Reuse shared logic for models, metrics, and data processing where practical. Avoid copying the same implementation across experiments.
- Extract shared modules only when there is an actual need for reuse. Do not build complex frameworks for hypothetical future requirements.
- Keep changes focused. Do not refactor unrelated code as part of another task.
- Do not silently change default parameters, model behavior, or metric definitions.

## Project Organization
- Organize folders by responsibility, keep nesting moderate, and follow the existing structure where it is reasonable.
- Separate shared implementations from individual experiments. Experiment entry points should primarily configure and invoke shared components.
- Keep code, configuration, and experiment outputs clearly distinguishable.
- Make each experiment's outputs traceable to the configuration and code version used.
- Do not create redundant plans, summaries, or reports. Prefer updating existing records.
- Ask the user before large-scale moves, renaming, or deletion of existing files.
- Preserve unrelated uncommitted changes. Do not create merge commits, reset, or discard existing changes without explicit authorization.

## Git Synchronization
- At the start of each task and before launching an experiment run, pull the latest changes from the current branch's configured GitHub tracking branch using `git pull --ff-only`. Confirm synchronization succeeded before proceeding.
- Preserve local changes during synchronization. If local changes or divergent history prevent a fast-forward pull, stop dependent work and ask the user how to resolve it. Do not automatically stash, reset, discard changes, or force-push.
- After each completed milestone, perform the relevant validation, update the progress record when required, commit the changes within the assigned scope, and push to the branch's configured GitHub remote. This is standing user authorization for routine milestone commits and pushes; no repeated confirmation is needed.
- Include the code, configurations, documentation, and saved experiment outputs needed to reproduce or inspect the completed work, so progress links resolve on GitHub. Exclude caches and unrelated local changes.
- Verify the push and relevant GitHub links before reporting completion. If synchronization or publication fails, report the failure and preserve local work; do not claim that GitHub is up to date.

## Progress Tracking
- Before starting a task, read the relevant existing `PROGRESS.md` to understand completed work, current conclusions, unresolved issues, and user-approved next steps. For `motion_static` work, use `motion_static/PROGRESS.md`.
- After substantive changes to code, experiments, or validated conclusions, update the same progress record before delivery. Record what was completed, what was actually run and verified, the results, remaining issues, and any next steps explicitly agreed upon with the user.
- Keep progress records concise and current. Update existing entries rather than creating duplicate summaries, and distinguish verified results from proposals or untested conclusions.
- Discussion-only tasks do not require a progress update unless the user requests one or confirms a decision that changes the recorded plan.

## Validation and Delivery
- Save loss curves for every future training run, together with the numeric loss history and learning-rate schedule. Where held-out observations exist, also record fixed-sample training and validation losses without using validation labels for optimization.
- Perform focused validation sufficient to verify correctness, including numerical and mathematical consistency where relevant. Do not launch full experiments or large parameter searches without authorization.
- If validation fails, investigate within the assigned scope. Report unresolved failures rather than changing research assumptions or expanding the task to force success.
- Preserve actual results and errors. Do not change the evaluation procedure to conceal problems.
- On completion, briefly explain what changed, what validation was performed, and what remains unresolved.
- Clearly distinguish what was actually run and verified from conclusions that have not yet been validated.
