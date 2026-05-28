# Goal Log: MJLab QS Rollout Policy Improvement

Date started: 2026-05-28
Branch: `mjlab-qs-rollout-policy-improvement`

## Scope

Continue the Velocity Flat Unitree G1 MJLab QS / PWM-Flow project with rollout-first evidence. Use prior completed work as evidence where it already satisfies the goal, especially `docs/phaseA/pwm_extension_submission_20260507.md` and `docs/plans/mjlab_pwm_flow_next_steps_20260528.md`.

## Current Evidence

- Collector/reference rollouts exist with W&B videos and MP4s. Expert collector return is `82.6090`, length is `1000.00`, and fall rate is `0.000`.
- Expert-noisy collector is similarly stable: return `80.3525`, length `1000.00`, fall rate `0.000`.
- Medium collector reaches return `49.1935`, length `653.33`, fall rate `0.667`.
- Expert-filtered BC is nontrivial but unstable: return `19.0827`, length `238.22`, fall rate `0.333`.
- Current PWM-style extracted policies remain collapsed. The least-bad conservative row is return `-1.0999`, length `54.67`, fall rate `1.000`.
- True best actor snapshots are now saved for new policy-extraction runs and rollout tooling evaluates final plus true-best checkpoints.

## Work This Goal

- Created this concise goal log so future turns have a single current-state checkpoint.
- Added required protocol docs: `docs/RUNBOOK.md`, `docs/EXPERIMENT_LEDGER.md`, `docs/DATASET_CARD_MJLAB_QS.md`, `docs/CLAIM_POLICY.md`, and `docs/DEBUG_TREE.md`.
- Tightened formal-run metadata logging for policy extraction and rollout renderers:
  - W&B config and local summaries now include git branch and command line.
  - Policy extraction summaries now include dataset, metadata, normalization, WM checkpoint, output directory, final checkpoint, and best checkpoint paths.
  - Policy rollout summaries now log fall rate directly and distinguish termination from time-limit truncation.
  - Rollout comparison export now prefers `fall_rate_mean` from summary JSON when available.

## Next Action

Do not launch larger PWM or Flow sweeps yet. The next experiment should improve or debug BC/IL so expert-filtered BC preserves more collector behavior. Success requires real MJLab eval plus rollout MP4/W&B video that beats the current expert-filtered BC baseline, not just lower BC loss.
