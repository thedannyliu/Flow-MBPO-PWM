# Progress Log (Development)

Purpose: Chronicle day-to-day development progress, decisions, fixes, blockers, and next actions for Flow-MBPO/PWM. Use this file to keep a running timeline of what changed and why. Keep entries concise, in English, and append-only (newest at top).

How to update:
- Add a dated heading for each work session.
- Bullet key changes, issues found, fixes applied, TODOs carried forward.
- Link to relevant PRs/commits, config files, scripts, and experiment entries in `docs/experiment_log.md`.
- Note any unresolved problems explicitly so they are not forgotten.

Template for each entry:
```
## YYYY-MM-DD – Author
- Changes: ...
- Issues/risks: ...
- Follow-ups: ...
```

---

## 2025-12-07 – Copilot
- Added `wm_weight_decay` option to PWM optimizer to support Phase 1.5 regularization sweeps.
- Added structured Slurm scripts under `scripts/phase1`, `scripts/phase1p5`, and `scripts/phase2` for baseline vs Flow WM (5M/48M), Flow WM reg grid (H∈{8,16}, base/strong L2), and Flow policy runs.
- Translated remaining Chinese strings to English in `scripts/train_dflex.py` and `scripts/submit_all_verified.sh` to keep comments/logs consistent.
- Next: run Hydra/FlowActor smoke tests on the login node if feasible, then submit Phase 1 baseline vs Flow WM jobs and record in `docs/experiment_log.md`.

## 2025-12-07 – Copilot
- Added experiment operations/naming/logging guidance to `docs/master_plan.md` (section 10).
- Initialized progress log and experiment log templates to standardize record-keeping.
- Next: start scheduling Phase 1/1.5/2 runs on L40S (baseline vs Flow) using the documented Slurm headers and update `docs/experiment_log.md` per submission.
