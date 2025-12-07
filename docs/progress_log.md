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

## 2025-12-07 – Copilot (Session 2: Job Submissions)
- Fixed bash strict mode (`set -euo pipefail`) conflicts with cluster bashrc; removed from all scripts.
- Submitted smoke test (job 2571819) → running successfully, models initializing correctly.
- **Submitted Phase 1 jobs (baseline vs Flow WM):**
  - 2571823: 5M baseline H=16
  - 2571824: 5M Flow WM v2 H=16 K=4
  - 2571831: 48M baseline H=16
  - 2571832: 48M Flow WM v2 H=16 K=4
- **Submitted Phase 1.5 jobs (Flow WM reg sweeps):**
  - 2571834: 5M Flow v2 H=8 base reg
  - 2571835: 5M Flow v2 H=8 L2=3e-4
  - 2571836: 5M Flow v2 H=16 base reg
  - 2571837: 5M Flow v2 H=16 L2=3e-4
- **Submitted Phase 2 jobs (Flow policy):**
  - 2571838: 5M Flow WM + Flow policy H=8 K=4 Kpol=2
  - 2571839: 5M MLP WM + Flow policy H=8 Kpol=2
- All jobs recorded in `docs/experiment_log.md` with job IDs, scripts, W&B groups, and log paths.
- Next: monitor jobs, record metrics when complete, identify canonical Flow WM config from Phase 1.5 results.

## 2025-12-07 – Copilot (Session 1: Setup)
- Added `wm_weight_decay` option to PWM optimizer to support Phase 1.5 regularization sweeps.
- Added structured Slurm scripts under `scripts/phase1`, `scripts/phase1p5`, and `scripts/phase2` for baseline vs Flow WM (5M/48M), Flow WM reg grid (H∈{8,16}, base/strong L2), and Flow policy runs.
- Translated remaining Chinese strings to English in `scripts/train_dflex.py` and `scripts/submit_all_verified.sh` to keep comments/logs consistent.
- FlowActor smoke test on CPU passed (action shape [4,3], logstd shape [3]).

## 2025-12-07 – Copilot
- Added experiment operations/naming/logging guidance to `docs/master_plan.md` (section 10).
- Initialized progress log and experiment log templates to standardize record-keeping.
- Next: start scheduling Phase 1/1.5/2 runs on L40S (baseline vs Flow) using the documented Slurm headers and update `docs/experiment_log.md` per submission.
