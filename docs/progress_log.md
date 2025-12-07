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

## 2025-12-07 – Copilot (Session 4: Final Fixes and Full Deployment)
- **Fixed critical CUDA device conflicts:**
  - Issue: Multiple jobs failing with "CUDA device busy or unavailable"
  - Root cause: dflex_ant config uses 256 parallel envs (too much for L40s 48GB)
  - Solution 1: Switch to dflex_ant_l40s config (128 envs)
  - Solution 2: Add `export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` to isolate GPU access
- **Successfully deployed all 10 jobs (Phase 1/1.5/2):**
  - Jobs 2581563-2581564: Phase 1 5M baseline + Flow WM → RUNNING with wandb
  - Jobs 2581581-2581582: Phase 1 48M baseline + Flow WM → Queued
  - Jobs 2581583-2581586: Phase 1.5 H8/H16 reg sweeps → Queued
  - Jobs 2581587-2581588: Phase 2 Flow policy variants → Queued
- **Wandb tracking confirmed working:**
  - Project: `flow-mbpo-pwm` (danny010324)
  - Baseline 5M job training at 1% (122/15000 epochs)
  - Links visible in logs: https://wandb.ai/danny010324/flow-mbpo-pwm
- **Phase 2 WM/Policy matrix coverage:**
  - MLP WM + MLP policy: Phase 1 baseline ✓
  - Flow WM + MLP policy: Phase 1 Flow WM ✓
  - MLP WM + Flow policy: Phase 2 job 2581588 ✓
  - Flow WM + Flow policy: Phase 2 job 2581587 ✓
- Git commit 4957af9: CUDA fixes and env config.
- Next: Monitor jobs (ETA 8-12h for 5M, 24-30h for 48M), extract metrics, analyze Phase 1.5 results for canonical config.

## 2025-12-07 – Copilot (Session 3: Successful Training Launch)
- **Fixed critical Hydra parsing errors in wandb config overrides:**
  - Issue 1: Single `+` vs double `++` for existing config keys (wandb.notes) → fixed with sed
  - Issue 2: Equals signs in wandb field values (K=4, H=16, L2=3e-4) interpreted as nested config → removed all `=` from wandb strings
  - Issue 3: Parentheses in wandb.notes causing parse errors → removed all parentheses
- **Successfully submitted all Phase 1/1.5/2 training jobs (10 total):**
  - Phase 1: Jobs 2572352-2572355 (5M+48M baseline+Flow WM) → all TRAINING
  - Phase 1.5: Jobs 2572356-2572359 (5M Flow WM H8/H16 reg sweeps) → 1 running, 3 queued
  - Phase 2: Jobs 2572360-2572361 (5M Flow policy variants) → both queued
- **Verified training startup:**
  - Baseline (job 2572354): WorldModel 1.52M params, self-check passed, training started
  - Flow WM (job 2572355): FlowWorldModel 1.51M params, Heun K=4, self-check passed, training started
- All job logs confirm models initialize correctly, no NaN/config errors.
- Git commit 74e537f: Hydra-safe wandb fields fix.
- Next: Monitor jobs over hours/days, collect final metrics, update experiment_log.md with results.

## 2025-12-07 – Copilot (Session 2: Initial Job Submissions - FAILED)
- Fixed bash strict mode (`set -euo pipefail`) conflicts with cluster bashrc; removed from all scripts.
- Submitted smoke test (job 2571819) → succeeded, models initializing correctly.
- **First job submission attempt (jobs 2571823-2571839):**
  - All 10 jobs failed immediately due to Hydra ConfigCompositionException
  - Root cause: Used `+wandb.project` but wandb already exists in config.yaml (should use `++` for override)
- Applied sed fix: `+wandb.*` → `++wandb.*` across all scripts.
- **Second submission attempt (jobs 2571866-2571882):**
  - Some jobs still failed with "mismatched input '(' expecting <EOF>" and "mismatched input '=' expecting <EOF>"
  - Root cause: Parentheses and equals signs in wandb field values confuse Hydra parser
- Issues documented; fixes applied in Session 3.

## 2025-12-07 – Copilot (Session 1: Setup)
- Added `wm_weight_decay` option to PWM optimizer to support Phase 1.5 regularization sweeps.
- Added structured Slurm scripts under `scripts/phase1`, `scripts/phase1p5`, and `scripts/phase2` for baseline vs Flow WM (5M/48M), Flow WM reg grid (H∈{8,16}, base/strong L2), and Flow policy runs.
- Translated remaining Chinese strings to English in `scripts/train_dflex.py` and `scripts/submit_all_verified.sh` to keep comments/logs consistent.
- FlowActor smoke test on CPU passed (action shape [4,3], logstd shape [3]).

## 2025-12-07 – Copilot
- Added experiment operations/naming/logging guidance to `docs/master_plan.md` (section 10).
- Initialized progress log and experiment log templates to standardize record-keeping.
- Next: start scheduling Phase 1/1.5/2 runs on L40S (baseline vs Flow) using the documented Slurm headers and update `docs/experiment_log.md` per submission.
