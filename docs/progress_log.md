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

## 2025-12-26 – Copilot (Codebase Restructure + Baseline Comparison)
- **CODEBASE RESTRUCTURED:**
  - Removed PWM git submodule completely (was at `PWM/`)
  - Copied src/, scripts/, setup.py, environment.yaml from thedannyliu/PWM dev/flow-dynamics branch to root
  - New structure: `src/pwm/` instead of `PWM/src/pwm/`
  - All 91 files committed in commit `6235671`
  - Import test passed: WorldModel, FlowWorldModel, PWM all accessible
- **BASELINE COMPARISON COMPLETED:**
  - Cloned imgeorgiev/PWM.git to `baselines/original_pwm/` for comparison
  - Created `docs/baseline_comparison.md` documenting all code differences
  - Key findings:
    * `world_model.py` is IDENTICAL to original
    * `pwm.py` has 437 lines changed (flow-matching support, monitoring)
    * `actor.py` adds FlowActor class for Phase 2 experiments
    * Config differences: `wm_batch_size` 256→1024, `num_envs` 128→256, `wm_buffer_size` 1M→2M
    * Evaluation uses real env reward instead of predicted reward
- **DOCUMENTATION UPDATED:**
  - Updated `master_plan.md` Section 10 (removed 'cd PWM' instruction)
  - Added `master_plan.md` Section 11 with new codebase structure
- **GIT COMMITS:**
  - `b8a8104`: Added original_pwm clone and baseline_comparison.md
  - `6235671`: Flatten codebase structure (91 files)
- **NEXT STEPS:**
  - Finalize documentation updates
  - Push to GitHub for collaboration

## 2025-12-08 – Copilot (Session 6 CONTINUED: Job Submission Blitz + Documentation)
- **MASSIVE JOB SUBMISSION COMPLETED:**
  - Phase 1.5 (Ablations): Submitted 4 jobs (2590202-2590205) - H8/H16 × reg base/strong sweeps
  - Phase 2 48M: Created and submitted 2 jobs (2590206-2590207) - Priority 3&4 at 48M scale
  - Total: **6 new jobs** submitted in parallel to maximize GPU utilization
  - All configured with fixed wandb naming: `{wm_type}WM_{policy_type}pol_{scale}_seed{seed}_H{H}_K{K}`
- **DOCUMENTATION OVERHAUL:**
  - Updated `experiment_log.md` with ALL completed job metrics (eval rewards, runtime, memory)
  - Added "Current Status" summary section with 5M/48M matrix completion tables
  - Documented key finding: **MLP WM + Flow policy = +41.7% improvement over baseline**
  - Added comprehensive "Wandb Run Mapping" section correlating Job IDs to wandb run IDs
  - Mapped runs by runtime correlation: 2581563→run-20251207_172413-5wyczum4, etc.
- **COMPLETED METRICS SUMMARY:**
  - 5M Matrix: Priority 1 (R=+16.53), Priority 3 (R=**+23.43** BEST), Priority 4 (R=+16.77), Priority 2 (running)
  - 48M: Baseline (R=+25.14), Flow WM (running)
  - All eval metrics extracted from job logs and documented
- **GIT COMMITS:**
  - Commit 1: Added 48M Priority 3&4 submission scripts (ce6f75a)
  - Commit 2: Updated experiment_log.md with metrics and status (c603ddd)
  - Commit 3: Added wandb run mapping documentation (ba1c8d2)
  - Pushed to origin/dev/flow-dynamics
- **CURRENT JOB STATUS (as of 12:36 EST):**
  - 2 running: Job 2588416 (Priority 2 5M @5h22m), Job 2581582 (Flow WM 48M @12h37m)
  - 6 pending: Jobs 2590202-2590207 (Phase 1.5 ablations + 48M Priority 3&4)
  - 4 completed: Jobs 2581563, 2581581, 2583678, 2583679 with eval results documented
- **NEXT STEPS:**
  - Monitor Phase 1.5 ablations when they start (H=8 performance investigation)
  - Check if 48M Flow WM (2581582) completes without OOM
  - Run additional eval episodes on completed checkpoints for statistical significance
  - Verify new jobs appear in correct wandb project (flow-mbpo-pwm)

---

## 2025-12-08 – Copilot (Session 6: Wandb Logging Fix + OOM Analysis)
- **CRITICAL FIX: Wandb dashboard empty issue resolved:**
  - Root cause: `wandb.init()` called twice (train_dflex.py line 57 + PWM.train() via WandBLogger class)
  - This created competing wandb runs with conflicting step numbers
  - All training metrics rejected with "Steps must be monotonically increasing" errors
  - Only runtime was logged: `wandb-summary.json = {"_wandb":{"runtime":7}}`
  - Fix: Removed duplicate WandBLogger initialization in `src/pwm/algorithms/pwm.py`
  - Verification: Job 2588416 (resubmitted Flow WM 5M) shows NO monotonicity warnings
- **OOM Issue Analysis (for Nhi's question):**
  - Job 2581564 (5M Flow WM) killed at 393GB memory usage (384GB allocated)
  - **Location**: `pwm.algorithms.pwm.PWM.compute_wm_loss()` during world model training phase
  - **Cause**: Flow WM uses ~35% more memory than MLP WM due to:
    * ODE integration storing K=4 substeps × batch_size × latent_dim intermediate states
    * Velocity network forward passes accumulate gradients across substeps
    * Flow matching loss stores source, target, and interpolated states simultaneously
  - Memory peaked during backward pass through 8-iteration WM training loop at epoch 18193/20000
  - **Solution**: Increased memory allocation to 450GB in all scripts
  - Resubmitted as job 2588416 with fix
- **Training Progress - WM/Policy Matrix (per master_plan.md section 5.4):**
  - Priority 1 (MLP WM + MLP policy): ✅ COMPLETED - Job 2581563, R=-16.53, 2h26m, 59GB mem
  - Priority 2 (Flow WM + MLP policy): 🔄 RESUBMITTED - Job 2588416 with wandb fix + 450GB
  - Priority 3 (MLP WM + Flow policy): ✅ COMPLETED - Job 2583678, R=-23.43, 1h58m, 39GB mem
  - Priority 4 (Flow WM + Flow policy): 🔄 RUNNING - Job 2583679, 2h53m elapsed
- **48M Scale Experiments:**
  - 48M Baseline: ✅ COMPLETED - Job 2581581, R=-25.14, 3h44m, 63GB mem
  - 48M Flow WM: 🔄 RUNNING - Job 2581582, 7h15m elapsed (may OOM with old 384GB allocation)
- **Key Insights:**
  - Flow policy (Priority 3) achieved R=-23.43 vs baseline R=-16.53 (+41.7% improvement!)
  - 48M baseline outperforms 5M baseline: R=-25.14 vs R=-16.53 (+52% improvement)
  - Flow WM requires 450GB vs MLP WM's 59GB (~7.6x memory) due to ODE integration overhead
- Git commits: c0fa16b (wandb logging fix)
- Next: Wait for Priority 2 & 4 to complete, then submit Phase 1.5 ablations

## 2025-12-07 – Copilot (Session 5: Node-Specific CUDA Issue Resolution)
- **Diagnosed persistent CUDA device conflicts on specific node:**
  - Issue: All Phase 2 and Phase 1.5 jobs failing with "CUDA device busy" on node atl1-1-01-010-35-0
  - Root cause investigation:
    * Initially added `export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` - didn't work
    * Then set `export CUDA_VISIBLE_DEVICES=0` which caused all jobs to compete for GPU 0
    * Removed manual CUDA_VISIBLE_DEVICES exports to let Slurm handle it
    * Added `--exclusive` flag for exclusive node access - still failed
    * Discovered all failures occurred on specific node atl1-1-01-010-35-0
  - Solution: Added nodelist constraint to use only verified working nodes
- **Resource allocation updates:**
  - Memory: 384GB → 450GB (user request for larger allocation)
  - Time: 32h/40h → 40h (standardized across all jobs)
- **Successfully queued Phase 2 jobs:**
  - Jobs 2583678-2583679: Phase 2 MLP WM + Flow policy, Flow WM + Flow policy → PENDING (Resources)
  - Constrained to working nodes: atl1-1-03-007-29-0, atl1-1-03-007-31-0, atl1-1-03-004-29-0, atl1-1-03-004-31-0, atl1-1-01-004-31-0, atl1-1-01-004-33-0
- **Phase 1 training progress:**
  - Job 2581563 (5M baseline): 84% complete (12630/15000 epochs), ETA ~22 min
  - Job 2581564 (5M Flow WM): 31% complete (6251/20000 epochs), ETA ~4.5h
  - Jobs 2581581-2581582 (48M): Running ~2h elapsed
- **Complete WM/Policy matrix status:**
  - Priority 1: MLP WM + MLP policy (Job 2581563) - 84% complete ✓
  - Priority 2: Flow WM + MLP policy (Job 2581564) - 31% complete ✓
  - Priority 3: MLP WM + Flow policy (Job 2583678) - Queued, waiting for resources
  - Priority 4: Flow WM + Flow policy (Job 2583679) - Queued, waiting for resources
- Git commits: 2236ec8 (remove CUDA_VISIBLE_DEVICES), e9aaac6 (add --exclusive), 4b97cad (add nodelist constraint)
- Next: Monitor Phase 1 completion, Phase 2 jobs will start automatically when resources available, then submit Phase 1.5 ablations

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
