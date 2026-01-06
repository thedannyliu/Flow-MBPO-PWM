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

## 2026-01-03 – Copilot (Config Fixes + Aligned Experiments V2)

### CRITICAL CONFOUND FIXED:
- **Issue**: Baseline used `rew_rms: True` but Flow variants used `rew_rms: False`.
- **Fix**: Aligned all configs to `rew_rms: True` for fair comparison.
- **Affected Configs**: `pwm_5M_flow_v3_aligned`, `pwm_5M_flowpolicy_aligned`, `pwm_48M_flow`, `pwm_5M_flow_lowLR`, `pwm_5M_flow_strongReg`.

### ALIGNED EXPERIMENTS V2 (Resubmitted):
- **Issue with V1**: Most jobs (3143565-3143644) failed with `CUDA error: busy` due to shared nodes.
- **Resolution**: Resubmitted 63 jobs with SSH `--exclusive` flag for ALL tasks (Ant, Anymal, Humanoid).
- **Queue Behavior**: Only 4 jobs are running simultaneously due to the cluster's `QOSMaxGRESPerUser` limit (max 4 GPUs). The remaining 59 jobs are pending and will start automatically as running jobs complete.
- **Health Check (Update)**: Results show **Instability/Bifurcation**:
  - Some seeds achieve SOTA (~1200) (e.g., Ant FlowPolicy s7).
  - Others collapse completely (~20) (e.g., Ant FlowPolicy s2).
  - This affects both Baseline and Flow runs, likely due to `rew_rms: True` scaling issues.
- **Dev Fixes**:
  - Fixed `eval_pwm.py` crash by auto-detecting `FlowActor` from checkpoint weights (bypassed config mismatch).
  - Fixed metadata bug where Baseline was mislabeled as FlowPolicy due to package name substring match.

- **Hardware Failure**: Node **atl1-1-03-004** identified as defective (ECC errors). All jobs on this node fail evaluation. Marked as "Do Not Use".
- **Hung Job**: Ant Flow s5 (3143925) hung for >34h. Canceled.
- **Recovery (V5)**: Resubmitted 9 seeds (Failed OOM + Hung Job) in `submit_resubmit_v5.py` (Batches 3162639-41) with safe 3 jobs/node density.

#### Current Job Status:
- **Completed (Mixed)**: Ant Aligned s0-7 (16 jobs). Evaluation confirms bifurcation.
- **Pending (Queue)**: 
  - 12 Packed V4 Batches (Running/Pending).
  - 3 Packed V5 Batches (Pending).

#### WandB Projects:
- `flow-mbpo-aligned-ant`, `flow-mbpo-aligned-anymal`, `flow-mbpo-aligned-humanoid`
- `flow-mbpo-scaling`, `flow-mbpo-tuning`

---

## 2025-12-31 / 2026-01-01 – Copilot (Humanoid Experiments + Evaluation Pipeline)

### HUMANOID EXPERIMENTS STATUS:
- **COMPLETED (11 jobs):**
  - Baseline: 3 jobs (3101831-3101833) - ~2h20m each ✅
  - Flow WM K=4: 3 jobs (3104842-3104845) - ~5h each ✅
  - Flow WM K=2: 3 jobs (3104846-3104848) - ~3h40m each ✅
  - Flow WM K=8: 2 jobs (3107946-3107947) - ~5-6h each ✅
- **STILL FAILING (7 jobs):**
  - Flow WM K=8 s456, FlowPolicy×3, FullFlow×3
  - Root cause: CUDA device busy errors (GPU contention on cluster)

### EVALUATION PIPELINE DEVELOPMENT:
1. **Created `evaluate_unified.py`:**
   - Supports all 3 envs: dflex_ant, dflex_anymal, dflex_humanoid
   - Uses REAL environment rewards (not WM predictions)
   - Added module aliasing for backward compat with 'pwm' checkpoints
2. **Disk Quota Issue:**
   - Project directory 100% full (~21GB in outputs)
   - Deleted wandb/ (3.2GB) and outputs/2025-12-27/ (3.7GB)
   - WORKAROUND: Using scratch directory `/storage/scratch1/9/eliu354/flow_mbpo/`
3. **Batch Eval Script:**
   - Created but jobs failing due to disk quota
   - Will retry after using scratch directory

### DISK SPACE CLEANUP:
- Removed `scripts/wandb/` (3.2GB) - already synced to cloud
- Removed `scripts/outputs/2025-12-27/` (3.7GB) - old smoke tests
- Kept Dec 28-30 outputs (checkpoints needed for eval)

### SCRATCH DIRECTORY SETUP:
- Created: `/storage/scratch1/9/eliu354/flow_mbpo/`
  - `scripts/` - evaluation scripts
  - `eval_results/` - CSV output
  - `logs/` - SLURM logs

### GIT COMMITS: TBD
### NEXT STEPS:
1. Resubmit remaining 7 Humanoid jobs with staggered timing
2. Run batch evaluation using scratch directory
3. Aggregate results into CSV
4. Complete documentation updates

---

## 2025-12-29 – Copilot (Ant Experiments Setup)
- **ANT EXPERIMENTS LAUNCHED:**
  - Created `scripts/ant/` directory with smoke tests and full training scripts
  - Smoke tests: baseline, flowWM, flowpolicy, fullflow (jobs 3093139-3093142)
  - Full training: 18 jobs submitted (3093165-3093182)
  - Variants: Baseline, Flow WM (K=2,4,8), Flow Policy, Full Flow - 3 seeds each
  - WandB project: `flow-mbpo-single-task-ant`
- **FIXED STRONGREG CONFIG:**
  - Removed unsupported `wm_weight_decay` parameter
  - Changed to tighter gradient clipping for regularization
  - Resubmitted jobs 3093114-3093116
- **GIT COMMITS:** TBD
- **NEXT:** Monitor job status, verify smoke tests complete successfully

---

## 2025-12-26 – Copilot (Baseline Restoration + Flow-MBPO Alignment)
- **BASELINE RESTORED TO ORIGINAL VALUES:**
  - `pwm_5M_baseline_final.yaml`: wm_batch_size=256, wm_buffer_size=1M, planning=False
  - `dflex_ant.yaml`: num_envs=128
  - Maximum epochs: 15,000 (matching original)
- **FLOW-MBPO CONFIGS ALIGNED WITH BASELINE:**
  - All flow configs (v1, v2, v3) now use same base params as baseline
  - Only flow-specific params differ: use_flow_dynamics, flow_integrator, flow_substeps
  - units=[512,512] aligned with baseline (was [510,510])
- **BASELINE IMMUTABILITY POLICY DOCUMENTED:**
  - Added to `master_plan.md` Section 2 Core Invariants
  - baselines/ added to .gitignore (local reference only)
- **GIT COMMIT:** `784b32f`
- **NEXT:** Push to GitHub, run comparison experiments

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
