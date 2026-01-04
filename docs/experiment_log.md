# Experiment Log

> **Purpose**: This file serves as a **persistent experiment registry** for all training, evaluation, and analysis jobs.
> - **Never delete old entries** - append new entries at the top (newest first)
> - Each entry tracks: Job ID, status, configuration, metrics, checkpoint paths, and WandB links
> - Status lifecycle: `PENDING` → `QUEUED` → `RUNNING` → `COMPLETED` / `FAILED` → `EVALUATED`
>
> **How to use this file**:
> 1. When submitting a job, add an entry with status `QUEUED` and job details
> 2. Update status to `RUNNING` when the job starts
> 3. Update to `COMPLETED`/`FAILED` with runtime and checkpoint path when finished
> 4. After evaluation, update to `EVALUATED` with final metrics
>
> **Important fields to track**:
> - `Job ID`: Slurm job ID for tracking
> - `Config`: Algorithm config file used (e.g., `pwm_48M_mt_baseline.yaml`)
> - `Task`: Task name (e.g., `reacher-easy`, `walker-stand`)
> - `Seed`: Random seed for reproducibility
> - `Checkpoint`: Path to saved model checkpoint
> - `WandB`: Link to WandB run for visualization
> - `Notes`: Any important observations or issues

---

## Active / Recent Experiments

### Attempt 8: 48M Multitask MT30 (Fixed Logging)
- **Date**: 2026-01-03
- **Job IDs**: 
  - `4011713` (Baseline, H100)
  - `4011714` (Flow Policy, H200)
- **Status**: PARTIAL FAILURE
- **Outcome**: 
  - **Baseline (4011713)**: Index 0 Success (`cheetah-run` s42). Index 1 Running. Others Pending.
  - **Flow Policy (4011714)**: 
    - Indices 0, 1, 2, 4 Running/Success.
### Attempt 10: 48M MT30 Final Status
**All Phase 3 experiments are COMPLETED.** Weights have been cleaned up to save space.

### Final Results Summary (Phase 3)
*Deduplicated from 18 valid runs (Completed Jan 04)*

| Task | Metric | Baseline (MLP) | Flow Policy (ODE) | Winner |
|---|---|---|---|---|
| **reacher-easy** | Reward | **982.30** | 980.67 | Tie (Solved) |
| **walker-stand** | Reward | **957.72** | 839.78 | Baseline (+14%) |
| **cheetah-run** | Reward | **112.48** | 98.74 | Tie (Both Failed) |

### Detailed Seed Metrics
| Algo | Task | Seed | R | Plan R | Job ID |
|---|---|---|---|---|---|
| **Baseline** | `cheetah-run` | 42 | 93.69 | 88.13 | 4011771 |
| **Baseline** | `cheetah-run` | 123 | 108.80 | 81.67 | Attempt 8 (4011772) |
| **Baseline** | `cheetah-run` | 456 | 134.97 | 114.09 | 4011713 |
| **Baseline** | `reacher-easy` | 42 | 981.20 | 981.70 | 4011715 |
| **Baseline** | `reacher-easy` | 123 | 983.50 | 985.60 | 4011737 |
| **Baseline** | `reacher-easy` | 456 | 982.20 | 985.50 | 4011758 |
| **Baseline** | `walker-stand` | 42 | 972.32 | 943.97 | 4011759 |
| **Baseline** | `walker-stand` | 123 | 923.48 | 933.11 | 4011760 |
| **Baseline** | `walker-stand` | 456 | 977.35 | 978.33 | 4011761 |
| **Flow** | `cheetah-run` | 42 | 80.97 | 82.31 | 4011744 |
| **Flow** | `cheetah-run` | 123 | 94.75 | 73.21 | Attempt 8 (4011794) |
| **Flow** | `cheetah-run` | 456 | 120.52 | 110.27 | 4011740 |
| **Flow** | `reacher-easy` | 42 | 976.70 | 982.90 | 4011716 |
| **Flow** | `reacher-easy` | 123 | 983.40 | 986.00 | 4011717 |
| **Flow** | `reacher-easy` | 456 | 981.90 | 983.90 | 4011718 |
| **Flow** | `walker-stand` | 42 | 854.53 | 864.39 | 4011741 |
| **Flow** | `walker-stand` | 123 | 744.92 | 796.67 | 4011720 |
| **Flow** | `walker-stand` | 456 | 919.90 | 933.66 | 4011743 |



### Attempt 7: 48M Multitask MT30 (Collision Issue)
- **Date**: 2026-01-03
- **Job IDs**: 
  - `4011522` (Baseline, H100)
  - `4011523` (Flow Policy, H200) - Job `_5` failed (OOM/Bad Node), re-run as `4011712`.
- **Status**: COMPLETED (PARTIAL LOGS)
- **Outcome**: 
  - Jobs completed successfully in ~27 mins.
  - **Issue**: Default `hydra.run.dir` (1-second precision) caused output directory collisions for array jobs starting simultaneously. Only the last writing seed per task survived in the logs.
  - **Recovered Metrics**:
    - **Baseline (s456)**: `cheetah-run` R=107.15, PlanR=105.16
    - **Baseline (s123)**: `cheetah-run` R=122.84, PlanR=132.98
    - **Flow Policy (s123)**: `cheetah-run` R=83.38, PlanR=101.21
    - **Flow Policy (s42)**: `walker-stand` R=901.05, PlanR=887.53 (Success!)

**Baseline Array Job 4011522** (9 jobs: 3 tasks × 3 seeds)
- Logs polluted by array job collision. Re-running as Attempt 8 (`4011713`).

**Flow Policy Array Job 4011523** (9 jobs: 3 tasks × 3 seeds)
- Logs polluted by array job collision. Job `_5` failed (OOM). Re-running as Attempt 8 (`4011714`).

> [!IMPORTANT]
> **Ghost Runs**: Runs named `PWM_reacher-easy`, `PWM_walker-stand`, etc., are artifacts from Attempt 3. **Please ignore them.** Valid Attempt 5 runs use `baseline_H100_` or `flowpolicy_H200_`.

---

### Previous Attempts (Archived)
<details>
<summary>View Attempt 4 (Stopped due to missing metadata)</summary>
- **Status**: 🛑 **CANCELLED**
- **Reason**: Missed global WandB metadata fix. Transitioned to Attempt 5.
</details>

---

### MT30 Full Training - Attempt 3 (2026-01-03 18:30 EST)
- **Status**: 🛑 **CANCELLED**
- **Reason**: WandB run names were unclear (ignored CLI overrides). Fixed in `Attempt 4`.

**Baseline Array Job 4011428** (9 jobs: 3 tasks × 3 seeds)
| Array ID | Task | Seed | Job ID | Status | Hardware | Output Path |
|----------|------|------|--------|--------|----------|-------------|
| 0 | reacher-easy | 42 | 4011428_0 | � CANCELLED | H100 | `outputs/mt30/baseline/reacher-easy/seed42` |
| 1 | reacher-easy | 123 | 4011428_1 | � CANCELLED | H100 | `outputs/mt30/baseline/reacher-easy/seed123` |
| 2 | reacher-easy | 456 | 4011428_2 | � CANCELLED | H100 | `outputs/mt30/baseline/reacher-easy/seed456` |
| 3 | walker-stand | 42 | 4011428_3 | � CANCELLED | H100 | `outputs/mt30/baseline/walker-stand/seed42` |
| 4 | walker-stand | 123 | 4011428_4 | � CANCELLED | H100 | `outputs/mt30/baseline/walker-stand/seed123` |
| 5 | walker-stand | 456 | 4011428_5 | � CANCELLED | H100 | `outputs/mt30/baseline/walker-stand/seed456` |
| 6 | cheetah-run | 42 | 4011428_6 | � CANCELLED | H100 | `outputs/mt30/baseline/cheetah-run/seed42` |
| 7 | cheetah-run | 123 | 4011428_7 | � CANCELLED | H100 | `outputs/mt30/baseline/cheetah-run/seed123` |
| 8 | cheetah-run | 456 | 4011428_8 | � CANCELLED | H100 | `outputs/mt30/baseline/cheetah-run/seed456` |

**Flow Policy Array Job 4011429** (9 jobs: 3 tasks × 3 seeds)
| Array ID | Task | Seed | Job ID | Status | Hardware | Output Path |
|----------|------|------|--------|--------|----------|-------------|
| 0 | reacher-easy | 42 | 4011429_0 | 🛑 CANCELLED | H100 | `outputs/mt30/flowpolicy/reacher-easy/seed42` |
| 1 | reacher-easy | 123 | 4011429_1 | 🛑 CANCELLED | H100 | `outputs/mt30/flowpolicy/reacher-easy/seed123` |
| 2 | reacher-easy | 456 | 4011429_2 | 🛑 CANCELLED | H100 | `outputs/mt30/flowpolicy/reacher-easy/seed456` |
| 3 | walker-stand | 42 | 4011429_3 | 🛑 CANCELLED | H100 | `outputs/mt30/flowpolicy/walker-stand/seed42` |
| 4 | walker-stand | 123 | 4011429_4 | 🛑 CANCELLED | H100 | `outputs/mt30/flowpolicy/walker-stand/seed123` |
| 5 | walker-stand | 456 | 4011429_5 | 🛑 CANCELLED | H100 | `outputs/mt30/flowpolicy/walker-stand/seed456` |
| 6 | cheetah-run | 42 | 4011429_6 | 🛑 CANCELLED | H100 | `outputs/mt30/flowpolicy/cheetah-run/seed42` |
| 7 | cheetah-run | 123 | 4011429_7 | 🛑 CANCELLED | H100 | `outputs/mt30/flowpolicy/cheetah-run/seed123` |
| 8 | cheetah-run | 456 | 4011429_8 | 🛑 CANCELLED | H100 | `outputs/mt30/flowpolicy/cheetah-run/seed456` |

---

#### Attempt 12: Cheetah Debug (Jan 04 - Revised)
- **Job ID**: `4011968` (Array 0-2)
- **Status**: ⏳ PENDING
- **Config**: `pwm_48M_mt_fullflow` + `horizon=30`
- **WandB**: `MT30-Detailed` / `mt30_debug_h30`
- **Notes**: Added `model_best` saving and resume support.

#### Attempt 11: Full Flow Model (Jan 04 - Revised)
- **Job ID**: `4011967` (Array 0-8)
- **Status**: ⏳ PENDING
- **Config**: `pwm_48M_mt_fullflow` (Flow WM + Flow Policy)
- **WandB**: `MT30-Detailed` / `mt30_fullflow`
- **Notes**: From scratch training. Added `model_best` saving and resume support.

#### Phase 3: MT30 Multitask Comparison (Completed)
**Goal**: Compare Baseline vs Flow Policy on 3 tasks (3 seeds each).
**Status**: ✅ COMPLETED (Attempt 7-9)

### MT30 Full Training - Attempt 2 (2026-01-03 18:15 EST)
- **Status**:  **CANCELLED**
- **Reason**: User requested change of WandB project name and more detailed run tracking.
- **Goal**: Complete Baseline (MLP WM + MLP Policy) vs Flow Policy (MLP WM + Flow Policy) comparison.

**Baseline Array Job 4011379** (9 jobs: 3 tasks × 3 seeds)
| Array ID | Task | Seed | Job ID | Status | WandB Name |
|----------|------|------|--------|--------|------------|
| 2 | reacher-easy | 456 | 4011379_2 | � CANCELLED | baseline_reacher-easy_s456 |
| 3 | walker-stand | 42 | 4011379_3 | � CANCELLED | baseline_walker-stand_s42 |
| 4 | walker-stand | 123 | 4011379_4 | � CANCELLED | baseline_walker-stand_s123 |
| 5 | walker-stand | 456 | 4011379_5 | � CANCELLED | baseline_walker-stand_s456 |
| 6 | cheetah-run | 42 | 4011379_6 | � CANCELLED | baseline_cheetah-run_s42 |
| 7 | cheetah-run | 123 | 4011379_7 | � CANCELLED | baseline_cheetah-run_s123 |
| 8 | cheetah-run | 456 | 4011379_8 | � CANCELLED | baseline_cheetah-run_s456 |

**Flow Policy Array Job 4011380** (9 jobs: 3 tasks × 3 seeds)
| Array ID | Task | Seed | Job ID | Status | WandB Name |
|----------|------|------|--------|--------|------------|
| 0 | reacher-easy | 42 | 4011380_0 | 🛑 CANCELLED | flowpolicy_reacher-easy_s42 |
| 1 | reacher-easy | 123 | 4011380_1 | 🛑 CANCELLED | flowpolicy_reacher-easy_s123 |
| 2 | reacher-easy | 456 | 4011380_2 | 🛑 CANCELLED | flowpolicy_reacher-easy_s456 |
| 3 | walker-stand | 42 | 4011380_3 | 🛑 CANCELLED | flowpolicy_walker-stand_s42 |
| 4 | walker-stand | 123 | 4011380_4 | 🛑 CANCELLED | flowpolicy_walker-stand_s123 |
| 5 | walker-stand | 456 | 4011380_5 | 🛑 CANCELLED | flowpolicy_walker-stand_s456 |
| 6 | cheetah-run | 42 | 4011380_6 | 🛑 CANCELLED | flowpolicy_cheetah-run_s42 |
| 7 | cheetah-run | 123 | 4011380_7 | 🛑 CANCELLED | flowpolicy_cheetah-run_s123 |
| 8 | cheetah-run | 456 | 4011380_8 | 🛑 CANCELLED | flowpolicy_cheetah-run_s456 |

---

### MT30 Full Training - Attempt 1 (2026-01-03 04:37 EST)
- **Status**: ❌ **FAILED**
- **Reason**: `omegaconf.errors.ConfigKeyError: Key 'notes' is not in struct`. The `create_wandb_run` function forced notes but they weren't provided in the config/overrides.
- **Fix**: Modified `scripts/train_multitask.py` to use `wandb_cfg.get("notes", None)` to handle missing notes gracefully.

**Baseline Array Job 4010895** (9 jobs: 3 tasks × 3 seeds)
| Array ID | Task | Seed | Job ID | Status | WandB Name |
|----------|------|------|--------|--------|------------|
| 0 | reacher-easy | 42 | 4010895_0 | ❌ FAILED | baseline_reacher-easy_s42 |
| 1 | reacher-easy | 123 | 4010895_1 | ❌ FAILED | baseline_reacher-easy_s123 |
| 2 | reacher-easy | 456 | 4010895_2 | ❌ FAILED | baseline_reacher-easy_s456 |
| 3 | walker-stand | 42 | 4010895_3 | ❌ FAILED | baseline_walker-stand_s42 |
| 4 | walker-stand | 123 | 4010895_4 | ❌ FAILED | baseline_walker-stand_s123 |
| 5 | walker-stand | 456 | 4010895_5 | ❌ FAILED | baseline_walker-stand_s456 |
| 6 | cheetah-run | 42 | 4010895_6 | ❌ FAILED | baseline_cheetah-run_s42 |
| 7 | cheetah-run | 123 | 4010895_7 | ❌ FAILED | baseline_cheetah-run_s123 |
| 8 | cheetah-run | 456 | 4010895_8 | ❌ FAILED | baseline_cheetah-run_s456 |

**Flow Policy Array Job 4010896** (9 jobs: 3 tasks × 3 seeds)
| Array ID | Task | Seed | Job ID | Status | WandB Name |
|----------|------|------|--------|--------|------------|
| 0 | reacher-easy | 42 | 4010896_0 | ❌ FAILED | flowpolicy_reacher-easy_s42 |
| 1 | reacher-easy | 123 | 4010896_1 | ❌ FAILED | flowpolicy_reacher-easy_s123 |
| 2 | reacher-easy | 456 | 4010896_2 | ❌ FAILED | flowpolicy_reacher-easy_s456 |
| 3 | walker-stand | 42 | 4010896_3 | ❌ FAILED | flowpolicy_walker-stand_s42 |
| 4 | walker-stand | 123 | 4010896_4 | ❌ FAILED | flowpolicy_walker-stand_s123 |
| 5 | walker-stand | 456 | 4010896_5 | ❌ FAILED | flowpolicy_walker-stand_s456 |
| 6 | cheetah-run | 42 | 4010896_6 | ❌ FAILED | flowpolicy_cheetah-run_s42 |
| 7 | cheetah-run | 123 | 4010896_7 | ❌ FAILED | flowpolicy_cheetah-run_s123 |
| 8 | cheetah-run | 456 | 4010896_8 | ❌ FAILED | flowpolicy_cheetah-run_s456 |

**Cluster Resources**: ice-gpu partition, H100 GPUs, 450GB memory, 16h time limit

---

### MT30 Validation Test (2026-01-03 04:34 EST)

| Job ID | Config | Task | Seed | Status | Runtime | Notes |
|--------|--------|------|------|--------|---------|-------|
| 4010894 | pwm_48M_mt_baseline | walker-stand | 42 | ✅ COMPLETED | 02:34 | Validation test: R=356.81, Planning R=309.42 |

---

### MT30 Smoke Test (2026-01-03)

| Job ID | Config | Task | Seed | Status | Runtime | Notes |
|--------|--------|------|------|--------|---------|-------|
| 4010862 | pwm_48M_mt_baseline | reacher-easy | 42 | ✅ **COMPLETED** | 02:16 | 🎉 **SUCCESS!** R=1000.00, Planning R=993.00, 24000 eps loaded |
| 4010834 | pwm_48M_mt_baseline | reacher-easy | 42 | ❌ FAILED | 01:30 | Crashed on save: missing `best_policy_loss` attr. Training worked (R=978) |
| 4010832 | pwm_48M_mt_baseline | reacher-easy | 42 | ❌ FAILED | 00:14 | Buffer path error: `pwm.utils.buffer` → fixed to `flow_mbpo_pwm` |

**Environment Setup Jobs (2026-01-03)**:
| Job ID | Purpose | Status |
|--------|---------|--------|
| 4010830 | Install pandas | ✅ COMPLETED |
| 4010828 | Install hydra, omegaconf, tensordict, torchrl, wandb | ✅ COMPLETED |
| 4010820 | Install lxml for dm_control | ✅ COMPLETED |
| 4010818 | Install dm_env in flow-mbpo env | ✅ COMPLETED |

---

## Experiment Matrix - MT30 Multitask

### WandB Project: `flow-mbpo-multitask`

### Baseline (MLP WM + MLP Policy) - Array Job 4010895
| Task | Seed 42 | Seed 123 | Seed 456 |
|------|---------|----------|----------|
| reacher-easy | 🔄 4010895_0 | 🔄 4010895_1 | 🔄 4010895_2 |
| walker-stand | 🔄 4010895_3 | 🔄 4010895_4 | 🔄 4010895_5 |
| cheetah-run | 🔄 4010895_6 | 🔄 4010895_7 | ⏳ 4010895_8 |

### Flow Policy (MLP WM + Flow Policy) - Array Job 4010896
| Task | Seed 42 | Seed 123 | Seed 456 |
|------|---------|----------|----------|
| reacher-easy | ⏳ 4010896_0 | ⏳ 4010896_1 | ⏳ 4010896_2 |
| walker-stand | ⏳ 4010896_3 | ⏳ 4010896_4 | ⏳ 4010896_5 |
| cheetah-run | ⏳ 4010896_6 | ⏳ 4010896_7 | ⏳ 4010896_8 |

### Full Flow (Flow WM + Flow Policy) - Not yet submitted
| Task | Seed 42 | Seed 123 | Seed 456 |
|------|---------|----------|----------|
| reacher-easy | ⏳ Pending | ⏳ Pending | ⏳ Pending |
| walker-stand | ⏳ Pending | ⏳ Pending | ⏳ Pending |
| cheetah-run | ⏳ Pending | ⏳ Pending | ⏳ Pending |

---

## Configuration Files

| Config | World Model | Policy | Description |
|--------|-------------|--------|-------------|
| `pwm_48M_mt_baseline.yaml` | MLP | MLP | Original PWM baseline |
| `pwm_48M_mt_flowpolicy.yaml` | MLP | Flow ODE | Flow-based policy only |
| `pwm_48M_mt_fullflow.yaml` | Flow | Flow ODE | Full flow (WM + Policy) |

---

## Archived Experiments

### Single-Task ANT Experiments (Dec 2025)

<details>
<summary>Click to expand ANT experiments</summary>

#### WandB: flow-mbpo-single-task-ant

| Variant | Job IDs | Seeds | Status |
|---------|---------|-------|--------|
| Baseline | 3093165-3093167 | 42,123,456 | ✅ COMPLETED |
| Flow WM K=4 Heun | 3093168-3093170 | 42,123,456 | ✅ COMPLETED |
| Flow WM K=2 Heun | 3093171-3093173 | 42,123,456 | ✅ COMPLETED |
| Flow WM K=8 Euler | 3093174-3093176 | 42,123,456 | ✅ COMPLETED |
| Flow Policy | 3093177-3093179 | 42,123,456 | ✅ COMPLETED |
| Full Flow | 3093180, 3099532, 3099534 | 42,123,456 | ✅ COMPLETED |

</details>

### Single-Task ANYMAL Experiments (Dec 2025)

<details>
<summary>Click to expand ANYMAL experiments</summary>

#### WandB: flow-mbpo-single

| Variant | Job IDs | Status |
|---------|---------|--------|
| Baseline | 3080227-3080229 | ✅ COMPLETED |
| Flow WM K=4 | 3082681-3082683 | ✅ COMPLETED |
| All other variants | Various | ✅ COMPLETED |

</details>

---

## Resource Allocation

| Partition | GPU | Memory | Time Limit | Account |
|-----------|-----|--------|------------|---------|
| ice-gpu | H100/H200 | 450GB | 16 hours | coc |
| gpu-l40s | L40S | 400GB | 40 hours | gts-agarg35 |

---

## Checkpoint Locations

| Experiment | Path |
|------------|------|
| MT30 Pre-trained WM | `/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/checkpoints/multitask/mt30_48M_4900000.pt` |
| MT30 Data | `/home/hice1/eliu354/scratch/Data/tdmpc2/mt30/` |
| MT30 Outputs | `outputs/mt30/<variant>/<task>/seed<seed>/` |
