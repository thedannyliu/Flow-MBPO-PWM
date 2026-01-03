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

### MT30 Full Training - Attempt 4 (2026-01-03 18:38 EST)
- **Status**: 🔄 **RUNNING**
- **Goal**: Compare Baseline (H100) vs Flow Policy (H200) with clear naming.
- **Hardware**: H100 for Baseline, **H200 for Flow Policy**.

**Baseline Array Job 4011449** (9 jobs: 3 tasks × 3 seeds)
| Array ID | Task | Seed | Job ID | Status | Hardware | WandB Name |
|----------|------|------|--------|--------|----------|------------|
| 0 | reacher-easy | 42 | 4011449_0 | 🔄 RUNNING | H100 | `baseline_H100_reacher-easy_s42` |
| 1 | reacher-easy | 123 | 4011449_1 | 🔄 RUNNING | H100 | `baseline_H100_reacher-easy_s123` |
| 2 | reacher-easy | 456 | 4011449_2 | 🔄 RUNNING | H100 | `baseline_H100_reacher-easy_s456` |
| 3 | walker-stand | 42 | 4011449_3 | 🔄 RUNNING | H100 | `baseline_H100_walker-stand_s42` |
| 4 | walker-stand | 123 | 4011449_4 | 🔄 RUNNING | H100 | `baseline_H100_walker-stand_s123` |
| 5 | walker-stand | 456 | 4011449_5 | 🔄 RUNNING | H100 | `baseline_H100_walker-stand_s456` |
| 6 | cheetah-run | 42 | 4011449_6 | 🔄 RUNNING | H100 | `baseline_H100_cheetah-run_s42` |
| 7 | cheetah-run | 123 | 4011449_7 | 🔄 RUNNING | H100 | `baseline_H100_cheetah-run_s123` |
| 8 | cheetah-run | 456 | 4011449_8 | 🔄 RUNNING | H100 | `baseline_H100_cheetah-run_s456` |

**Flow Policy Array Job 4011450** (9 jobs: 3 tasks × 3 seeds)
| Array ID | Task | Seed | Job ID | Status | Hardware | WandB Name |
|----------|------|------|--------|--------|----------|------------|
| 0 | reacher-easy | 42 | 4011450_0 | 🔄 RUNNING | **H200** | `flowpolicy_H200_reacher-easy_s42` |
| 1 | reacher-easy | 123 | 4011450_1 | 🔄 RUNNING | **H200** | `flowpolicy_H200_reacher-easy_s123` |
| 2 | reacher-easy | 456 | 4011450_2 | 🔄 RUNNING | **H200** | `flowpolicy_H200_reacher-easy_s456` |
| 3 | walker-stand | 42 | 4011450_3 | 🔄 RUNNING | **H200** | `flowpolicy_H200_walker-stand_s42` |
| 4 | walker-stand | 123 | 4011450_4 | 🔄 RUNNING | **H200** | `flowpolicy_H200_walker-stand_s123` |
| 5 | walker-stand | 456 | 4011450_5 | 🔄 RUNNING | **H200** | `flowpolicy_H200_walker-stand_s456` |
| 6 | cheetah-run | 42 | 4011450_6 | 🔄 RUNNING | **H200** | `flowpolicy_H200_cheetah-run_s42` |
| 7 | cheetah-run | 123 | 4011450_7 | 🔄 RUNNING | **H200** | `flowpolicy_H200_cheetah-run_s123` |
| 8 | cheetah-run | 456 | 4011450_8 | 🔄 RUNNING | **H200** | `flowpolicy_H200_cheetah-run_s456` |

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

### MT30 Full Training - Attempt 2 (2026-01-03 18:15 EST)
- **Status**: � **CANCELLED**
- **Reason**: User requested change of WandB project name and more detailed run tracking.
- **Goal**: Complete Baseline (MLP WM + MLP Policy) vs Flow Policy (MLP WM + Flow Policy) comparison.

**Baseline Array Job 4011379** (9 jobs: 3 tasks × 3 seeds)
| Array ID | Task | Seed | Job ID | Status | WandB Name |
|----------|------|------|--------|--------|------------|
| 0 | reacher-easy | 42 | 4011379_0 | � CANCELLED | baseline_reacher-easy_s42 |
| 1 | reacher-easy | 123 | 4011379_1 | � CANCELLED | baseline_reacher-easy_s123 |
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
