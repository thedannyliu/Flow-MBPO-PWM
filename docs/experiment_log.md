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

### MT30 Full Training (2026-01-03 04:37 EST)

**Baseline Array Job 4010895** (9 jobs: 3 tasks × 3 seeds)
| Array ID | Task | Seed | Job ID | Status | WandB Name |
|----------|------|------|--------|--------|------------|
| 0 | reacher-easy | 42 | 4010895_0 | 🔄 RUNNING | baseline_reacher-easy_s42 |
| 1 | reacher-easy | 123 | 4010895_1 | 🔄 RUNNING | baseline_reacher-easy_s123 |
| 2 | reacher-easy | 456 | 4010895_2 | 🔄 RUNNING | baseline_reacher-easy_s456 |
| 3 | walker-stand | 42 | 4010895_3 | 🔄 RUNNING | baseline_walker-stand_s42 |
| 4 | walker-stand | 123 | 4010895_4 | 🔄 RUNNING | baseline_walker-stand_s123 |
| 5 | walker-stand | 456 | 4010895_5 | 🔄 RUNNING | baseline_walker-stand_s456 |
| 6 | cheetah-run | 42 | 4010895_6 | 🔄 RUNNING | baseline_cheetah-run_s42 |
| 7 | cheetah-run | 123 | 4010895_7 | 🔄 RUNNING | baseline_cheetah-run_s123 |
| 8 | cheetah-run | 456 | 4010895_8 | ⏳ PENDING | baseline_cheetah-run_s456 |

**Flow Policy Array Job 4010896** (9 jobs: 3 tasks × 3 seeds)
| Array ID | Task | Seed | Job ID | Status | WandB Name |
|----------|------|------|--------|--------|------------|
| 0 | reacher-easy | 42 | 4010896_0 | ⏳ PENDING | flowpolicy_reacher-easy_s42 |
| 1 | reacher-easy | 123 | 4010896_1 | ⏳ PENDING | flowpolicy_reacher-easy_s123 |
| 2 | reacher-easy | 456 | 4010896_2 | ⏳ PENDING | flowpolicy_reacher-easy_s456 |
| 3 | walker-stand | 42 | 4010896_3 | ⏳ PENDING | flowpolicy_walker-stand_s42 |
| 4 | walker-stand | 123 | 4010896_4 | ⏳ PENDING | flowpolicy_walker-stand_s123 |
| 5 | walker-stand | 456 | 4010896_5 | ⏳ PENDING | flowpolicy_walker-stand_s456 |
| 6 | cheetah-run | 42 | 4010896_6 | ⏳ PENDING | flowpolicy_cheetah-run_s42 |
| 7 | cheetah-run | 123 | 4010896_7 | ⏳ PENDING | flowpolicy_cheetah-run_s123 |
| 8 | cheetah-run | 456 | 4010896_8 | ⏳ PENDING | flowpolicy_cheetah-run_s456 |

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
