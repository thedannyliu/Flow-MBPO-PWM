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

### MT30 Smoke Test (2026-01-03)

| Job ID | Config | Task | Seed | Status | Runtime | Notes |
|--------|--------|------|------|--------|---------|-------|
| 4010862 | pwm_48M_mt_baseline | reacher-easy | 42 | ✅ **COMPLETED** | 02:16 | 🎉 **SUCCESS!** R=1000.00, Planning R=993.00, 24000 eps loaded |
| 4010834 | pwm_48M_mt_baseline | reacher-easy | 42 | ❌ FAILED | 01:30 | Crashed on save: missing `best_policy_loss` attr. Training worked (R=978) |
| 4010832 | pwm_48M_mt_baseline | reacher-easy | 42 | ❌ FAILED | 00:14 | Buffer path error: `pwm.utils.buffer` → fixed to `flow_mbpo_pwm` |
| 4010831 | pwm_48M_mt_baseline | reacher-easy | 42 | ❌ FAILED | 00:29 | Buffer locating error |
| 4010829 | pwm_48M_mt_baseline | reacher-easy | 42 | ❌ FAILED | 00:08 | Missing pandas |
| 4010821 | pwm_48M_mt_baseline | reacher-easy | 42 | ❌ FAILED | 00:12 | Missing hydra |
| 4010817 | pwm_48M_mt_baseline | reacher-easy | 42 | ❌ FAILED | 00:03 | dm_env missing |

**Environment Setup Jobs (2026-01-03)**:
| Job ID | Purpose | Status |
|--------|---------|--------|
| 4010830 | Install pandas | ✅ COMPLETED |
| 4010828 | Install hydra, omegaconf, tensordict, torchrl, wandb | ✅ COMPLETED |
| 4010820 | Install lxml for dm_control | ✅ COMPLETED |
| 4010818 | Install dm_env in flow-mbpo env | ✅ COMPLETED |
| 4010814 | Conda install dm_control | ✅ COMPLETED |
| 4010811-4010815 | Various dm_control install attempts | ✅ COMPLETED |

---

## Experiment Matrix - MT30 Multitask

### WandB Project: `flow-mbpo-multitask`

### Baseline (MLP WM + MLP Policy)
| Task | Seed 42 | Seed 123 | Seed 456 |
|------|---------|----------|----------|
| reacher-easy | ✅ 4010862 (R=1000) | ⏳ Pending | ⏳ Pending |
| walker-stand | ⏳ Pending | ⏳ Pending | ⏳ Pending |
| cheetah-run | ⏳ Pending | ⏳ Pending | ⏳ Pending |

### Flow Policy (MLP WM + Flow Policy)
| Task | Seed 42 | Seed 123 | Seed 456 |
|------|---------|----------|----------|
| reacher-easy | ⏳ Pending | ⏳ Pending | ⏳ Pending |
| walker-stand | ⏳ Pending | ⏳ Pending | ⏳ Pending |
| cheetah-run | ⏳ Pending | ⏳ Pending | ⏳ Pending |

### Full Flow (Flow WM + Flow Policy)
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
| Flow WM K=2 | 3082684-3082686 | ✅ COMPLETED |
| Flow WM K=8 | 3084837-3084839 | ✅ COMPLETED |
| Flow Policy | 3084840-3084842 | ✅ COMPLETED |
| Full Flow | 3084843, 3087706-3087707 | ✅ COMPLETED |
| Flow WM H=8 | 3087697-3087699 | ✅ COMPLETED |
| All other variants | Various | ✅ COMPLETED |

</details>

---

## Resource Allocation

| Partition | GPU | Memory | Time Limit | Account |
|-----------|-----|--------|------------|---------|
| ice-gpu | H100/H200 | 495GB | 16 hours | coc |
| gpu-l40s | L40S | 400GB | 40 hours | gts-agarg35 |

---

## Checkpoint Locations

| Experiment | Path |
|------------|------|
| MT30 Pre-trained WM | `/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/checkpoints/multitask/mt30_48M_4900000.pt` |
| MT30 Data | `/home/hice1/eliu354/scratch/Data/tdmpc2/mt30/` |
| MT30 Outputs | `outputs/mt30/<task>/seed<seed>/` |

