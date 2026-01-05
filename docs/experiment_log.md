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
> **Important fields to track (each run/job)**:
> - `Job ID`: Slurm job ID for tracking
> - `Config`: Algorithm config file used (e.g., `pwm_48M_mt_baseline.yaml`)
> - `Task`: Task name (e.g., `reacher-easy`, `walker-stand`)
> - `Seed`: Random seed for reproducibility
> - `Checkpoint`: Path to saved model checkpoint
> - `WandB`: Link to WandB run for visualization
> - `Notes`: Any important observations or issues
> - `Location`: Where the checkpoints locate

---

## 🟢 Active Experiments

### Phase 6: Epoch Sweep (Baseline vs Full Flow)
- **Date**: Jan 04, 2026
- **Goal**: Determine optimal training duration for Flow vs Baseline (from scratch).
- **Methodology**:
    - **Baseline**: MLP World Model + MLP Policy. `finetune_wm=True` (Train from scratch).
    - **Flow**: Flow World Model + Flow Policy. `finetune_wm=True` (Train from scratch).
    - **Note**: Original PWM pretraining takes ~2 weeks. We are testing if Flow can learn efficiently in shorter horizons (15k-150k).
- **Config Alignment**: All parameters match original PWM (`wm_batch_size=256`).
- **WandB Project**: `MT30-Detailed`

| Epochs | GPU | Baseline Job | Flow Job | Status |
|--------|-----|--------------|----------|--------|
| **15,000** | H100 | `4012533` | `4012534` | ⏳ QUEUED |
| **50,000** | H100 | `4012535` | `4012536` | ⏳ QUEUED |
| **100,000** | H200 | `4012537` | `4012538` | 🟢 RUNNING |
| **150,000** | H200 | `4012555` | `4012556` | 🟢 RUNNING |

### Phase 5: Flow Tuning (Ongoing)
- **Job ID**: `4012434` (Array 0-17)
- **Status**: 🟢 **RUNNING** (14/18 complete)
- **Methodology**: Full Flow (From Scratch), 15,000 epochs. Tuning `substeps` and `integrator`.

---

## ✅ Completed Phases

### Phase 4: Full Flow Training (Attempt 12)
- **Goal**: End-to-end training of Flow WM + Flow Policy (From Scratch).
- **Job ID**: `4012433` (Array 0-8)
- **Status**: ✅ **COMPLETED** (Jan 04 2026)
- **Methodology**:
    - **Training**: Joint Training (WM + Policy) from scratch (`finetune_wm=True`).
    - **Duration**: **10,000 Epochs**.
    - **Outcome**: **Severely Undertrained**. Reacher Reward ~112 (vs ~980 Baseline).
- **Analysis**: 10k epochs is insufficient for learning dynamics from scratch. Original PWM uses pretraining (millions of steps).

### Phase 3: Baseline vs Flow Policy (Pretrained WM)
**Goal**: Isolate policy performance by using a FROZEN, PRETRAINED World Model.
**Status**: ✅ **COMPLETED** (Jan 04 2026)
**Data Source**: Results aggregated in **`mt30_results_summary.csv`**
**Methodology**:
    - **Training**: **Policy Only**. World Model was loaded from `mt30_48M_4900000.pt` and **Frozen** (`finetune_wm=False`).
    - **Duration**: 15,000 Epochs (Standard for Policy Fine-tuning).
    - **Outcome**: Flow Policy matches Baseline on easy tasks, but Baseline is more robust on complex tasks (`walker-stand`).

**Aggregate Results (Mean Reward)**:
| Task | Baseline (MLP) | Flow Policy (ODE) | Diff | Winner |
|------|---------------|-------------------|------|--------|
| **reacher-easy** | **982.30** | 980.67 | -0.2% | Tie |
| **walker-stand** | **957.72** | 839.78 | -12% | Baseline |
| **cheetah-run** | **112.48** | 98.74 | -12% | Tie (Both Low) |

<details>
<summary>View Detailed Seed Metrics (Phase 3)</summary>

| Algo | Task | Seed | Reward | Plan Reward | Job ID |
|---|---|---|---|---|---|
| **Baseline** | reacher-easy | 42 | 981.20 | 981.70 | 4011713 |
| **Baseline** | reacher-easy | 123 | 983.50 | 985.60 | 4011713 |
| **Baseline** | reacher-easy | 456 | 982.20 | 985.50 | 4011713 |
| **Baseline** | walker-stand | 42 | 972.32 | 943.97 | 4011713 |
| **Baseline** | walker-stand | 123 | 923.48 | 933.11 | 4011713 |
| **Baseline** | walker-stand | 456 | 977.35 | 978.33 | 4011713 |
| **Baseline** | cheetah-run | 42 | 93.69 | 88.13 | 4011713 |
| **Baseline** | cheetah-run | 123 | 108.80 | 81.67 | 4011713 |
| **Baseline** | cheetah-run | 456 | 134.97 | 114.09 | 4011713 |
| **Flow** | reacher-easy | 42 | 976.70 | 982.90 | 4011714 |
| **Flow** | reacher-easy | 123 | 983.40 | 986.00 | 4011714 |
| **Flow** | reacher-easy | 456 | 981.90 | 983.90 | 4011714 |
| **Flow** | walker-stand | 42 | 854.53 | 864.39 | 4011740 |
| **Flow** | walker-stand | 123 | 744.92 | 796.67 | 4011714 |
| **Flow** | walker-stand | 456 | 919.90 | 933.66 | 4011740 |
| **Flow** | cheetah-run | 42 | 80.97 | 82.31 | 4011740 |
| **Flow** | cheetah-run | 123 | 94.75 | 73.21 | 4011740 |
| **Flow** | cheetah-run | 456 | 120.52 | 110.27 | 4011740 |

</details>

---

## 📂 Archived Experiments (Failed / Historical)

<details>
<summary>View Previous Attempts (Attempts 1-14)</summary>

### Attempt 14: Flow Tuning (Jan 04)
- **Job ID**: `4011988`
- **Status**: ❌ **FAILED** (Storage Full)
- **Methodology**: Flow Tuning (From Scratch), 15k epochs.

### Attempt 13: Baseline From Scratch (Jan 04)
- **Job ID**: `4011987`
- **Status**: ❌ **FAILED** (Undertrained)
- **Methodology**: Baseline (From Scratch). `finetune_wm=True`, 15k epochs.
- **Outcome**: Confirmed that 15k epochs is insufficient for Baseline to learn dynamics from scratch.

### Attempt 12 & 11: Full Flow & Debug (Jan 04)
- **Job IDs**: `4012027`, `4012028`
- **Status**: ❌ **FAILED** (Storage Full)

### Attempt 8 & 9: Phase 3 Production (Jan 03)
- **Job IDs**: `4011713`, `4011714`, `4011740`
- **Status**: ✅ **COMPLETED**
- **Methodology**: Pretrained WM (Frozen) + Policy Training.

### Attempt 7: MT30 Collision Incident
- **Status**: ⚠️ **PARTIAL LOSS** (Hydra dir collision)

### Attempt 1-6: Initial Setup
- **Status**: ❌ **FAILED** / **CANCELLED**

</details>

---

## 🛠 Resource & Config Reference

### Training Methodologies
| Type | Pretrained WM | `finetune_wm` | Epochs | Purpose |
|------|---------------|---------------|--------|---------|
| **Policy Fine-tuning** | ✅ (Loaded) | `False` | 15k | Isolate Policy Performance (Phase 3) |
| **Joint Training** | ❌ (None) | `True` | 100k+ | Train WM + Policy From Scratch (Full Flow) |
| **Full Pretraining** | ❌ (None) | N/A | Millions | Original PWM Pretraining (Standard) |

### Configuration Files
| Config | World Model | Policy | Purpose |
|--------|-------------|--------|---------|
| `pwm_48M_mt_baseline.yaml` | MLP | MLP | Standard Baseline |
| `pwm_48M_mt_flowpolicy.yaml` | MLP | Flow ODE | Policy Comparison |
| `pwm_48M_mt_fullflow.yaml` | Flow | Flow ODE | Full Algorithm |

### Checkpoint Locations
| Type | Path |
|------|------|
| **Pretrained WM** | `checkpoints/multitask/mt30_48M_4900000.pt` |
| **MT30 Data** | `/home/hice1/eliu354/scratch/Data/tdmpc2/mt30/` |
| **Epoch Sweep Output** | `outputs/epoch_sweep/<variant>_<epochs>/` |
