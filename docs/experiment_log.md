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

### Phase 8: 2×2 Factorial WM Pretraining
**Method**: **Pretrain WM → (Freeze/Finetune) → Train Policy**
**Goal**: Create matched Flow WM and MLP WM checkpoints for fair 2×2 comparison.

**Config Alignment Verification** (vs Original PWM README):
| Parameter | Original PWM | Our Config | Status |
|-----------|--------------|------------|--------|
| `horizon` | 16 | 16 | ✅ Aligned |
| `batch_size` (pretraining) | 1024 | 1024 | ✅ Aligned |
| `wm_batch_size` (policy training) | 256 | 256 | ✅ Aligned |

| Job ID | WM Type | Config | Iterations | GPU | Status | Output |
|--------|---------|--------|------------|-----|--------|--------|
| `4012664` | Flow WM | `pwm_48M_mt_flowwm` | 200,000 | H100 | ⏳ QUEUED | `flowwm_mt30_best.pt` |
| `4012665` | MLP WM | `pwm_48M_mt_baseline` | 200,000 | H100 | ⏳ QUEUED | `mlpwm_mt30_best.pt` |

**After WM pretraining completes**, use checkpoints for 2×2 factorial:
| WM | Policy | Config | Checkpoint |
|---|---|---|---|
| MLP | MLP | `pwm_48M_mt_baseline` | `mlpwm_mt30_best.pt` |
| MLP | Flow | `pwm_48M_mt_flowpolicy` | `mlpwm_mt30_best.pt` |
| Flow | MLP | `pwm_48M_mt_flowwm` | `flowwm_mt30_best.pt` |
| Flow | Flow | `pwm_48M_mt_fullflow` | `flowwm_mt30_best.pt` |

---

### Phase 7: Flow Policy Fine-tuning (Pretrained MLP WM)
**Method**: **Load Pretrained Weight** (Original PWM checkpoint)
**Checkpoint**: `checkpoints/multitask/mt30_48M_4900000.pt`
**Setting**: `finetune_wm=True` (Fine-tune the pretrained WM)

| Job ID | Array | Config | Variant | Task | Seed | Status | Notes |
|--------|-------|--------|---------|------|------|--------|-------|
| `4012601_0` | 0 | `pwm_48M_mt_baseline` | Baseline | reacher-easy | 42 | ⏳ QUEUED | |
| `4012601_1` | 1 | `pwm_48M_mt_baseline` | Baseline | reacher-easy | 123 | ⏳ QUEUED | |
| `4012601_2` | 2 | `pwm_48M_mt_baseline` | Baseline | reacher-easy | 456 | ⏳ QUEUED | |
| `4012601_3` | 3 | `pwm_48M_mt_baseline` | Baseline | walker-stand | 42 | ⏳ QUEUED | |
| `4012601_4` | 4 | `pwm_48M_mt_baseline` | Baseline | walker-stand | 123 | ⏳ QUEUED | |
| `4012601_5` | 5 | `pwm_48M_mt_baseline` | Baseline | walker-stand | 456 | ⏳ QUEUED | |
| `4012601_6` | 6 | `pwm_48M_mt_baseline` | Baseline | cheetah-run | 42 | ⏳ QUEUED | |
| `4012601_7` | 7 | `pwm_48M_mt_baseline` | Baseline | cheetah-run | 123 | ⏳ QUEUED | |
| `4012601_8` | 8 | `pwm_48M_mt_baseline` | Baseline | cheetah-run | 456 | ⏳ QUEUED | |
| `4012601_9` | 9 | `pwm_48M_mt_flowpolicy` | Flow Std (substeps=2) | reacher-easy | 42 | ⏳ QUEUED | |
| `4012601_10` | 10 | `pwm_48M_mt_flowpolicy` | Flow Std (substeps=2) | reacher-easy | 123 | ⏳ QUEUED | |
| `4012601_11` | 11 | `pwm_48M_mt_flowpolicy` | Flow Std (substeps=2) | reacher-easy | 456 | ⏳ QUEUED | |
| `4012601_12` | 12 | `pwm_48M_mt_flowpolicy` | Flow Std (substeps=2) | walker-stand | 42 | ⏳ QUEUED | |
| `4012601_13` | 13 | `pwm_48M_mt_flowpolicy` | Flow Std (substeps=2) | walker-stand | 123 | ⏳ QUEUED | |
| `4012601_14` | 14 | `pwm_48M_mt_flowpolicy` | Flow Std (substeps=2) | walker-stand | 456 | ⏳ QUEUED | |
| `4012601_15` | 15 | `pwm_48M_mt_flowpolicy` | Flow Std (substeps=2) | cheetah-run | 42 | ⏳ QUEUED | |
| `4012601_16` | 16 | `pwm_48M_mt_flowpolicy` | Flow Std (substeps=2) | cheetah-run | 123 | ⏳ QUEUED | |
| `4012601_17` | 17 | `pwm_48M_mt_flowpolicy` | Flow Std (substeps=2) | cheetah-run | 456 | ⏳ QUEUED | |
| `4012601_18` | 18 | `pwm_48M_mt_flowpolicy` | Flow High (substeps=4) | reacher-easy | 42 | ⏳ QUEUED | |
| `4012601_19` | 19 | `pwm_48M_mt_flowpolicy` | Flow High (substeps=4) | reacher-easy | 123 | ⏳ QUEUED | |
| `4012601_20` | 20 | `pwm_48M_mt_flowpolicy` | Flow High (substeps=4) | reacher-easy | 456 | ⏳ QUEUED | |
| `4012601_21` | 21 | `pwm_48M_mt_flowpolicy` | Flow High (substeps=4) | walker-stand | 42 | ⏳ QUEUED | |
| `4012601_22` | 22 | `pwm_48M_mt_flowpolicy` | Flow High (substeps=4) | walker-stand | 123 | ⏳ QUEUED | |
| `4012601_23` | 23 | `pwm_48M_mt_flowpolicy` | Flow High (substeps=4) | walker-stand | 456 | ⏳ QUEUED | |
| `4012601_24` | 24 | `pwm_48M_mt_flowpolicy` | Flow High (substeps=4) | cheetah-run | 42 | ⏳ QUEUED | |
| `4012601_25` | 25 | `pwm_48M_mt_flowpolicy` | Flow High (substeps=4) | cheetah-run | 123 | ⏳ QUEUED | |
| `4012601_26` | 26 | `pwm_48M_mt_flowpolicy` | Flow High (substeps=4) | cheetah-run | 456 | ⏳ QUEUED | |

---

### Phase 6: Epoch Sweep (From Scratch)
**Method**: **Joint Training (From Scratch)**
**Checkpoint**: **None** (Random Initialization)
**Setting**: `finetune_wm=True`

#### Phase 6-D: 150k Epochs (H200)
| Job ID | Config | Task | Seed | Status | Notes |
|--------|--------|------|------|--------|-------|
| `4012555_0` | `pwm_48M_mt_baseline` | reacher-easy | 42 | ⏳ QUEUED | |
| `4012555_1` | `pwm_48M_mt_baseline` | reacher-easy | 123 | ⏳ QUEUED | |
| `4012555_2` | `pwm_48M_mt_baseline` | reacher-easy | 456 | ⏳ QUEUED | |
| `4012555_3` | `pwm_48M_mt_baseline` | walker-stand | 42 | 🟢 RUNNING | ~2h |
| `4012555_4-8` | `pwm_48M_mt_baseline` | walker/cheetah | * | ⏳ QUEUED | |
| `4012556_0-8` | `pwm_48M_mt_fullflow` | All | * | ⏳ QUEUED | Flow 150k |

#### Phase 6-C: 100k Epochs (H200)
| Job ID | Config | Task | Seed | Status | Runtime | Notes |
|--------|--------|------|------|--------|---------|-------|
| `4012537_0` | `pwm_48M_mt_baseline` | reacher-easy | 42 | 🟢 RUNNING | ~2h | |
| `4012537_1` | `pwm_48M_mt_baseline` | reacher-easy | 123 | 🟢 RUNNING | ~2h | |
| `4012537_2` | `pwm_48M_mt_baseline` | reacher-easy | 456 | 🟢 RUNNING | ~2h | |
| `4012537_3` | `pwm_48M_mt_baseline` | walker-stand | 42 | 🟢 RUNNING | ~2h | |
| `4012537_4` | `pwm_48M_mt_baseline` | walker-stand | 123 | 🟢 RUNNING | ~2h | |
| `4012537_5` | `pwm_48M_mt_baseline` | walker-stand | 456 | 🟢 RUNNING | ~2h | |
| `4012537_6` | `pwm_48M_mt_baseline` | cheetah-run | 42 | 🟢 RUNNING | ~2h | |
| `4012537_7` | `pwm_48M_mt_baseline` | cheetah-run | 123 | 🟢 RUNNING | ~2h | |
| `4012537_8` | `pwm_48M_mt_baseline` | cheetah-run | 456 | 🟢 RUNNING | ~2h | |
| `4012538_0` | `pwm_48M_mt_fullflow` | reacher-easy | 42 | 🟢 RUNNING | ~2h | Flow |
| `4012538_2-8` | `pwm_48M_mt_fullflow` | All | * | 🟢 RUNNING | ~2h | Flow |

#### Phase 6-B: 50k Epochs (H100)
| Job ID | Config | Task | Seed | Status | Notes |
|--------|--------|------|------|--------|-------|
| `4012535_0` | `pwm_48M_mt_baseline` | reacher-easy | 42 | 🟢 RUNNING | ~18min |
| `4012535_1` | `pwm_48M_mt_baseline` | reacher-easy | 123 | 🟢 RUNNING | ~18min |
| `4012535_2` | `pwm_48M_mt_baseline` | reacher-easy | 456 | 🟢 RUNNING | ~18min |
| `4012535_3` | `pwm_48M_mt_baseline` | walker-stand | 42 | 🟢 RUNNING | ~16min |
| `4012535_4-7` | `pwm_48M_mt_baseline` | walker/cheetah | * | 🟢 RUNNING | |
| `4012535_8` | `pwm_48M_mt_baseline` | cheetah-run | 456 | ⏳ QUEUED | |
| `4012536_0-8` | `pwm_48M_mt_fullflow` | All | * | ⏳ QUEUED | Flow 50k |

#### Phase 6-A: 15k Epochs (H100) - COMPLETED
| Job ID | Config | Task | Seed | Status | Final Reward |
|--------|--------|------|------|--------|--------------|
| `4012533_0` | `pwm_48M_mt_baseline` | reacher-easy | 42 | ✅ COMPLETED | 54.00 |
| `4012533_1` | `pwm_48M_mt_baseline` | reacher-easy | 123 | ✅ COMPLETED | 153.40 |
| `4012533_2` | `pwm_48M_mt_baseline` | reacher-easy | 456 | ✅ COMPLETED | 3.30 |
| `4012533_3` | `pwm_48M_mt_baseline` | walker-stand | 42 | ✅ COMPLETED | 139.98 |
| `4012533_4` | `pwm_48M_mt_baseline` | walker-stand | 123 | ✅ COMPLETED | 156.05 |
| `4012533_5` | `pwm_48M_mt_baseline` | walker-stand | 456 | ✅ COMPLETED | 151.09 |
| `4012533_6` | `pwm_48M_mt_baseline` | cheetah-run | 42 | ✅ COMPLETED | 0.19 |
| `4012533_7` | `pwm_48M_mt_baseline` | cheetah-run | 123 | ✅ COMPLETED | 0.14 |
| `4012533_8` | `pwm_48M_mt_baseline` | cheetah-run | 456 | ✅ COMPLETED | ~0.2 |

**Flow 15k Epochs** (Running):
| Job ID | Config | Task | Seed | Status | Notes |
|--------|--------|------|------|--------|-------|
| `4012534_0` | `pwm_48M_mt_fullflow` | reacher-easy | 42 | 🟢 RUNNING | ~50min |
| `4012534_1` | `pwm_48M_mt_fullflow` | reacher-easy | 123 | 🟢 RUNNING | ~42min |
| `4012534_2` | `pwm_48M_mt_fullflow` | reacher-easy | 456 | 🟢 RUNNING | ~42min |
| `4012534_3` | `pwm_48M_mt_fullflow` | walker-stand | 42 | 🟢 RUNNING | ~36min |
| `4012534_4` | `pwm_48M_mt_fullflow` | walker-stand | 123 | 🟢 RUNNING | ~36min |
| `4012534_5` | `pwm_48M_mt_fullflow` | walker-stand | 456 | 🟢 RUNNING | ~35min |
| `4012534_6` | `pwm_48M_mt_fullflow` | cheetah-run | 42 | 🟢 RUNNING | ~20min |
| `4012534_7` | `pwm_48M_mt_fullflow` | cheetah-run | 123 | 🟢 RUNNING | ~20min |
| `4012534_8` | `pwm_48M_mt_fullflow` | cheetah-run | 456 | 🟢 RUNNING | ~19min |

---

## ✅ Completed Phases

### Phase 5: Flow Tuning (Hyperparameters)
**Method**: Joint Training From Scratch, 15k epochs
**Status**: ✅ COMPLETED

| Job ID | Config Variant | Task | Seed | Final Reward | Notes |
|--------|----------------|------|------|--------------|-------|
| `4012434_0` | high_prec_wm (substeps=8) | walker-stand | 42 | 142.26 | Best |
| `4012434_1` | high_prec_wm | walker-stand | 123 | 117.88 | |
| `4012434_2` | high_prec_wm | walker-stand | 456 | ~130 | |
| `4012434_3` | high_prec_wm | cheetah-run | 42 | ~120 | |
| `4012434_4` | high_prec_wm | cheetah-run | 123 | ~130 | |
| `4012434_5` | high_prec_wm | cheetah-run | 456 | ~120 | |
| `4012434_6` | high_prec_policy (substeps=4) | walker-stand | 42 | ~80 | |
| `4012434_7` | high_prec_policy | walker-stand | 123 | ~80 | |
| `4012434_8` | high_prec_policy | walker-stand | 456 | ~80 | |
| `4012434_9` | high_prec_policy | cheetah-run | 42 | ~0.2 | Failed |
| `4012434_10` | high_prec_policy | cheetah-run | 123 | 0.19 | Failed |
| `4012434_11` | high_prec_policy | cheetah-run | 456 | 0.24 | Failed |
| `4012434_12` | euler_fast | walker-stand | 42 | 0.21 | Failed |
| `4012434_13-17` | euler_fast | * | * | ~0.2 | All Failed |

**Conclusion**: From-scratch tuning with 15k epochs is insufficient. High precision WM helped slightly.

---

### Phase 4: Full Flow Training (10k epochs)
**Method**: Joint Training From Scratch
**Status**: ✅ COMPLETED

| Job ID | Task | Seed | Final Reward | Notes |
|--------|------|------|--------------|-------|
| `4012433_0` | reacher-easy | 42 | 112.20 | |
| `4012433_1` | reacher-easy | 123 | 39.10 | |
| `4012433_2` | reacher-easy | 456 | 104.50 | |
| `4012433_3` | walker-stand | 42 | 141.68 | |
| `4012433_4` | walker-stand | 123 | 140.52 | |
| `4012433_5` | walker-stand | 456 | 127.64 | |
| `4012433_6` | cheetah-run | 42 | 0.16 | Failed |
| `4012433_7` | cheetah-run | 123 | 1.40 | Failed |
| `4012433_8` | cheetah-run | 456 | ~0.2 | Failed |

---

### Phase 3: Baseline vs Flow Policy (Pretrained WM, Frozen)
**Method**: **Load Pretrained Weight** + Policy Training (Frozen WM)
**Checkpoint**: `checkpoints/multitask/mt30_48M_4900000.pt`
**Status**: ✅ COMPLETED

| Job ID | Algo | Task | Seed | Reward | Notes |
|--------|------|------|------|--------|-------|
| `4011713_0` | Baseline | reacher-easy | 42 | 981.20 | |
| `4011713_1` | Baseline | reacher-easy | 123 | 983.50 | |
| `4011713_2` | Baseline | reacher-easy | 456 | 982.20 | |
| `4011713_3` | Baseline | walker-stand | 42 | 972.32 | |
| `4011713_4` | Baseline | walker-stand | 123 | 923.48 | |
| `4011713_5` | Baseline | walker-stand | 456 | 977.35 | |
| `4011713_6` | Baseline | cheetah-run | 42 | 93.69 | |
| `4011713_7` | Baseline | cheetah-run | 123 | 108.80 | |
| `4011713_8` | Baseline | cheetah-run | 456 | 134.97 | |
| `4011714_0` | Flow Policy | reacher-easy | 42 | 976.70 | |
| `4011714_1` | Flow Policy | reacher-easy | 123 | 983.40 | |
| `4011714_2` | Flow Policy | reacher-easy | 456 | 981.90 | |
| `4011740_0` | Flow Policy | walker-stand | 42 | 854.53 | |
| `4011714_1` | Flow Policy | walker-stand | 123 | 744.92 | |
| `4011740_2` | Flow Policy | walker-stand | 456 | 919.90 | |
| `4011740_3` | Flow Policy | cheetah-run | 42 | 80.97 | |
| `4011740_4` | Flow Policy | cheetah-run | 123 | 94.75 | |
| `4011740_5` | Flow Policy | cheetah-run | 456 | 120.52 | |

---

## 🛠 Resource & Config Reference

### Training Methodologies
| Type | Checkpoint | `finetune_wm` | Purpose |
|------|------------|---------------|---------|
| **Policy Fine-tuning** | Load Pretrained | `False` | Phase 3 |
| **Policy + WM Tuning** | Load Pretrained | `True` | Phase 7 |
| **Joint Training** | None (Scratch) | `True` | Phase 4/5/6 |
| **WM Pretraining** | None (Scratch) | N/A | Phase 8 |

### Config Alignment (Original PWM)
| Parameter | Value | Source |
|-----------|-------|--------|
| `horizon` | 16 | Original PWM README |
| `batch_size` (WM pretrain) | 1024 | Original PWM README |
| `wm_batch_size` (policy) | 256 | `pwm_48M.yaml` |
| `wm_iterations` | 8 | `pwm_48M.yaml` |
| `wm_buffer_size` | 1,000,000 | `pwm_48M.yaml` |

### Checkpoint Locations
| Type | Path |
|------|------|
| **Original PWM** | `checkpoints/multitask/mt30_48M_4900000.pt` |
| **Flow WM (Phase 8)** | `outputs/<hydra>/logs/flowwm_mt30_best.pt` |
| **MLP WM (Phase 8)** | `outputs/<hydra>/logs/mlpwm_mt30_best.pt` |
| **MT30 Data** | `/home/hice1/eliu354/scratch/Data/tdmpc2/mt30/` |
