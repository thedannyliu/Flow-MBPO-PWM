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
**Method**: **Pretrain WM → Freeze/Finetune → Train Policy**
**Goal**: Create matched Flow WM and MLP WM checkpoints for fair 2×2 comparison.

| Job ID | WM Type | Iterations | GPU | Status | Notes |
|--------|---------|------------|-----|--------|-------|
| `4012664` | Flow WM | 200,000 | H100 | ⏳ QUEUED | `pwm_48M_mt_flowwm`, Output: `flowwm_mt30_best.pt` |
| `4012665` | MLP WM | 200,000 | H100 | ⏳ QUEUED | `pwm_48M_mt_baseline`, Output: `mlpwm_mt30_best.pt` |

**After WM pretraining completes**:
Use pretrained checkpoints for 2×2 factorial policy training:
- MLP WM + MLP Policy: `alg=pwm_48M_mt_baseline checkpoint=mlpwm_mt30_best.pt`
- MLP WM + Flow Policy: `alg=pwm_48M_mt_flowpolicy checkpoint=mlpwm_mt30_best.pt`
- Flow WM + MLP Policy: `alg=pwm_48M_mt_flowwm checkpoint=flowwm_mt30_best.pt`
- Flow WM + Flow Policy: `alg=pwm_48M_mt_fullflow checkpoint=flowwm_mt30_best.pt`

---

### Phase 7: Flow Policy Fine-tuning (Pretrained MLP WM)
**Method**: **Load Pretrained Weight** (Original PWM checkpoint)
**Checkpoint**: `checkpoints/multitask/mt30_48M_4900000.pt`

| Job ID | Status | Config | Variant | Methodology | Notes |
|--------|--------|--------|---------|-------------|-------|
| `4012601` | ⏳ QUEUED | Multiple | Array 0-26 | `finetune_wm=True` | H100, 15k Epochs. 3 Variants × 3 Tasks × 3 Seeds |

---

### Phase 6: Epoch Sweep (From Scratch)
**Method**: **Joint Training (From Scratch)**
**Checkpoint**: **None** (Random Initialization)

| Job ID | Status | Config | Epochs | Methodology | Notes |
|--------|--------|--------|--------|-------------|-------|
| `4012555` | 🟢 RUNNING | `pwm_48M_mt_baseline` | 150,000 | `finetune_wm=True` | Baseline. H200. |
| `4012556` | ⏳ QUEUED | `pwm_48M_mt_fullflow` | 150,000 | `finetune_wm=True` | Flow. H200. |
| `4012537` | 🟢 RUNNING | `pwm_48M_mt_baseline` | 100,000 | `finetune_wm=True` | Baseline. H200. |
| `4012538` | 🟢 RUNNING | `pwm_48M_mt_fullflow` | 100,000 | `finetune_wm=True` | Flow. H200. |
| `4012535` | 🟢 RUNNING | `pwm_48M_mt_baseline` | 50,000 | `finetune_wm=True` | Baseline. H100. |
| `4012536` | ⏳ QUEUED | `pwm_48M_mt_fullflow` | 50,000 | `finetune_wm=True` | Flow. H100. |
| `4012533` | ✅ COMPLETED | `pwm_48M_mt_baseline` | 15,000 | `finetune_wm=True` | Baseline. H100. ~50min. |
| `4012534` | 🟢 RUNNING | `pwm_48M_mt_fullflow` | 15,000 | `finetune_wm=True` | Flow. H100. |

---

## ✅ Completed Phases

### Phase 6 (15k Baseline) Results
**Method**: Joint Training From Scratch, 15k epochs
**Status**: ✅ COMPLETED

| Job ID | Algo | Task | Seed | Final Reward | Planning Reward |
|--------|------|------|------|--------------|-----------------|
| `4012533_0` | Baseline | reacher-easy | 42 | 54.00 | 0.60 |
| `4012533_1` | Baseline | reacher-easy | 123 | 153.40 | 121.20 |
| `4012533_2` | Baseline | reacher-easy | 456 | 3.30 | 92.10 |
| `4012533_3` | Baseline | walker-stand | 42 | 139.98 | 137.81 |
| `4012533_4` | Baseline | walker-stand | 123 | 156.05 | 99.90 |
| `4012533_5` | Baseline | walker-stand | 456 | 151.09 | 143.17 |
| `4012533_6` | Baseline | cheetah-run | 42 | 0.19 | 0.14 |
| `4012533_7` | Baseline | cheetah-run | 123 | 0.14 | N/A |
| `4012533_8` | Baseline | cheetah-run | 456 | N/A | N/A |

**Conclusion**: 15k epochs from scratch is severely undertrained. Cheetah-run failed completely.

---

### Phase 5: Flow Tuning Results
**Method**: Joint Training From Scratch, 15k epochs
**Status**: ✅ COMPLETED

| Job ID | Config | Task | Seed | Final Reward | Notes |
|--------|--------|------|------|--------------|-------|
| `4012434_0` | high_prec_wm | walker-stand | 42 | 142.26 | Best result |
| `4012434_1` | high_prec_wm | walker-stand | 123 | 117.88 | |
| `4012434_10` | high_prec_policy | cheetah-run | 42 | 0.19 | Failed |
| `4012434_11` | high_prec_policy | cheetah-run | 123 | 0.24 | Failed |
| ... | ... | ... | ... | ~0.2 | Most cheetah-run failed |

**Conclusion**: From-scratch tuning experiments failed due to insufficient training budget.

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
| `4012433_8` | cheetah-run | 456 | N/A | Failed |

**Conclusion**: 10k epochs is severely insufficient for from-scratch WM+Policy training.

---

### Phase 3: Baseline vs Flow Policy (Pretrained WM)
**Method**: **Load Pretrained Weight** + Policy Fine-tuning
**Checkpoint**: `checkpoints/multitask/mt30_48M_4900000.pt`
**Status**: ✅ COMPLETED

| Job ID | Algo | Task | Seed | Reward | Checkpoint Location |
|--------|------|------|------|--------|---------------------|
| `4011713` | Baseline | reacher-easy | 42 | 981.20 | `outputs/mt30/baseline/` |
| `4011713` | Baseline | reacher-easy | 123 | 983.50 | |
| `4011713` | Baseline | reacher-easy | 456 | 982.20 | |
| `4011713` | Baseline | walker-stand | 42 | 972.32 | |
| `4011713` | Baseline | walker-stand | 123 | 923.48 | |
| `4011713` | Baseline | walker-stand | 456 | 977.35 | |
| `4011713` | Baseline | cheetah-run | 42 | 93.69 | |
| `4011713` | Baseline | cheetah-run | 123 | 108.80 | |
| `4011713` | Baseline | cheetah-run | 456 | 134.97 | |
| `4011714` | Flow | reacher-easy | 42 | 976.70 | `outputs/mt30/flow_policy/` |
| `4011714` | Flow | reacher-easy | 123 | 983.40 | |
| `4011714` | Flow | reacher-easy | 456 | 981.90 | |
| `4011740` | Flow | walker-stand | 42 | 854.53 | |
| `4011714` | Flow | walker-stand | 123 | 744.92 | |
| `4011740` | Flow | walker-stand | 456 | 919.90 | |
| `4011740` | Flow | cheetah-run | 42 | 80.97 | |
| `4011740` | Flow | cheetah-run | 123 | 94.75 | |
| `4011740` | Flow | cheetah-run | 456 | 120.52 | |

**Conclusion**: With pretrained WM, both Baseline and Flow Policy achieve good results. Baseline slightly better on walker-stand.

---

## 🛠 Resource & Config Reference

### Training Methodologies
| Type | Checkpoint Strategy | `finetune_wm` | Purpose |
|------|---------------------|---------------|---------|
| **Policy Fine-tuning** | **Load Pretrained** | `False` | Isolate Policy (Phase 3) |
| **Policy + WM Tuning** | **Load Pretrained** | `True` | Adapt WM to Policy (Phase 7) |
| **Joint Training** | **None (From Scratch)** | `True` | Train WM+Policy (Phase 4/5/6) |
| **WM Pretraining** | **None (From Scratch)** | N/A | Pretrain WM only (Phase 8) |

### Checkpoint Locations
| Type | Path |
|------|------|
| **Original PWM** | `checkpoints/multitask/mt30_48M_4900000.pt` |
| **Flow WM (Phase 8)** | `outputs/<hydra_dir>/logs/flowwm_mt30_best.pt` |
| **MLP WM (Phase 8)** | `outputs/<hydra_dir>/logs/mlpwm_mt30_best.pt` |
| **MT30 Data** | `/home/hice1/eliu354/scratch/Data/tdmpc2/mt30/` |
