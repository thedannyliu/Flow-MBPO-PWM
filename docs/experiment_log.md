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

### Phase 8: 2×2 Factorial (WM × Policy) on Pretrained WM
**Method**: **Pretrain WM → (Freeze / Fine-tune) → Train Policy**

**Goal**: Compare Flow impact location with a clean 2×2:
- **World Model**: `MLP` vs `Flow`
- **Policy**: `MLP` vs `Flow ODE`

| WM | Policy | `alg=` | Notes |
|---|---|---|---|
| MLP | MLP | `pwm_48M_mt_baseline` | Baseline |
| MLP | Flow | `pwm_48M_mt_flowpolicy` | Flow policy only |
| Flow | MLP | `pwm_48M_mt_flowwm` | Flow WM only |
| Flow | Flow | `pwm_48M_mt_fullflow` | Full Flow |

**WM checkpoints (must match architecture)**:
| WM Type | Checkpoint | Status |
|---|---|---|
| MLP WM | `checkpoints/multitask/mt30_48M_4900000.pt` (original PWM) | ✅ Available |
| Flow WM | `outputs/.../logs/flowwm_<iters>.pt` (from pretrain script) | ⏳ TBD |

**WM pretraining (for Flow WM; optional for MLP WM)**:
- Entry: `scripts/pretrain_multitask_wm.py -cn pretrain_mt30_wm`
- Examples:
  - Flow WM: `alg=pwm_48M_mt_flowwm general.out_name=flowwm general.wm_pretrain_iters=...`
  - MLP WM: `alg=pwm_48M_mt_baseline general.out_name=mlpwm general.wm_pretrain_iters=...`

**Policy training runs to queue (per task × seed)**:
| Setting | `finetune_wm` | Quadrants | Notes |
|---|---:|---|---|
| Frozen WM | `False` | 4 | Primary 2×2 comparison |
| Fine-tuned WM | `True` | 4 | Ablation: does WM adaptation change ranking? |

---

### Phase 7: Flow Policy Fine-tuning (Pretrained WM)
**Method**: **Load Pretrained Weight**
**Pretrained Checkpoint**: `/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/checkpoints/multitask/mt30_48M_4900000.pt`

| Job ID | Status | Config | Variant | Task/Seed | Methodology | Notes |
|--------|--------|--------|---------|-----------|-------------|-------|
| `4012601` | ⏳ QUEUED | Multiple | Array 0-26 | 3 Variants × 3 Tasks × 3 Seeds | `finetune_wm=True` | H100, 15k Epochs. Testing if Fine-tuning helps Pretrained WM adaptability. |

<details>
<summary>Job Detail Specifications (Phase 7)</summary>

- **Config Mappings**:
  - **Baseline**: `pwm_48M_mt_baseline`
  - **Flow Std**: `pwm_48M_mt_flowpolicy` (+ `substeps=2`)
  - **Flow High**: `pwm_48M_mt_flowpolicy` (+ `substeps=4`)
- **Common Params**: `finetune_wm=True`, `load_checkpoint=True`, `epochs=15000`
- **Location**: `outputs/phase7/<variant>/<task>/<seed>`
- **WandB Group**: `phase7_finetuning`
</details>

---

### Phase 6: Epoch Sweep (Baseline vs Flow)
**Method**: **Joint Training (From Scratch)**
**Pretrained Checkpoint**: **None** (Random Initialization)

| Job ID | Status | Config | Epochs | Task/Seed | Methodology | Notes |
|--------|--------|--------|--------|-----------|-------------|-------|
| `4012555` | 🟢 RUNNING | `pwm_48M_mt_baseline` | 150,000 | Array 0-8 | `finetune_wm=True` | Baseline. H200. Long horizon test. |
| `4012556` | ⏳ QUEUED | `pwm_48M_mt_fullflow` | 150,000 | Array 0-8 | `finetune_wm=True` | Flow. H200. High Precision (`substeps=8/4`). |
| `4012537` | 🟢 RUNNING | `pwm_48M_mt_baseline` | 100,000 | Array 0-8 | `finetune_wm=True` | Baseline. H200. |
| `4012538` | 🟢 RUNNING | `pwm_48M_mt_fullflow` | 100,000 | Array 0-8 | `finetune_wm=True` | Flow. H200. High Precision (`substeps=8/4`). |
| `4012535` | ⏳ QUEUED | `pwm_48M_mt_baseline` | 50,000 | Array 0-8 | `finetune_wm=True` | Baseline. H100. |
| `4012536` | ⏳ QUEUED | `pwm_48M_mt_fullflow` | 50,000 | Array 0-8 | `finetune_wm=True` | Flow. H100. High Precision. |
| `4012533` | 🟢 RUNNING | `pwm_48M_mt_baseline` | 15,000 | Array 0-8 | `finetune_wm=True` | Baseline. H100. Control group for Short Horizon. |
| `4012534` | ⏳ QUEUED | `pwm_48M_mt_fullflow` | 15,000 | Array 0-8 | `finetune_wm=True` | Flow. H100. |

<details>
<summary>Job Detail Specifications (Phase 6)</summary>

- **Tasks**: `reacher-easy`, `walker-stand`, `cheetah-run`
- **Seeds**: 42, 123, 456
- **Common Params**: `wm_batch_size=256` (Aligned with Original PWM)
- **Location**: `outputs/epoch_sweep/<variant>_<epochs>/<job_id>/`
- **WandB Group**: `epoch_sweep_<epochs>k_<variant>`
</details>

---

### Phase 5: Flow Tuning (Hyperparameters)
**Method**: **Joint Training (From Scratch)**
**Pretrained Checkpoint**: **None**

| Job ID | Status | Config | Task/Seed | Methodology | Notes |
|--------|--------|--------|-----------|-------------|-------|
| `4012434` | 🟢 RUNNING | `pwm_48M_mt_fullflow` | Array 0-17 | `finetune_wm=True` | Testing `substeps` (4 vs 8) and `integrator` (Euler vs Heun). 15k Epochs. |

<details>
<summary>Job Detail Specifications (Phase 5)</summary>

- **Variants**: High Prec WM, High Prec Policy, Euler Fast
- **Location**: `outputs/mt30_tuning/<task>/<config>/<seed>`
- **WandB Group**: `mt30_tuning_<config>`
</details>

---

## ✅ Completed Phases

### Phase 4: Full Flow Training (Attempt 12)
**Method**: **Joint Training (From Scratch)**
**Pretrained Checkpoint**: **None**

| Job ID | Status | Config | Epochs | Results | Notes |
|--------|--------|--------|--------|---------|-------|
| `4012433` | ✅ COMPLETED | `pwm_48M_mt_fullflow` | 10,000 | Reward ~112 | Severely Undertrained. 10k epochs insufficient for From-Scratch learning. |

### Phase 3: Baseline vs Flow Policy
**Method**: **Load Pretrained Weight**
**Pretrained Checkpoint**: `/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/checkpoints/multitask/mt30_48M_4900000.pt`

| Job ID | Algo | Task | Seed | Reward | Checkpoint Location |
|---|---|---|---|---|---|
| `4011713` | Baseline | reacher-easy | 42 | 981.20 | `outputs/mt30/baseline/reacher-easy/seed42` |
| `4011713` | Baseline | reacher-easy | 123 | 983.50 | `outputs/mt30/baseline/reacher-easy/seed123` |
| `4011713` | Baseline | reacher-easy | 456 | 982.20 | `outputs/mt30/baseline/reacher-easy/seed456` |
| `4011713` | Baseline | walker-stand | 42 | 972.32 | `outputs/mt30/baseline/walker-stand/seed42` |
| `4011713` | Baseline | walker-stand | 123 | 923.48 | `outputs/mt30/baseline/walker-stand/seed123` |
| `4011713` | Baseline | walker-stand | 456 | 977.35 | `outputs/mt30/baseline/walker-stand/seed456` |
| `4011713` | Baseline | cheetah-run | 42 | 93.69 | `outputs/mt30/baseline/cheetah-run/seed42` |
| `4011713` | Baseline | cheetah-run | 123 | 108.80 | `outputs/mt30/baseline/cheetah-run/seed123` |
| `4011713` | Baseline | cheetah-run | 456 | 134.97 | `outputs/mt30/baseline/cheetah-run/seed456` |
| `4011714` | Flow | reacher-easy | 42 | 976.70 | `outputs/mt30/flow_policy/reacher-easy/seed42` |
| `4011714` | Flow | reacher-easy | 123 | 983.40 | `outputs/mt30/flow_policy/reacher-easy/seed123` |
| `4011714` | Flow | reacher-easy | 456 | 981.90 | `outputs/mt30/flow_policy/reacher-easy/seed456` |
| `4011740` | Flow | walker-stand | 42 | 854.53 | `outputs/mt30/flow_policy/walker-stand/seed42` |
| `4011714` | Flow | walker-stand | 123 | 744.92 | `outputs/mt30/flow_policy/walker-stand/seed123` |
| `4011740` | Flow | walker-stand | 456 | 919.90 | `outputs/mt30/flow_policy/walker-stand/seed456` |
| `4011740` | Flow | cheetah-run | 42 | 80.97 | `outputs/mt30/flow_policy/cheetah-run/seed42` |
| `4011740` | Flow | cheetah-run | 123 | 94.75 | `outputs/mt30/flow_policy/cheetah-run/seed123` |
| `4011740` | Flow | cheetah-run | 456 | 120.52 | `outputs/mt30/flow_policy/cheetah-run/seed456` |

---

## 📂 Archived Experiments

<details>
<summary>View Previous Attempts (Attempts 1-14)</summary>

### Attempt 14: Flow Tuning (Jan 04)
- **Job ID**: `4011988`
- **Method**: Joint Training (From Scratch)
- **Status**: ❌ **FAILED** (Storage Full)

### Attempt 13: Baseline From Scratch (Jan 04)
- **Job ID**: `4011987`
- **Method**: Joint Training (From Scratch)
- **Status**: ❌ **FAILED** (Undertrained)

### Attempt 12 & 11: Full Flow & Debug (Jan 04)
- **Job IDs**: `4012027`, `4012028`
- **Method**: Joint Training (From Scratch)
- **Status**: ❌ **FAILED** (Storage Full)

### Attempt 8 & 9: Phase 3 Production (Jan 03)
- **Job IDs**: `4011713`, `4011714`, `4011740`
- **Method**: Load Pretrained Weight
- **Status**: ✅ **COMPLETED**

</details>

---

## 🛠 Resource & Config Reference

### Training Methodologies
| Type | Checkpoint Strategy | `finetune_wm` | Purpose |
|------|---------------------|---------------|---------|
| **Policy Fine-tuning** | **Load Pretrained** | `False` | Isolate Policy (Phase 3) |
| **Policy + WM Tuning** | **Load Pretrained** | `True` | Adapt WM to Policy (Phase 7) |
| **Joint Training** | **None (From Scratch)** | `True` | Train WM+Policy (Phase 4/6) |

### Checkpoint Locations
| Type | Path |
|------|------|
| **Pretrained WM** | `checkpoints/multitask/mt30_48M_4900000.pt` |
| **MT30 Data** | `/home/hice1/eliu354/scratch/Data/tdmpc2/mt30/` |
| **Output** | `outputs/<phase>/<job_id>/` |
