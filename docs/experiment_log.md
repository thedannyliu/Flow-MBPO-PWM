# Experiment Log

> **Purpose**: This file serves as a **persistent experiment registry** for all training, evaluation, and analysis jobs.
> - **Never delete old entries** - append new entries at the top (newest first)
> - Each entry tracks: Job ID, status, configuration, metrics, checkpoint paths, and WandB links
> - Status lifecycle: `PENDING` → `QUEUED` → `RUNNING` → `COMPLETED` / `FAILED` → `EVALUATED`
>
> **Important fields to track (each run/job)**:
> - `Job ID`: Slurm job ID for tracking
> - `Config`: Algorithm config file used
> - `Task`: Task name (e.g., `reacher-easy`)
> - `Seed`: Random seed (42, 123, 456)
> - `Checkpoint`: Path to saved model checkpoint
> - `Runtime`: Total training time
> - `Hardware`: GPU used

---

## 🟢 Active Experiments

### Phase 8: WM Pretraining (2×2 Factorial)
**Method**: Pretrain WM From Scratch
| Job ID | WM Type | Config | Iters | Hardware | Status | Runtime |
|--------|---------|--------|-------|----------|--------|---------|
| `4012664` | Flow WM | `pwm_48M_mt_flowwm` | 200k | H100 | ⏳ QUEUED | - |
| `4012665` | MLP WM | `pwm_48M_mt_baseline` | 200k | H100 | ⏳ QUEUED | - |

---

### Phase 7: Flow Policy Fine-tuning
**Method**: Load Pretrained MLP WM + `finetune_wm=True`
**Checkpoint**: `checkpoints/multitask/mt30_48M_4900000.pt`

| Job ID | Variant | Task | Seed | Hardware | Status | Runtime |
|--------|---------|------|------|----------|--------|---------|
| `4012601_0` | Baseline | reacher-easy | 42 | H100 | 🟢 RUNNING | ~5min |
| `4012601_1` | Baseline | reacher-easy | 123 | H100 | 🟢 RUNNING | ~3min |
| `4012601_2-26` | Multiple | All | * | H100 | ⏳ QUEUED | - |

---

### Phase 6: Epoch Sweep (From Scratch)
**Method**: Joint Training (`finetune_wm=True`, no checkpoint)

#### 6-D: 100k Epochs (H200)
| Job ID | Config | Task | Seed | Hardware | Status | Runtime |
|--------|--------|------|------|----------|--------|---------|
| `4012537_0` | Baseline | reacher-easy | 42 | H200 | 🟢 RUNNING | ~4h45m |
| `4012537_1` | Baseline | reacher-easy | 123 | H200 | 🟢 RUNNING | ~4h45m |
| `4012537_2` | Baseline | reacher-easy | 456 | H200 | 🟢 RUNNING | ~4h45m |
| `4012537_3` | Baseline | walker-stand | 42 | H200 | 🟢 RUNNING | ~4h45m |
| `4012537_4` | Baseline | walker-stand | 123 | H200 | 🟢 RUNNING | ~4h45m |
| `4012537_5` | Baseline | walker-stand | 456 | H200 | 🟢 RUNNING | ~4h45m |
| `4012537_6` | Baseline | cheetah-run | 42 | H200 | 🟢 RUNNING | ~4h45m |
| `4012537_7` | Baseline | cheetah-run | 123 | H200 | 🟢 RUNNING | ~4h45m |
| `4012537_8` | Baseline | cheetah-run | 456 | H200 | 🟢 RUNNING | ~4h45m |
| `4012538_0` | Full Flow | reacher-easy | 42 | H200 | 🟢 RUNNING | ~4h45m |
| `4012538_2-8` | Full Flow | All | * | H200 | 🟢 RUNNING | ~4h45m |

#### 6-C: 50k Epochs (H100)
| Job ID | Config | Task | Seed | Hardware | Status | Runtime | Final Reward |
|--------|--------|------|------|----------|--------|---------|--------------|
| `4012535_0` | Baseline | reacher-easy | 42 | H100 | ✅ COMPLETED | 2h36m | 130.20 |
| `4012535_1` | Baseline | reacher-easy | 123 | H100 | ✅ COMPLETED | 2h32m | 1.00 |
| `4012535_2` | Baseline | reacher-easy | 456 | H100 | ✅ COMPLETED | 2h34m | 188.60 |
| `4012535_3` | Baseline | walker-stand | 42 | H100 | ✅ COMPLETED | 2h38m | 93.63 |
| `4012535_4` | Baseline | walker-stand | 123 | H100 | ✅ COMPLETED | 2h37m | 136.76 |
| `4012535_5` | Baseline | walker-stand | 456 | H100 | ✅ COMPLETED | 2h37m | 147.34 |
| `4012535_6` | Baseline | cheetah-run | 42 | H100 | ✅ COMPLETED | 2h34m | 44.35 |
| `4012535_7` | Baseline | cheetah-run | 123 | H100 | ✅ COMPLETED | 2h33m | 0.64 |
| `4012535_8` | Baseline | cheetah-run | 456 | H100 | 🟢 RUNNING | ~20min | - |
| `4012536_0-8` | Full Flow | All | * | H100 | 🟢 RUNNING | ~18min | - |

#### 6-B: 15k Epochs (H100)
| Job ID | Config | Task | Seed | Hardware | Status | Runtime | Final Reward |
|--------|--------|------|------|----------|--------|---------|--------------|
| `4012533_0` | Baseline | reacher-easy | 42 | H100 | ✅ COMPLETED | 49m | 54.00 |
| `4012533_1` | Baseline | reacher-easy | 123 | H100 | ✅ COMPLETED | 51m | 153.40 |
| `4012533_2` | Baseline | reacher-easy | 456 | H100 | ✅ COMPLETED | 50m | 3.30 |
| `4012533_3` | Baseline | walker-stand | 42 | H100 | ✅ COMPLETED | 49m | 139.98 |
| `4012533_4` | Baseline | walker-stand | 123 | H100 | ✅ COMPLETED | 50m | 156.05 |
| `4012533_5` | Baseline | walker-stand | 456 | H100 | ✅ COMPLETED | 50m | 151.09 |
| `4012533_6` | Baseline | cheetah-run | 42 | H100 | ✅ COMPLETED | 48m | 0.19 |
| `4012533_7` | Baseline | cheetah-run | 123 | H100 | ✅ COMPLETED | 48m | 0.14 |
| `4012533_8` | Baseline | cheetah-run | 456 | H100 | ✅ COMPLETED | 48m | ~0.2 |
| `4012534_0` | Full Flow | reacher-easy | 42 | H100 | 🟢 RUNNING | ~3h24m | - |
| `4012534_1` | Full Flow | reacher-easy | 123 | H100 | ✅ COMPLETED | 3h09m | 147.20 |
| `4012534_2` | Full Flow | reacher-easy | 456 | H100 | ✅ COMPLETED | 3h09m | 33.20 |
| `4012534_3` | Full Flow | walker-stand | 42 | H100 | ✅ COMPLETED | 3h08m | 142.36 |
| `4012534_4` | Full Flow | walker-stand | 123 | H100 | ✅ COMPLETED | 3h05m | 153.06 |
| `4012534_5` | Full Flow | walker-stand | 456 | H100 | 🟢 RUNNING | ~3h09m | 156.10 |
| `4012534_6-8` | Full Flow | cheetah-run | * | H100 | 🟢 RUNNING | ~2h53m | - |

---

## ✅ Completed Phases

### Phase 5: Flow Tuning (15k epochs, From Scratch)
**Method**: Joint Training (`finetune_wm=True`)

| Job ID | Config Variant | Task | Seed | Hardware | Runtime | Final Reward |
|--------|----------------|------|------|----------|---------|--------------|
| `4012434_0` | high_precision_wm | walker-stand | 42 | H100 | 2h59m | 142.26 |
| `4012434_1` | high_precision_wm | walker-stand | 123 | H100 | 2h59m | 117.88 |
| `4012434_2` | high_precision_wm | walker-stand | 456 | H100 | 2h56m | 156.10 |
| `4012434_3` | high_precision_policy | walker-stand | 42 | H100 | 2h07m | 142.25 |
| `4012434_4` | high_precision_policy | walker-stand | 123 | H100 | 2h06m | 156.06 |
| `4012434_5` | high_precision_policy | walker-stand | 456 | H100 | 2h07m | 146.24 |
| `4012434_6` | euler_fast | walker-stand | 42 | H100 | 1h19m | 142.26 |
| `4012434_7` | euler_fast | walker-stand | 123 | H100 | 1h20m | 156.07 |
| `4012434_8` | euler_fast | walker-stand | 456 | H100 | 1h19m | 156.10 |
| `4012434_9` | high_precision_wm | cheetah-run | 42 | H100 | 2h55m | 0.27 |
| `4012434_10` | high_precision_wm | cheetah-run | 123 | H100 | 2h54m | 0.19 |
| `4012434_11` | high_precision_wm | cheetah-run | 456 | H100 | 2h55m | 0.24 |
| `4012434_12` | high_precision_policy | cheetah-run | 42 | H100 | 2h07m | 0.21 |
| `4012434_13` | high_precision_policy | cheetah-run | 123 | H100 | 2h06m | 0.19 |
| `4012434_14` | high_precision_policy | cheetah-run | 456 | H100 | 2h17m | 0.16 |
| `4012434_15` | euler_fast | cheetah-run | 42 | H100 | 1h20m | 0.21 |
| `4012434_16` | euler_fast | cheetah-run | 123 | H100 | 1h19m | 4.08 |
| `4012434_17` | euler_fast | cheetah-run | 456 | H100 | 1h19m | 0.16 |

---

### Phase 4: Full Flow Training (10k epochs, From Scratch)
**Method**: Joint Training (`finetune_wm=True`)

| Job ID | Task | Seed | Hardware | Runtime | Final Reward |
|--------|------|------|----------|---------|--------------|
| `4012433_0` | reacher-easy | 42 | H100 | 1h19m | 112.20 |
| `4012433_1` | reacher-easy | 123 | H100 | 1h18m | 39.10 |
| `4012433_2` | reacher-easy | 456 | H100 | 1h19m | 104.50 |
| `4012433_3` | walker-stand | 42 | H100 | 1h26m | 141.68 |
| `4012433_4` | walker-stand | 123 | H100 | 1h19m | 140.52 |
| `4012433_5` | walker-stand | 456 | H100 | 1h19m | 127.64 |
| `4012433_6` | cheetah-run | 42 | H100 | 1h18m | 0.16 |
| `4012433_7` | cheetah-run | 123 | H100 | 1h18m | 1.40 |
| `4012433_8` | cheetah-run | 456 | H100 | 1h19m | ~0.2 |

---

### Phase 3: Baseline vs Flow Policy (Pretrained WM, Frozen)
**Method**: Load Pretrained MLP WM + Policy Training (`finetune_wm=False`)
**Checkpoint**: `checkpoints/multitask/mt30_48M_4900000.pt`

| Job ID | Algo | Task | Seed | Hardware | Final Reward |
|--------|------|------|------|----------|--------------|
| `4011713_0` | Baseline | reacher-easy | 42 | H100 | 981.20 |
| `4011713_1` | Baseline | reacher-easy | 123 | H100 | 983.50 |
| `4011713_2` | Baseline | reacher-easy | 456 | H100 | 982.20 |
| `4011713_3` | Baseline | walker-stand | 42 | H100 | 972.32 |
| `4011713_4` | Baseline | walker-stand | 123 | H100 | 923.48 |
| `4011713_5` | Baseline | walker-stand | 456 | H100 | 977.35 |
| `4011713_6` | Baseline | cheetah-run | 42 | H100 | 93.69 |
| `4011713_7` | Baseline | cheetah-run | 123 | H100 | 108.80 |
| `4011713_8` | Baseline | cheetah-run | 456 | H100 | 134.97 |
| `4011714_0` | Flow Policy | reacher-easy | 42 | H100 | 976.70 |
| `4011714_1` | Flow Policy | reacher-easy | 123 | H100 | 983.40 |
| `4011714_2` | Flow Policy | reacher-easy | 456 | H100 | 981.90 |
| `4011714/40_0` | Flow Policy | walker-stand | 42 | H100 | 854.53 |
| `4011714_1` | Flow Policy | walker-stand | 123 | H100 | 744.92 |
| `4011740_2` | Flow Policy | walker-stand | 456 | H100 | 919.90 |
| `4011740_3` | Flow Policy | cheetah-run | 42 | H100 | 80.97 |
| `4011740_4` | Flow Policy | cheetah-run | 123 | H100 | 94.75 |
| `4011740_5` | Flow Policy | cheetah-run | 456 | H100 | 120.52 |

---

## 📂 Archived/Failed Experiments

| Job ID | Name | Status | Reason | Hardware |
|--------|------|--------|--------|----------|
| `4011988_0-17` | Flow Tuning | ❌ FAILED | Storage Full | H100 |
| `4012027_0-8` | Full Flow Debug | ❌ FAILED | Quick Exit (~1-3min) | H100 |
| `4012028_0-2` | Cheetah Debug | ❌ FAILED | Quick Exit | H100 |

---

## 🛠 Resource & Config Reference

### Config Alignment (Original PWM)
| Parameter | Value | Source |
|-----------|-------|--------|
| `horizon` | 16 | Original PWM README |
| `batch_size` (WM pretrain) | 1024 | Original PWM README |
| `wm_batch_size` (policy) | 256 | `pwm_48M.yaml` |

### Checkpoint Locations
| Type | Path |
|------|------|
| **Original PWM** | `checkpoints/multitask/mt30_48M_4900000.pt` |
| **Output** | `outputs/<phase>/<job_id>/` |
