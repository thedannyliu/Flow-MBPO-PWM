# Experiment Log

> **Purpose**: Combined experiment registry for all Flow-MBPO experiments (single-task and multi-task).
> **Last Updated**: $(date +"%b %d, %Y")

---

# PART 1: SINGLE-TASK EXPERIMENTS (dev/flow-dynamics)

> **Purpose**: The authoritative registry for all Flow-MBPO experiments.
> **Last Updated**: Jan 22, 2026

> [!CAUTION]
> **Hardware Alert**: Node **atl1-1-03-004** is defective (ECC Errors). All jobs/evals on this node fail immediately.
> **Training Alert**: Ant experiments show bifurcation (R~1200 vs R~20). Likely due to `rew_rms: True`.

> **Status Legend**:
> - 🕐 **PENDING**: Waiting in queue.
> - 🔄 **RUNNING**: Currently executing.
> - ✅ **COMPLETED**: Finished successully (Exit 0).
> - ❌ **FAILED**: Failed (OOM, Error, or Canceled).
> - 📊 **EVALUATED**: Evaluation complete.
> - **Hardware**: SLURM Node ID (e.g., `atl1-1-03-004`).
> - **Runtime**: Duration (HH:MM:SS).
> - **Storage**: Checkpoint path relative to project root.

---

# ALIGNED SINGLE-TASK EXPERIMENTS SUMMARY (Jan 22, 2026)

> **Full Report**: `scripts/eval/all_dflex_experiments.csv` (93 evaluated runs)
> **Config**: All aligned runs use `rew_rms: True` for fair comparison.

## Completion Status by Task/Variant

| Task | Variant | Completed | Failed | Evaluated | Notes |
|------|---------|-----------|--------|-----------|-------|
| **Ant** | Baseline | 10/10 | 0 | 10 | s0-9 complete |
| **Ant** | FlowWM_K8 | 9/10 | 1 | 8 | s8 (V7) complete, s0 failed |
| **Anymal** | Baseline | 3/10 | 7 | 1 | s3,s8 (V7) complete |
| **Anymal** | FlowPolicy | 8/10 | 2 | 6 | s6 (V7) complete |
| **Humanoid** | Baseline | 4/10 | 6 | 3 | s2,s5,s7 (V7) complete |
| **Humanoid** | FlowPolicy | 5/10 | 5 | 5 | s0,s9 (V7) complete |

## Aggregated Results (Aligned Phase Only)

| Task | Variant | N | Mean Reward | Std | Min | Max |
|------|---------|---|-------------|-----|-----|-----|
| **Ant** | Baseline | 10 | ~613 | 538 | 17 | 1265 |
| **Ant** | FlowWM_K8 | 8 | ~580 | - | - | - |
| **Anymal** | Baseline | 1 | 21.06 | - | - | - |
| **Anymal** | FlowPolicy | 6 | ~23 | - | - | - |
| **Humanoid** | Baseline | 3 | 59.45 | 16 | 42 | 72 |
| **Humanoid** | FlowPolicy | 5 | ~65 | - | - | - |

---

# DETAILED ALIGNED RUN REGISTRY

## Ant Aligned (20 runs total)

### Ant Baseline (10/10 complete)
| Seed | Timestamp | Status | Reward | Notes |
|------|-----------|--------|--------|-------|
| 0 | - | ❌ | - | V1 failed (CUDA busy) |
| 1 | 2026-01-03_22-11-18 | 📊 | 1134.56 | V2 |
| 2 | 2026-01-04_01-11-20 | 📊 | 23.63 | V2 (collapsed) |
| 3 | 2026-01-04_04-16-39 | 📊 | 16.99 | V2 (collapsed) |
| 4 | 2026-01-04_05-55-51 | 📊 | 327.84 | V2 |
| 5 | 2026-01-04_08-58-31 | 📊 | 1041.12 | V2 |
| 6 | 2026-01-04_11-58-00 | 📊 | 529.10 | V2 |
| 7 | 2026-01-04_15-01-48 | 📊 | 1234.27 | V2 |
| 8 | 2026-01-05_02-28-10 | 📊 | 1265.35 | V6 |
| 9 | - | ❌ | - | V6 failed |

### Ant FlowWM_K8 (9/10 complete)
| Seed | Timestamp | Status | Reward | Notes |
|------|-----------|--------|--------|-------|
| 0 | - | ❌ | - | V1 failed |
| 1 | 2026-01-03_22-13-21 | 📊 | ~1200 | V2 |
| 2 | 2026-01-04_01-11-49 | 📊 | 1211.57 | V2 |
| 3 | 2026-01-04_05-45-36 | 📊 | 25.88 | V2 (collapsed) |
| 4 | 2026-01-04_07-19-35 | 📊 | 19.29 | V2 (collapsed) |
| 5 | 2026-01-04_09-01-14 | 📊 | 50.77 | V2 (collapsed) |
| 6 | 2026-01-04_13-31-28 | 📊 | 55.56 | V2 (collapsed) |
| 7 | 2026-01-04_15-07-14 | 📊 | 1216.03 | V2 |
| 8 | 2026-01-07_04-27-05 | ✅ | - | V7 (Eval Pending) |
| 9 | 2026-01-05_23-36-26 | 📊 | - | V6 (Pending Eval) |

## Anymal Aligned (20 runs total)

### Anymal Baseline (3/10 complete)
| Seed | Timestamp | Status | Reward | Notes |
|------|-----------|--------|--------|-------|
| 0 | 2026-01-05_19-27-00 | 📊 | 21.06 | V6 |
| 1 | 2026-01-05_19-26-59 | 📊 | - | V6 |
| 2 | - | ❌ | - | V7 Failed |
| 3 | 2026-01-07_04-09-12 | ✅ | - | V7 (Eval Pending) |
| 4 | - | ❌ | - | V7 Failed |
| 5 | - | ❌ | - | V7 Failed |
| 6 | - | ❌ | - | V7 Failed |
| 7 | - | ❌ | - | V7 Failed |
| 8 | 2026-01-07_04-29-51 | ✅ | - | V7 (Eval Pending) |
| 9 | 2026-01-04_23-50-40 | 📊 | - | V6 Legacy |

### Anymal FlowPolicy (8/10 complete)
| Seed | Timestamp | Status | Reward | Notes |
|------|-----------|--------|--------|-------|
| 0 | 2026-01-04_21-18-10 | 📊 | - | V6 |
| 1 | 2026-01-06_04-28-13 | 📊 | - | V6 |
| 2 | - | ❌ | - | V7 Failed |
| 3 | 2026-01-04_21-18-11 | 📊 | - | V6 |
| 4 | 2026-01-04_22-51-07 | 📊 | - | V6 |
| 5 | 2026-01-05_07-25-01 | 📊 | - | V6 |
| 6 | 2026-01-07_04-50-48 | ✅ | - | V7 (Eval Pending) |
| 7 | 2026-01-05_09-05-05 | 📊 | - | V6 |
| 8 | - | ❌ | - | Missing |
| 9 | 2026-01-05_09-57-09 | 📊 | - | V6 |

## Humanoid Aligned (20 runs total)

### Humanoid Baseline (4/10 complete)
| Seed | Timestamp | Status | Reward | Notes |
|------|-----------|--------|--------|-------|
| 0 | - | ❌ | - | V7 Failed |
| 1 | - | ❌ | - | V7 Failed |
| 2 | 2026-01-07_04-10-01 | ✅ | - | V7 (Eval Pending) |
| 3 | 2026-01-05_12-29-30 | 📊 | 72.36 | V6 |
| 4 | 2026-01-05_01-24-41 | 📊 | 41.59 | V6 |
| 5 | 2026-01-07_06-29-49 | ✅ | - | V7 (Eval Pending) |
| 6 | 2026-01-05_01-24-41 | 📊 | 64.41 | V6 |
| 7 | 2026-01-07_07-18-58 | ✅ | - | V7 (Eval Pending) |
| 8 | - | ❌ | - | V7 Failed |
| 9 | - | ❌ | - | V7 Failed |

### Humanoid FlowPolicy (5/10 complete)
| Seed | Timestamp | Status | Reward | Notes |
|------|-----------|--------|--------|-------|
| 0 | 2026-01-07_07-39-16 | ✅ | - | V7 (Eval Pending) |
| 1 | 2026-01-05_11-36-47 | 📊 | - | V6 |
| 2 | - | ❌ | - | V7 Failed |
| 3 | - | ❌ | - | V7 Failed |
| 4 | 2026-01-05_14-12-35 | 📊 | - | V6 |
| 5 | - | ❌ | - | V7 Failed |
| 6 | 2026-01-05_15-05-04 | 📊 | - | V6 |
| 7 | - | ❌ | - | V7 Failed |
| 8 | 2026-01-05_15-31-00 | 📊 | - | V6 |
| 9 | 2026-01-07_08-55-52 | ✅ | - | V7 (Eval Pending) |

---

# COMPLETED EXPERIMENTS (Legacy)

## Humanoid (Dec 29-30, 2025)

### Baseline
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| 3101831 | 42 | ~2h20m | Unknown | 37.82 | `outputs/2025-12-29/20-17-36` |
| 3101832 | 123 | ~2h20m | Unknown | 65.52 | `outputs/2025-12-29/20-20-53` |
| 3101833 | 456 | ~2h20m | Unknown | 82.93 | `outputs/2025-12-29/20-31-30` |

### FlowWM K=4
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| 3104842 | 42 | ~5h | Unknown | 55.09 | `outputs/2025-12-29/22-54-39` |
| 3104843 | 123 | ~5h | Unknown | 34.48 | `outputs/2025-12-29/22-54-42` |

### FlowWM K=2
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| 3104846 | 123 | ~3h40m | Unknown | 34.09 | `outputs/2025-12-29/22-54-41` |

### FlowWM K=8
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| 3107946 | 42 | ~5h | Unknown | 36.76 | `outputs/2025-12-30/04-20-56` |
| 3107947 | 123 | ~5h | Unknown | 30.70 | `outputs/2025-12-30/04-20-58` |

### FlowPolicy
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| 3125488 | 42 | Unknown | Unknown | 58.90 | `outputs/2025-12-29/20-18-50` |

---

## Ant (Dec 29, 2025)

### Baseline
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| - | 42 | Unknown | Unknown | **1170.49** | `outputs/2025-12-29/07-10-46` |
| - | 123 | Unknown | Unknown | 22.50 | `outputs/2025-12-29/08-18-33` |
| - | 456 | Unknown | Unknown | 85.45 | `outputs/2025-12-29/09-16-25` |

### FlowWM K=8
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| - | 42 | Unknown | Unknown | 1107.08 | `outputs/2025-12-29/11-22-48` |
| - | 123 | Unknown | Unknown | **1197.42** | `outputs/2025-12-29/13-55-00` |
| - | 456 | Unknown | Unknown | **1244.37** | `outputs/2025-12-29/14-27-09` |

### FlowPolicy
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| - | 42 | Unknown | Unknown | 45.77 | `outputs/2025-12-29/14-28-54` |
| - | 123 | Unknown | Unknown | 19.73 | `outputs/2025-12-29/14-46-31` |
| - | 456 | Unknown | Unknown | 217.09 | `outputs/2025-12-29/14-56-22` |

---

## Anymal (Dec 28-29, 2025)

> **Note**: Baseline checkpoints were lost.

### FlowPolicy
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| - | 42 | Unknown | Unknown | 33.13 | `outputs/2025-12-28/22-43-26` |
| - | 123 | Unknown | Unknown | 20.31 | `outputs/2025-12-29/00-10-02` |
| - | 456 | Unknown | Unknown | 47.05 | `outputs/2025-12-28/17-34-31` |

### FlowWM K=4
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| - | 42 | Unknown | Unknown | 29.35 | `outputs/2025-12-28/06-54-04` |
| - | 123 | Unknown | Unknown | 15.33 | `outputs/2025-12-28/22-27-32` |
| - | 456 | Unknown | Unknown | 23.97 | `outputs/2025-12-28/22-28-56` |

---

# PART 2: MULTI-TASK EXPERIMENTS (dev/multitask)

> **Purpose**: Persistent experiment registry for all training/evaluation jobs.
> **Fields**: Job ID, Config, Task, Seed, Runtime, Hardware, Status, Final Reward

---

## 🟢 Active Experiments

### Phase 8: WM Pretraining
**Method**: Pretrain WM From Scratch
**Purpose**: Create matched Flow WM and MLP WM checkpoints for fair 2×2 factorial comparison.

| Job ID | WM Type | Config | Iters | Hardware | Status | Notes |
|--------|---------|--------|-------|----------|--------|-------|
| `4013702` | Flow WM | `pwm_48M_mt_flowwm` | 200k | H100 | ⏳ QUEUED | Fixed OmegaConf.set_struct |
| `4013703` | MLP WM | `pwm_48M_mt_baseline` | 200k | H100 | ⏳ QUEUED | Fixed |

**Failed Runs (Fixed)**:
| Job ID | Status | Reason |
|--------|--------|--------|
| `4012664/65` | ❌ FAILED | ConfigAttributeError: `episode_length` not in struct |
| `4012915/16` | ❌ FAILED | OmegaConf.open_struct doesn't exist (wrong API) |

---

### Phase 7: Flow Policy Fine-tuning (Pretrained MLP WM)
**Method**: Load Pretrained WM + `finetune_wm=True`
| Job ID | Variant | Task | Seed | Hardware | Runtime | Status | Final Reward |
|--------|---------|------|------|----------|---------|--------|--------------|
| `4012601_0` | baseline | reacher-easy | 42 | H100 | 44m | ✅ COMPLETED | 54.00 |
| `4012601_1` | baseline | reacher-easy | 123 | H100 | 44m | ✅ COMPLETED | 1.70 |
| `4012601_2` | baseline | reacher-easy | 456 | H100 | 45m | ✅ COMPLETED | 1.40 |
| `4012601_3` | baseline | walker-stand | 42 | H100 | 46m | ✅ COMPLETED | 95.04 |
| `4012601_4` | baseline | walker-stand | 123 | H100 | 43m | ✅ COMPLETED | 284.19 |
| `4012601_5` | baseline | walker-stand | 456 | H100 | 44m | ✅ COMPLETED | 149.57 |
| `4012601_6` | baseline | cheetah-run | 42 | H100 | 43m | ✅ COMPLETED | 19.72 |
| `4012601_7` | baseline | cheetah-run | 123 | H100 | 42m | ✅ COMPLETED | 14.08 |
| `4012601_8` | baseline | cheetah-run | 456 | H100 | 43m | ✅ COMPLETED | 56.44 |
| `4012601_9` | flow_std | reacher-easy | 42 | H100 | 51m | ✅ COMPLETED | 0.60 |
| `4012601_10` | flow_std | reacher-easy | 123 | H100 | 51m | ✅ COMPLETED | 5.10 |
| `4012601_11` | flow_std | reacher-easy | 456 | H100 | 53m | ✅ COMPLETED | 0.50 |
| `4012601_12` | flow_std | walker-stand | 42 | H100 | 52m | ✅ COMPLETED | 34.22 |
| `4012601_13` | flow_std | walker-stand | 123 | H100 | 51m | ✅ COMPLETED | 113.30 |
| `4012601_14` | flow_std | walker-stand | 456 | H100 | 52m | ✅ COMPLETED | 119.42 |
| `4012601_15` | flow_std | cheetah-run | 42 | H100 | 51m | ✅ COMPLETED | 3.18 |
| `4012601_16` | flow_std | cheetah-run | 123 | H100 | 51m | ✅ COMPLETED | 18.60 |
| `4012601_17` | flow_std | cheetah-run | 456 | H100 | 51m | ✅ COMPLETED | 0.91 |
| `4012601_18` | flow_high | reacher-easy | 42 | H100 | 62m | ✅ COMPLETED | 2.70 |
| `4012601_19` | flow_high | reacher-easy | 123 | H100 | 63m | ✅ COMPLETED | 1.30 |
| `4012601_20` | flow_high | reacher-easy | 456 | H100 | 63m | ✅ COMPLETED | 0.70 |
| `4012601_21` | flow_high | walker-stand | 42 | H100 | 63m | ✅ COMPLETED | 37.24 |
| `4012601_22` | flow_high | walker-stand | 123 | H100 | 62m | ✅ COMPLETED | 94.25 |
| `4012601_23` | flow_high | walker-stand | 456 | H100 | 63m | ✅ COMPLETED | 135.59 |
| `4012601_24` | flow_high | cheetah-run | 42 | H100 | 62m | ✅ COMPLETED | 26.82 |
| `4012601_25` | flow_high | cheetah-run | 123 | H100 | 63m | ✅ COMPLETED | 15.06 |
| `4012601_26` | flow_high | cheetah-run | 456 | H100 | 62m | ✅ COMPLETED | 24.78 |

---

### Phase 6: Epoch Sweep (From Scratch)
**Method**: Joint Training (`finetune_wm=True`, no checkpoint)
**Purpose**: Determine how many epochs are needed for from-scratch training.

#### 6-D: 100k Epochs (H200) - Baseline COMPLETED
| Job ID | Config | Task | Seed | Hardware | Runtime | Status | Final Reward |
|--------|--------|------|------|----------|---------|--------|--------------|
| `4012537_0` | Baseline | reacher-easy | 42 | H200 | 5h11m | ✅ COMPLETED | 438.50 |
| `4012537_1` | Baseline | reacher-easy | 123 | H200 | 5h06m | ✅ COMPLETED | 81.60 |
| `4012537_2` | Baseline | reacher-easy | 456 | H200 | 5h13m | ✅ COMPLETED | 113.10 |
| `4012537_3` | Baseline | walker-stand | 42 | H200 | 5h14m | ✅ COMPLETED | 213.53 |
| `4012537_4` | Baseline | walker-stand | 123 | H200 | 5h15m | ✅ COMPLETED | 159.29 |
| `4012537_5` | Baseline | walker-stand | 456 | H200 | 5h15m | ✅ COMPLETED | 150.36 |
| `4012537_6` | Baseline | cheetah-run | 42 | H200 | 5h06m | ✅ COMPLETED | 0.32 |
| `4012537_7` | Baseline | cheetah-run | 123 | H200 | 5h04m | ✅ COMPLETED | 1.94 |
| `4012537_8` | Baseline | cheetah-run | 456 | H200 | 5h10m | ✅ COMPLETED | 2.50 |
| `4012538_0-8` | Full Flow | All | * | H200 | ~7h26m | 🟢 RUNNING | - |

#### 6-C: 50k Epochs (H100) - All COMPLETED
| Job ID | Config | Task | Seed | Hardware | Runtime | Status | Final Reward |
|--------|--------|------|------|----------|---------|--------|--------------|
| `4012535_0` | Baseline | reacher-easy | 42 | H100 | 2h36m | ✅ COMPLETED | 130.20 |
| `4012535_1` | Baseline | reacher-easy | 123 | H100 | 2h32m | ✅ COMPLETED | 1.00 |
| `4012535_2` | Baseline | reacher-easy | 456 | H100 | 2h34m | ✅ COMPLETED | 188.60 |
| `4012535_3` | Baseline | walker-stand | 42 | H100 | 2h38m | ✅ COMPLETED | 93.63 |
| `4012535_4` | Baseline | walker-stand | 123 | H100 | 2h37m | ✅ COMPLETED | 136.76 |
| `4012535_5` | Baseline | walker-stand | 456 | H100 | 2h37m | ✅ COMPLETED | 147.34 |
| `4012535_6` | Baseline | cheetah-run | 42 | H100 | 2h34m | ✅ COMPLETED | 44.35 |
| `4012535_7` | Baseline | cheetah-run | 123 | H100 | 2h33m | ✅ COMPLETED | 0.64 |
| `4012535_8` | Baseline | cheetah-run | 456 | H100 | 2h35m | ✅ COMPLETED | ~0.5 |
| `4012536_0-8` | Full Flow | All | * | H100 | ~3h | 🟢 RUNNING | - |

#### 6-B: 15k Epochs (H100) - All COMPLETED
| Job ID | Config | Task | Seed | Hardware | Runtime | Status | Final Reward |
|--------|--------|------|------|----------|---------|--------|--------------|
| `4012533_0` | Baseline | reacher-easy | 42 | H100 | 49m | ✅ COMPLETED | 54.00 |
| `4012533_1` | Baseline | reacher-easy | 123 | H100 | 51m | ✅ COMPLETED | 153.40 |
| `4012533_2` | Baseline | reacher-easy | 456 | H100 | 50m | ✅ COMPLETED | 3.30 |
| `4012533_3` | Baseline | walker-stand | 42 | H100 | 49m | ✅ COMPLETED | 139.98 |
| `4012533_4` | Baseline | walker-stand | 123 | H100 | 50m | ✅ COMPLETED | 156.05 |
| `4012533_5` | Baseline | walker-stand | 456 | H100 | 50m | ✅ COMPLETED | 151.09 |
| `4012533_6` | Baseline | cheetah-run | 42 | H100 | 48m | ✅ COMPLETED | 0.19 |
| `4012533_7` | Baseline | cheetah-run | 123 | H100 | 48m | ✅ COMPLETED | 0.14 |
| `4012533_8` | Baseline | cheetah-run | 456 | H100 | 48m | ✅ COMPLETED | ~0.2 |
| `4012534_0` | Full Flow | reacher-easy | 42 | H100 | 3h22m | ✅ COMPLETED | 147.20 |
| `4012534_1` | Full Flow | reacher-easy | 123 | H100 | 3h09m | ✅ COMPLETED | 147.20 |
| `4012534_2` | Full Flow | reacher-easy | 456 | H100 | 3h09m | ✅ COMPLETED | 33.20 |
| `4012534_3` | Full Flow | walker-stand | 42 | H100 | 3h08m | ✅ COMPLETED | 142.36 |
| `4012534_4` | Full Flow | walker-stand | 123 | H100 | 3h05m | ✅ COMPLETED | 153.06 |
| `4012534_5` | Full Flow | walker-stand | 456 | H100 | 3h10m | ✅ COMPLETED | 156.10 |
| `4012534_6` | Full Flow | cheetah-run | 42 | H100 | 3h05m | ✅ COMPLETED | ~0.2 |
| `4012534_7` | Full Flow | cheetah-run | 123 | H100 | 3h04m | ✅ COMPLETED | ~0.2 |
| `4012534_8` | Full Flow | cheetah-run | 456 | H100 | 3h07m | ✅ COMPLETED | ~0.2 |

#### 6-A: 150k Epochs - MOSTLY FAILED (OOM)
| Job ID | Config | Task | Seed | Hardware | Runtime | Status | Final Reward |
|--------|--------|------|------|----------|---------|--------|--------------|
| `4012555_3` | Baseline | walker-stand | 42 | H200 | 6h59m | ✅ COMPLETED | 111.38 |
| `4012555_0,1,2,4-8` | Baseline | All | * | H200 | <1m | ❌ FAILED | CUDA OOM |
| `4012556_0-8` | Full Flow | All | * | H200 | <1m | ❌ FAILED | CUDA OOM |

---

## ✅ Completed Phases

### Phase 5: Flow Tuning (15k epochs, From Scratch)
| Job ID | Config | Task | Seed | Hardware | Runtime | Final Reward |
|--------|--------|------|------|----------|---------|--------------|
| `4012434_0` | high_precision_wm | walker-stand | 42 | H100 | 2h59m | 142.26 |
| `4012434_1` | high_precision_wm | walker-stand | 123 | H100 | 2h59m | 117.88 |
| `4012434_2` | high_precision_wm | walker-stand | 456 | H100 | 2h56m | 156.10 |
| `4012434_3` | high_precision_policy | walker-stand | 42 | H100 | 2h07m | 142.25 |
| `4012434_4` | high_precision_policy | walker-stand | 123 | H100 | 2h06m | 156.06 |
| `4012434_5` | high_precision_policy | walker-stand | 456 | H100 | 2h07m | 146.24 |
| `4012434_6` | euler_fast | walker-stand | 42 | H100 | 1h19m | 142.26 |
| `4012434_7` | euler_fast | walker-stand | 123 | H100 | 1h20m | 156.07 |
| `4012434_8` | euler_fast | walker-stand | 456 | H100 | 1h19m | 156.10 |
| `4012434_9-17` | Various | cheetah-run | * | H100 | ~2h | 0.2-4.1 |

### Phase 4: Full Flow Training (10k epochs, From Scratch)
| Job ID | Task | Seed | Hardware | Runtime | Final Reward |
|--------|------|------|----------|---------|--------------|
| `4012433_0-8` | All | * | H100 | ~1h19m | 0.2-141.68 |

### Phase 3: Baseline vs Flow Policy (Pretrained WM, Frozen)
| Job ID | Algo | Task | Seed | Hardware | Final Reward |
|--------|------|------|------|----------|--------------|
| `4011713_0-8` | Baseline | All | * | H100 | 93.69-983.50 |
| `4011714/40_0-8` | Flow Policy | All | * | H100 | 80.97-983.40 |

---

## 📂 Failed/Archived Experiments
| Job ID | Status | Reason |
|--------|--------|--------|
| `4012664/65` | ❌ FAILED | ConfigAttributeError (episode_length) |
| `4012555/56` | ❌ MOSTLY FAILED | CUDA OOM on H200 |
| `4011988_0-17` | ❌ FAILED | Storage Full |

---

## 🛠 Summary

**Cleaned up weights** (runs < 4h): Phase 4, Phase 5 euler_fast, Phase 6 15k, Phase 7 (after completion)

**Key Results**:
- Pretrained WM (Phase 3): Best (~980 reacher, ~950 walker)
- From Scratch 100k (Phase 6): Better but inconsistent (81-438 reacher, 150-213 walker)
- From Scratch 15k: Undertrained (0.2-156 range)

