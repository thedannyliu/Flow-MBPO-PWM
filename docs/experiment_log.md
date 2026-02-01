# Experiment Log

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

