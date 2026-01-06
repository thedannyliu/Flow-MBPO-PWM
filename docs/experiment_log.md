# Experiment Log

> **Purpose**: Persistent experiment registry. Fields: Job ID, Config, Task, Seed, Runtime, Hardware, Status, Final Reward

---

## 🟢 Active Experiments

### Phase 8: WM Pretraining (Re-resubmitted, Fixed)
| Job ID | WM Type | Config | Iters | Hardware | Status | Notes |
|--------|---------|--------|-------|----------|--------|-------|
| `4013702` | Flow WM | `pwm_48M_mt_flowwm` | 200k | H100 | ⏳ QUEUED | Fixed OmegaConf.set_struct |
| `4013703` | MLP WM | `pwm_48M_mt_baseline` | 200k | H100 | ⏳ QUEUED | Fixed |

**Failed Runs**:
| Job ID | Status | Reason |
|--------|--------|--------|
| `4012664/65` | ❌ FAILED | ConfigAttributeError |
| `4012915/16` | ❌ FAILED | OmegaConf.open_struct doesn't exist |

---

## ✅ Completed Phases

### Phase 7: Flow Policy Fine-tuning (ALL 27 COMPLETED)
**Method**: Load Pretrained MLP WM + `finetune_wm=True`, 15k epochs
| Job ID | Variant | Task | Seed | Hardware | Runtime | Final Reward |
|--------|---------|------|------|----------|---------|--------------|
| `4012601_0` | baseline | reacher-easy | 42 | H100 | 44m | 54.00 |
| `4012601_1` | baseline | reacher-easy | 123 | H100 | 44m | 1.70 |
| `4012601_2` | baseline | reacher-easy | 456 | H100 | 45m | 1.40 |
| `4012601_3` | baseline | walker-stand | 42 | H100 | 46m | 95.04 |
| `4012601_4` | baseline | walker-stand | 123 | H100 | 43m | 284.19 |
| `4012601_5` | baseline | walker-stand | 456 | H100 | 44m | 149.57 |
| `4012601_6` | baseline | cheetah-run | 42 | H100 | 43m | 19.72 |
| `4012601_7` | baseline | cheetah-run | 123 | H100 | 42m | 14.08 |
| `4012601_8` | baseline | cheetah-run | 456 | H100 | 43m | 56.44 |
| `4012601_9` | flow_std | reacher-easy | 42 | H100 | 51m | 0.60 |
| `4012601_10` | flow_std | reacher-easy | 123 | H100 | 51m | 5.10 |
| `4012601_11` | flow_std | reacher-easy | 456 | H100 | 53m | 0.50 |
| `4012601_12` | flow_std | walker-stand | 42 | H100 | 52m | 34.22 |
| `4012601_13` | flow_std | walker-stand | 123 | H100 | 51m | 113.30 |
| `4012601_14` | flow_std | walker-stand | 456 | H100 | 52m | 119.42 |
| `4012601_15` | flow_std | cheetah-run | 42 | H100 | 51m | 3.18 |
| `4012601_16` | flow_std | cheetah-run | 123 | H100 | 51m | 18.60 |
| `4012601_17` | flow_std | cheetah-run | 456 | H100 | 51m | 0.91 |
| `4012601_18` | flow_high | reacher-easy | 42 | H100 | 62m | 2.70 |
| `4012601_19` | flow_high | reacher-easy | 123 | H100 | 63m | 1.30 |
| `4012601_20` | flow_high | reacher-easy | 456 | H100 | 63m | 0.70 |
| `4012601_21` | flow_high | walker-stand | 42 | H100 | 63m | 37.24 |
| `4012601_22` | flow_high | walker-stand | 123 | H100 | 62m | 94.25 |
| `4012601_23` | flow_high | walker-stand | 456 | H100 | 63m | 135.59 |
| `4012601_24` | flow_high | cheetah-run | 42 | H100 | 62m | 26.82 |
| `4012601_25` | flow_high | cheetah-run | 123 | H100 | 63m | 15.06 |
| `4012601_26` | flow_high | cheetah-run | 456 | H100 | 62m | 24.78 |

---

### Phase 6: Epoch Sweep Results

#### 6-D: 100k Epochs
| Job ID | Config | Task | Seed | Hardware | Runtime | Status | Final Reward |
|--------|--------|------|------|----------|---------|--------|--------------|
| `4012537_0` | Baseline | reacher-easy | 42 | H200 | 5h11m | ✅ | 438.50 |
| `4012537_1` | Baseline | reacher-easy | 123 | H200 | 5h06m | ✅ | 81.60 |
| `4012537_2` | Baseline | reacher-easy | 456 | H200 | 5h13m | ✅ | 113.10 |
| `4012537_3` | Baseline | walker-stand | 42 | H200 | 5h14m | ✅ | 213.53 |
| `4012537_4` | Baseline | walker-stand | 123 | H200 | 5h15m | ✅ | 159.29 |
| `4012537_5` | Baseline | walker-stand | 456 | H200 | 5h15m | ✅ | 150.36 |
| `4012537_6` | Baseline | cheetah-run | 42 | H200 | 5h06m | ✅ | 0.32 |
| `4012537_7` | Baseline | cheetah-run | 123 | H200 | 5h04m | ✅ | 1.94 |
| `4012537_8` | Baseline | cheetah-run | 456 | H200 | 5h10m | ✅ | 2.50 |
| `4012538_0-8` | Full Flow | All | * | H200 | 16h | ❌ TIMEOUT | Needs 20h+ |

#### 6-C: 50k Epochs
| Job ID | Config | Task | Seed | Hardware | Runtime | Status | Final Reward |
|--------|--------|------|------|----------|---------|--------|--------------|
| `4012535_0` | Baseline | reacher-easy | 42 | H100 | 2h36m | ✅ | 130.20 |
| `4012535_1` | Baseline | reacher-easy | 123 | H100 | 2h32m | ✅ | 1.00 |
| `4012535_2` | Baseline | reacher-easy | 456 | H100 | 2h34m | ✅ | 188.60 |
| `4012535_3` | Baseline | walker-stand | 42 | H100 | 2h38m | ✅ | 93.63 |
| `4012535_4` | Baseline | walker-stand | 123 | H100 | 2h37m | ✅ | 136.76 |
| `4012535_5` | Baseline | walker-stand | 456 | H100 | 2h37m | ✅ | 147.34 |
| `4012535_6` | Baseline | cheetah-run | 42 | H100 | 2h34m | ✅ | 44.35 |
| `4012535_7` | Baseline | cheetah-run | 123 | H100 | 2h33m | ✅ | 0.64 |
| `4012535_8` | Baseline | cheetah-run | 456 | H100 | 2h35m | ✅ | ~0.5 |
| `4012536_0-8` | Full Flow | All | * | H100 | 8h | ❌ TIMEOUT | Needs 16h+ |

#### 6-B: 15k Epochs - ALL COMPLETED
| Job ID | Config | Task | Seed | Hardware | Runtime | Final Reward |
|--------|--------|------|------|----------|---------|--------------|
| `4012533_0-8` | Baseline | All | * | H100 | ~49m | 0.14-156.05 |
| `4012534_0-8` | Full Flow | All | * | H100 | ~3h08m | 0.2-156.10 |

#### 6-A: 150k Epochs - MOSTLY FAILED (OOM)
| Job ID | Status | Notes |
|--------|--------|-------|
| `4012555_3` | ✅ | walker-stand, 7h, 111.38 |
| `4012555/56 (rest)` | ❌ | CUDA OOM |

---

### Phase 5: Flow Tuning (15k epochs) - ALL 18 COMPLETED
| Job ID | Config | Task | Seed | Hardware | Runtime | Final Reward |
|--------|--------|------|------|----------|---------|--------------|
| `4012434_0-8` | Various | walker-stand | * | H100 | 1h-3h | 117-156 |
| `4012434_9-17` | Various | cheetah-run | * | H100 | 1h-3h | 0.16-4.08 |

### Phase 4: Full Flow 10k - ALL 9 COMPLETED
| Job ID | Task | Hardware | Runtime | Final Reward |
|--------|------|----------|---------|--------------|
| `4012433_0-8` | All | H100 | ~1h19m | 0.2-141.68 |

### Phase 3: Pretrained WM Policy Training - ALL COMPLETED
| Job ID | Algo | Hardware | Final Reward Range |
|--------|------|----------|-------------------|
| `4011713_0-8` | Baseline | H100 | 93.69-983.50 |
| `4011714/40_0-8` | Flow Policy | H100 | 80.97-983.40 |

---

## 📂 Failed/Timeout Summary
| Job | Phase | Issue | Resolution |
|-----|-------|-------|------------|
| `4012538` | 6-100k Flow | TIMEOUT 16h | Need 20h+ |
| `4012536` | 6-50k Flow | TIMEOUT 8h | Need 16h+ |
| `4012555/56` | 6-150k | CUDA OOM | Reduce batch |
| `4012664/65/915/16` | 8 WM | Config error | Fixed → 4013702/03 |

---

## Summary Statistics (Completed Runs)

### Best Results by Phase
| Phase | Method | Reacher-easy | Walker-stand | Cheetah-run |
|-------|--------|--------------|--------------|-------------|
| 3 | Pretrained WM | 983.50 | 977.35 | 134.97 |
| 6-100k | From Scratch | 438.50 | 213.53 | 2.50 |
| 6-50k | From Scratch | 188.60 | 147.34 | 44.35 |
| 7-baseline | Fine-tune WM | 54.00 | 284.19 | 56.44 |
| 7-flow_high | Fine-tune WM | 2.70 | 135.59 | 26.82 |

### Key Observations
1. **Pretrained WM (Phase 3) remains best**: ~980 reacher, ~950 walker
2. **From-scratch 100k** shows improvement but still inconsistent
3. **Phase 7 Fine-tuning** with baseline outperforms flow variants
4. **Flow training needs 2-3x more time** than baseline
