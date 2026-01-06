# Master Plan: Flow-MBPO MT30 Experiments

## Current Status (2026-01-06 04:20)

### 🟢 Running Jobs (38 total)

| Job ID | Type | Runs | Hardware | Time | Status |
|--------|------|------|----------|------|--------|
| `4013702` | Flow WM Pretrain | 1 | H100 | 9h12m | 🟢 RUNNING |
| `4015342` | Flow 50k | 9 | H100 | 8h | 🟢 ALL RUNNING |
| `4015343` | Flow 100k | 9 | H200 | 16h | 🟢 ALL RUNNING |
| `4015344` | 150k Epoch Sweep | 18 | H200 | 16h | 2 RUN, 16 QUEUE |

---

## Next Steps (After Current Jobs)

### 1. 2×2 Factorial Policy Training
After Flow WM pretrain completes (~7h remaining):
| WM | Policy | Checkpoint |
|---|---|---|
| MLP | MLP | `mlpwm_mt30_best.pt` |
| MLP | Flow | `mlpwm_mt30_best.pt` |
| Flow | MLP | `flowwm_mt30_best.pt` |
| Flow | Flow | `flowwm_mt30_best.pt` |

---

## Completed Summary
| Phase | Runs | Best Result |
|-------|------|-------------|
| 3 (Pretrained) | 18 | reacher 983, walker 977 |
| 6 (Baseline) | 27 | reacher 438, walker 213 |
| 7 (Fine-tune) | 27 | walker 284, cheetah 56 |

---

## Resources
- **MLP WM**: `outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt` ✅
- **Flow WM**: In progress (~7h remaining)
