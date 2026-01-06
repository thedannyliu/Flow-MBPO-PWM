# Master Plan: Flow-MBPO MT30 Experiments

## Current Status (2026-01-06 04:00)

### 🟢 Running/Queued Jobs

| Job ID | Type | Runs | Hardware | Status |
|--------|------|------|----------|--------|
| `4013702` | Flow WM Pretrain | 1 | H100 | 🟢 RUNNING (~8h left) |
| `4015240` | Resume Flow 50k | 9 | H200 | ⏳ QUEUED |
| `4015250` | Resume Flow 100k | 8 | H200 | ⏳ QUEUED |
| `4015251` | 150k Epoch Sweep | 18 | H200 | ⏳ QUEUED |

**Total Active: 36 jobs**

---

## Next Steps (After Current Jobs)

### 1. 2×2 Factorial Policy Training
After Flow WM pretrain completes:
| WM | Policy | Config | Checkpoint |
|---|---|---|---|
| MLP | MLP | `pwm_48M_mt_baseline` | `mlpwm_mt30_best.pt` |
| MLP | Flow | `pwm_48M_mt_flowpolicy` | `mlpwm_mt30_best.pt` |
| Flow | MLP | `pwm_48M_mt_flowwm` | `flowwm_mt30_best.pt` |
| Flow | Flow | `pwm_48M_mt_fullflow` | `flowwm_mt30_best.pt` |

---

## Completed Summary
| Phase | Runs | Best Reacher | Best Walker | Best Cheetah |
|-------|------|--------------|-------------|--------------|
| 3 | 18 | 983.50 | 977.35 | 134.97 |
| 6-100k Baseline | 9 | 438.50 | 213.53 | 2.50 |
| 7 | 27 | 54.00 | 284.19 | 56.44 |

---

## Resources
- **MLP WM**: `outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt`
- **Flow WM**: `outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt` (in progress)
