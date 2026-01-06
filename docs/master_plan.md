# Master Plan: Flow-MBPO MT30 Experiments

## Current Status (2026-01-06 03:45)

### ✅ Phase 8 MLP WM Pretrain - COMPLETED
- **Job**: `4013703`
- **Runtime**: 2h28m
- **Best Loss**: 0.0009
- **Checkpoint**: `outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt`

### 🟢 Phase 8 Flow WM Pretrain - RUNNING
- **Job**: `4013702`
- **Runtime**: ~8h36m (ongoing)
- **Best Loss**: 1.3040
- **Checkpoint**: `outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt`

---

## Next Steps (Priority Order)

### 1. Wait for Flow WM Pretrain to Complete
Expected ~8h more based on current progress.

### 2. Run 2×2 Factorial Policy Training
After both WM checkpoints are ready:

| WM | Policy | Config | Checkpoint |
|---|---|---|---|
| MLP | MLP | `pwm_48M_mt_baseline` | `mlpwm_mt30_best.pt` |
| MLP | Flow | `pwm_48M_mt_flowpolicy` | `mlpwm_mt30_best.pt` |
| Flow | MLP | `pwm_48M_mt_flowwm` | `flowwm_mt30_best.pt` |
| Flow | Flow | `pwm_48M_mt_fullflow` | `flowwm_mt30_best.pt` |

### 3. Optional: Resubmit Flow Epoch Sweep
- 50k Flow: 16h time limit
- 100k Flow: 24h time limit

---

## Completed Phases Summary
| Phase | Runs | Best Reward (Reacher) | Best (Walker) | Best (Cheetah) |
|-------|------|----------------------|---------------|----------------|
| 3 (Pretrained) | 18 | 983.50 | 977.35 | 134.97 |
| 4 (10k) | 9 | 112.20 | 141.68 | 1.40 |
| 5 (Tuning) | 18 | - | 156.10 | 4.08 |
| 6 (Epoch Sweep) | 36 | 438.50 | 213.53 | 44.35 |
| 7 (Fine-tune) | 27 | 54.00 | 284.19 | 56.44 |

---

## Resources
- **MLP WM Checkpoint**: `outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt`
- **Flow WM Checkpoint**: `outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt` (in progress)
- **Original PWM**: `checkpoints/multitask/mt30_48M_4900000.pt`
