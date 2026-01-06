# Master Plan: Flow-MBPO MT30 Experiments

## Overview
**Goal**: Fair 2×2 factorial comparison on MT30:
- **World Model**: MLP vs Flow
- **Policy**: MLP vs Flow ODE

---

## Current Status (2026-01-05)

### ✅ Completed
| Phase | Description | Status |
|-------|-------------|--------|
| Phase 3 | Pretrained MLP WM + Policy | ✅ Best results (~980 reacher) |
| Phase 4 | Full Flow 10k epochs | ✅ Undertrained |
| Phase 5 | Flow Tuning 15k | ✅ Walker ~156, Cheetah <5 |
| Phase 6 15k | Baseline/Flow 15k | ✅ All completed |
| Phase 6 50k | Baseline | ✅ All 9 completed |
| Phase 6 100k | Baseline | ✅ All 9 completed |
| Phase 7 | Fine-tune WM | ✅ All 27 completed |

### ⏳ In Progress / Queued
| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| Phase 8 WM Pretrain | Flow/MLP WM | ⏳ QUEUED | 4013702/03, Fixed script |

### ❌ Failed/Timeout
| Phase | Issue | Resolution |
|-------|-------|------------|
| 6-50k Flow | TIMEOUT 8h | Needs 16h+ time limit |
| 6-100k Flow | TIMEOUT 16h | Needs 20h+ time limit |
| 6-150k | CUDA OOM | Reduce batch or use H200 |

---

## Next Steps (Priority Order)

### 1. Wait for Phase 8 WM Pretraining
- Monitor jobs `4013702` (Flow WM) and `4013703` (MLP WM)
- Expected ~16h on H100

### 2. Run 2×2 Factorial Policy Training (After WM Pretrain)
Use pretrained checkpoints for fair comparison:
| WM | Policy | Config | Checkpoint |
|---|---|---|---|
| MLP | MLP | `pwm_48M_mt_baseline` | `mlpwm_mt30_best.pt` |
| MLP | Flow | `pwm_48M_mt_flowpolicy` | `mlpwm_mt30_best.pt` |
| Flow | MLP | `pwm_48M_mt_flowwm` | `flowwm_mt30_best.pt` |
| Flow | Flow | `pwm_48M_mt_fullflow` | `flowwm_mt30_best.pt` |

### 3. Optional: Resubmit Flow Epoch Sweep
- 50k Flow: Increase time limit to 16h
- 100k Flow: Increase time limit to 24h

---

## Key Findings

1. **Pretrained WM is critical**: Phase 3 (~980 reward) >> From-scratch (~150-400)
2. **Flow needs more training time**: 2-3x slower than baseline
3. **Fine-tuning WM (Phase 7)** doesn't help Flow policy (baseline still wins)
4. **Cheetah-run is hardest**: Most from-scratch runs fail (<1 reward)

---

## Resources
- **Original PWM Checkpoint**: `checkpoints/multitask/mt30_48M_4900000.pt`
- **WM Pretraining Script**: `scripts/pretrain_multitask_wm.py`
- **Data**: `/home/hice1/eliu354/scratch/Data/tdmpc2/mt30/`
