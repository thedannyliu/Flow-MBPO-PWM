# Master Plan: Flow-MBPO MT30 Experiments

## Current Status (2026-01-06 03:52)

### 🟢 Running
- **Phase 8 Flow WM** (`4013702`): ~8h36m elapsed, ~8h remaining

### ✅ Completed
- **Phase 8 MLP WM** (`4013703`): 2h28m, loss=0.0009
- **Phase 3-7**: ~108 runs completed

### ⏸️ Incomplete (Can Resume)
- **Flow 50k** (4012536): 9 runs, 78% complete (39k/50k)
- **Flow 100k** (4012538): 8 runs, 78% complete (78k/100k)

### ❌ Failed (Cannot Resume)
- **150k**: CUDA OOM

---

## Next Steps (Priority Order)

### 1. Wait for Phase 8 Flow WM (~8h)
Job `4013702` currently running.

### 2. Resume Incomplete Flow Epoch Sweep
```bash
# Resume Flow 50k with 16h time limit
python train_multitask.py general.resume_from=<checkpoint> --time=16:00:00

# Resume Flow 100k with 24h time limit  
python train_multitask.py general.resume_from=<checkpoint> --time=24:00:00
```

### 3. Run 2×2 Factorial (After WM Pretrain)
| WM | Policy | Config | Checkpoint |
|---|---|---|---|
| MLP | MLP | `pwm_48M_mt_baseline` | `mlpwm_mt30_best.pt` |
| MLP | Flow | `pwm_48M_mt_flowpolicy` | `mlpwm_mt30_best.pt` |
| Flow | MLP | `pwm_48M_mt_flowwm` | `flowwm_mt30_best.pt` |
| Flow | Flow | `pwm_48M_mt_fullflow` | `flowwm_mt30_best.pt` |

---

## Summary Statistics (108 Completed)
| Phase | Runs | Best Reacher | Best Walker | Best Cheetah |
|-------|------|--------------|-------------|--------------|
| 3 | 18 | 983.50 | 977.35 | 134.97 |
| 6-100k | 9 | 438.50 | 213.53 | 2.50 |
| 7 | 27 | 54.00 | 284.19 | 56.44 |

---

## Resources
- **MLP WM**: `outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt`
- **Flow WM** (in progress): `outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt`
- **Original PWM**: `checkpoints/multitask/mt30_48M_4900000.pt`
