# Experiment Log

> **Purpose**: Persistent experiment registry for all training/evaluation jobs.
> **Fields**: Job ID, Config, Task, Seed, Runtime, Hardware, Status, Final Reward

---

## 🟢 Active Experiments

### Phase 6: Resume/New Epoch Sweep (Third Attempt - SUCCESS)
**Fix**: Fixed `+wandb.name` in `resume_flow_50k.sh` (previously missed).
**Fix**: All scripts have both Conda activation and correct Hydra syntax.

| Job ID | Configuration | Epochs | Hardware | Runtime | Status | Notes |
|--------|---------------|--------|----------|---------|--------|-------|
| `4018496` | Resume Flow 50k | 50k | H100 | < 1h | 🟢 RUNNING | Logs clean (Gym warnings only) |
| `4018497` | Resume Flow 100k | 100k | H200 | - | ⏳ QUEUED | Resume from 78k checkpoint |
| `4018498` | 150k Sweep (Baseline/Flow) | 150k | H200 | - | ⏳ QUEUED | Batch size 128 to fix OOM |

---

## ✅ Completed Phases

### Phase 8: WM Pretraining
| Job ID | WM Type | Config | Iters | Hardware | Runtime | Status | Best Loss | Checkpoint |
|--------|---------|--------|-------|----------|---------|--------|-----------|------------|
| `4013702` | Flow WM | `pwm_48M_mt_flowwm` | 200k | H100 | ~12.5h | ✅ COMPLETED | 1.3040 | `outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt` |
| `4013703` | MLP WM | `pwm_48M_mt_baseline` | 200k | H100 | 2h28m | ✅ COMPLETED | 0.0009 | `outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt` |

### Phase 7: Flow Policy Fine-tuning (27 runs)
- See `results/phase7_results.csv` for full details.

### Phase 6: Baseline Epoch Sweep
- **15k/50k/100k Baseline**: All completed successfully.

### Phase 3, 4, 5
- All completed. See `results/all_experiments.csv`.

---

## 📂 Failed Runs Diagnostics
| Job ID | Phase | Issue | Resolution |
|--------|-------|-------|------------|
| `4018450` | Resume Flow 50k | `ConfigCompositionException` | Forgot `+` in wandb name. Fixed in `4018496`. |
| `4015342/43/44` | Resume/150k | `ConfigCompositionException` | Added `+` to wandb overrides in submit scripts |
| `4015240/50/51` | Resume/150k | `python codec error` | Fixed conda activation in scripts |

---

## Next Steps (Priority Order)

1.  **Monitor Jobs**: Ensure `4018496+` continue running past initialization.
2.  **Phase 9: 2x2 Factorial Policy Training**:
    *   Launch using Phase 8 checkpoints once cluster allows.
