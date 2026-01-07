# Experiment Log

> **Purpose**: Persistent experiment registry for all training/evaluation jobs.
> **Fields**: Job ID, Config, Task, Seed, Runtime, Hardware, Status, Final Reward

---

## 🟢 Active Experiments

### Phase 6: Resume/New Epoch Sweep (Resubmitted)
**Fix**: Added prefix `+` to `wandb.name` override to fix `ConfigCompositionException`.
**Fix**: Added explicit conda environment activation.

| Job ID | Configuration | Epochs | Hardware | Runtime | Status | Notes |
|--------|---------------|--------|----------|---------|--------|-------|
| `4018450` | Resume Flow 50k | 50k | H100 | - | ⏳ QUEUED | Resume from 39k checkpoint |
| `4018451` | Resume Flow 100k | 100k | H200 | - | ⏳ QUEUED | Resume from 78k checkpoint |
| `4018452` | 150k Sweep (Baseline/Flow) | 150k | H200 | - | ⏳ QUEUED | Batch size 128 to fix OOM |

---

## ✅ Completed Phases

### Phase 8: WM Pretraining
**Method**: Pretrain WM From Scratch
**Purpose**: Create matched Flow WM and MLP WM checkpoints for fair 2×2 factorial comparison.

| Job ID | WM Type | Config | Iters | Hardware | Runtime | Status | Best Loss | Checkpoint |
|--------|---------|--------|-------|----------|---------|--------|-----------|------------|
| `4013702` | Flow WM | `pwm_48M_mt_flowwm` | 200k | H100 | ~12.5h | ✅ COMPLETED | 1.3040 | `outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt` |
| `4013703` | MLP WM | `pwm_48M_mt_baseline` | 200k | H100 | 2h28m | ✅ COMPLETED | 0.0009 | `outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt` |

### Phase 7: Flow Policy Fine-tuning (27 runs)
- See `results/phase7_results.csv` for full details.
- **Best Result**: Baseline on Walker-Stand (284.19). Flow variants generally underperformed.

### Phase 6: Baseline Epoch Sweep (27 runs)
- **15k/50k/100k Baseline**: All completed successfully.
- **Best Result**: 100k Baseline on Reacher (438.50).

### Phase 3, 4, 5
- All completed. See `results/all_experiments.csv`.

---

## 📂 Failed Runs Diagnostics
| Job ID | Phase | Issue | Resolution |
|--------|-------|-------|------------|
| `4015342/43/44` | Resume/150k | `ConfigCompositionException` | Added `+` to wandb overrides in submit scripts |
| `4015240/50/51` | Resume/150k | `python codec error` | Fixed conda activation in scripts |
| `4012538` | Flow 100k | TIMEOUT 16h | Resubmitting as `4015403` with resume |
| `4012536` | Flow 50k | TIMEOUT 8h | Resubmitting as `4015402` with resume |
| `4012555/56` | 150k Sweep | OOM | Resubmitting as `4015404` with batch=128 |

---

## Next Steps (Priority Order)

1.  **Monitor Resubmitted Jobs**: Ensure `4015402`, `4015403`, `4015404` start and run correctly.
2.  **Phase 9: 2x2 Factorial Policy Training**:
    *   Once we confirm the cluster is free/jobs are stable, launch the 2x2 factorial using the Phase 8 checkpoints (`flowwm_mt30_best.pt` and `mlpwm_mt30_best.pt`).
    *   This is the final critical experiment for the paper/project.
