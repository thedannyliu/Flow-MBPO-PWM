# Experiment Log

> **Purpose**: Persistent experiment registry for all training/evaluation jobs.
> **Fields**: Job ID, Config, Task, Seed, Runtime, Hardware, Status, Final Reward

---

## 🟢 Active Experiments

### Phase 9: 2x2 Factorial Design
(Factorial of WM Type x Policy Type)

| Job ID | Configuration | Checkpoint | Status | Notes |
|--------|---------------|------------|--------|-------|
| `4018569` | (36 runs) | MLP/Flow WM | ⏳ QUEUED | 4 Conditions x 9 Task-Seeds |
| `Condition 0` | MLP WM + MLP Policy | MLP CKPT | - | Control |
| `Condition 1` | MLP WM + Flow Policy | MLP CKPT | - | Flow Policy Benefit |
| `Condition 2` | Flow WM + MLP Policy | Flow CKPT | - | Flow WM Benefit |
| `Condition 3` | Flow WM + Flow Policy | Flow CKPT | - | Full Method |

### Phase 6: Resume/New Epoch Sweep (Fourth Attempt - SUCCESS)
**Fixes**:
1. Added explicit `conda` activation.
2. Added `+` prefix to Hydra overrides.
3. Added `general.data_dir` path to scripts.

| Job ID | Configuration | Epochs | Hardware | Runtime | Status | Notes |
|--------|---------------|--------|----------|---------|--------|-------|
| `4018554` | Resume Flow 50k | 50k | H100 | < 1h | 🟢 RUNNING | Verified Logs Clean |
| `4018563` | Resume Flow 100k | 100k | H200 | < 1h | 🟢 RUNNING | Verified Logs Clean |
| `4018564` | 150k Sweep (Baseline/Flow) | 150k | H200 | - | ⏳ QUEUED | Batch size 128 |

---

## ✅ Completed Phases

### Phase 8: WM Pretraining
| Job ID | WM Type | Config | Iters | Hardware | Runtime | Status | Best Loss | Checkpoint |
|--------|---------|--------|-------|----------|---------|--------|-----------|------------|
| `4013702` | Flow WM | `pwm_48M_mt_flowwm` | 200k | H100 | ~12.5h | ✅ COMPLETED | 1.3040 | `outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt` |
| `4013703` | MLP WM | `pwm_48M_mt_baseline` | 200k | H100 | 2h28m | ✅ COMPLETED | 0.0009 | `outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt` |

---

## 📂 Failed Runs Diagnostics
| Job ID | Phase | Issue | Resolution |
|--------|-------|-------|------------|
| `4018496/97/98` | Resume/150k | `TypeError: NoneType` (missing data_dir) | Added `general.data_dir` to scripts |
| `4018450` | Resume Flow 50k | `ConfigCompositionException` | Forgot `+` in wandb name. Fixed. |
| `4015342/43/44` | Resume/150k | `ConfigCompositionException` | Added `+` to wandb overrides |
| `4015240/50/51` | Resume/150k | `python codec error` | Fixed conda activation |

---

## Next Steps
1.  **Monitor**: Wait for results.
2.  **Analysis**: Once Phase 9 completes, generate comparison plots.
