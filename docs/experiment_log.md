# Experiment Log

> **Purpose**: Persistent experiment registry for all training/evaluation jobs.
> **Fields**: Job ID, Config, Task, Seed, Runtime, Hardware, Status, Final Reward

---

## 🟢 Active Experiments

### Phase 8: WM Pretraining
| Job ID | WM Type | Config | Iters | Hardware | Runtime | Status | Best Loss | Checkpoint |
|--------|---------|--------|-------|----------|---------|--------|-----------|------------|
| `4013702` | Flow WM | `pwm_48M_mt_flowwm` | 200k | H100 | ~8h36m | 🟢 RUNNING | 1.3040 | `outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt` |
| `4013703` | MLP WM | `pwm_48M_mt_baseline` | 200k | H100 | 2h28m | ✅ COMPLETED | 0.0009 | `outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt` |

### Phase 6: Resume/New Epoch Sweep (Just Submitted)
| Job ID | Config | Epochs | Hardware | Time Limit | Status | Notes |
|--------|--------|--------|----------|------------|--------|-------|
| `4015240` | FullFlow | 50k | H200 | 16h | ⏳ QUEUED | Resume from 39k/50k |
| `4015250` | FullFlow | 100k | H200 | 16h | ⏳ QUEUED | Resume from 78k/100k |
| `4015251_0-8` | Baseline | 150k | H200 | 16h | ⏳ QUEUED | wm_batch_size=128 |
| `4015251_9-17` | FullFlow | 150k | H200 | 16h | ⏳ QUEUED | wm_batch_size=128 |

---

## ✅ Completed Phases

### Phase 7: Flow Policy Fine-tuning (27 runs)
- See `results/phase7_results.csv` for full details
- Best: baseline walker-stand 284.19

### Phase 6: Epoch Sweep (Baseline Complete)
- **15k Baseline**: ✅ 9 runs
- **15k FullFlow**: ✅ 9 runs
- **50k Baseline**: ✅ 9 runs
- **100k Baseline**: ✅ 9 runs

### Phase 5: Flow Tuning (18 runs)
- All completed

### Phase 4: Full Flow 10k (9 runs)
- All completed

### Phase 3: Pretrained WM (18 runs)
- Best: reacher 983.50, walker 977.35, cheetah 134.97

---

## 📂 Previous Failed Runs
| Phase | Job IDs | Issue | Resolution |
|-------|---------|-------|------------|
| 6-100k Flow | 4012538 | TIMEOUT 16h | ✅ Resume → 4015250 |
| 6-50k Flow | 4012536 | TIMEOUT 8h | ✅ Resume → 4015240 |
| 6-150k | 4012555/56 | CUDA OOM | ✅ Retry → 4015251 (batch=128) |
| 8 WM | 4012664/65/915/16 | Config error | ✅ Fixed → 4013702/03 |
