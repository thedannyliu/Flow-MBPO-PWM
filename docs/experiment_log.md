# Experiment Log

> **Purpose**: Persistent experiment registry for all training/evaluation jobs.

---

## 🟢 Active Experiments (Running Now)

### Phase 8: WM Pretraining
| Job ID | WM Type | Hardware | Runtime | Status | Best Loss |
|--------|---------|----------|---------|--------|-----------|
| `4013702` | Flow WM | H100 | ~9h12m | 🟢 RUNNING | 1.3040 |
| `4013703` | MLP WM | H100 | 2h28m | ✅ COMPLETED | 0.0009 |

### Phase 6: Flow 50k (9 runs, H100, 8h)
| Job ID | Config | Hardware | Status |
|--------|--------|----------|--------|
| `4015342_0-8` | FullFlow | H100 | 🟢 RUNNING |

### Phase 6: Flow 100k (9 runs, H200, 16h)
| Job ID | Config | Hardware | Status |
|--------|--------|----------|--------|
| `4015343_0-8` | FullFlow | H200 | 🟢 RUNNING |

### Phase 6: 150k Epoch Sweep (18 runs, H200, 16h)
| Job ID | Config | Hardware | Status | Notes |
|--------|--------|----------|--------|-------|
| `4015344_0-8` | Baseline | H200 | 🟢 RUNNING/QUEUED | wm_batch_size=128 |
| `4015344_9-17` | FullFlow | H200 | ⏳ QUEUED | wm_batch_size=128 |

**Summary**: 22 running, 16 queued

---

## ✅ Completed Phases

### Phase 7: Fine-tuning (27 runs)
- See `results/phase7_results.csv`

### Phase 6: Baseline Epoch Sweep
- 15k: ✅ 9 runs
- 50k: ✅ 9 runs
- 100k: ✅ 9 runs

### Phase 3-5
- All completed, see `results/all_experiments.csv`

---

## 📂 Previous Fixed Issues
| Issue | Resolution |
|-------|------------|
| Conda activation failed | Fixed with `eval "$(conda shell.bash hook)"` |
| 150k OOM | Reduced `wm_batch_size=128` |
| WM pretrain config error | Fixed OmegaConf.set_struct |
