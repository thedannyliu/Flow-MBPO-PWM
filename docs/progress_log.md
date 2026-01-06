# Progress Log

> Newest entries at top.

---

## 2026-01-06 04:00 – Submitted Resume and New Experiments

### Jobs Submitted
| Job ID | Type | Runs | Hardware | Time | Notes |
|--------|------|------|----------|------|-------|
| `4015240` | Resume Flow 50k | 9 | H200 | 16h | From 39k/50k |
| `4015250` | Resume Flow 100k | 8 | H200 | 16h | From 78k/100k |
| `4015251` | 150k Epoch Sweep | 18 | H200 | 16h | Baseline + Flow, batch=128 |

### Total New Jobs: 35

---

## 2026-01-06 03:52 – Incomplete Runs Analysis
- Flow 50k/100k: TIMEOUT, checkpoints exist, can resume
- 150k: OOM, retry with reduced batch_size=128

---

## 2026-01-06 03:45 – Phase 8 Progress Check
- MLP WM: ✅ COMPLETED
- Flow WM: 🟢 RUNNING (~8h remaining)
