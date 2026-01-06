# Progress Log

> Newest entries at top.

---

## 2026-01-06 03:52 – Incomplete Runs Analysis

### Findings
**Timeout runs have checkpoints and can be resumed:**
- Flow 50k (4012536): Reached 39k/50k (78%), checkpoints exist
- Flow 100k (4012538): Reached 78k/100k (78%), checkpoints exist

**Cannot resume:**
- 150k (OOM at startup, no checkpoints)

### Suggested Actions
1. Resume Flow 50k with 16h time limit
2. Resume Flow 100k with 24h time limit
3. Skip 150k (OOM issues)

---

## 2026-01-06 03:45 – Phase 8 Progress Check

### Status
- **MLP WM** (`4013703`): ✅ COMPLETED (2h28m, loss=0.0009)
- **Flow WM** (`4013702`): 🟢 RUNNING (~8h36m, loss=1.3040)

---

## 2026-01-05 19:00 – Final Comprehensive Audit
- All Phase 7 (27 jobs) completed
- Created CSV files for all experiment results
- Fixed WM pretrain script and submitted 4013702/03

---

## 2026-01-05 03:18 – WM Pretrain Fixes
- Fixed OmegaConf.set_struct error
- Cleaned weights for runs < 4h: 69GB → 40GB
