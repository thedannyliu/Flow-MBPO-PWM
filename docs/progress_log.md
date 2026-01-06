# Progress Log

> Newest entries at top.

---

## 2026-01-05 19:00 – Final Comprehensive Audit

### Completed
- **All Phase 7** (27 jobs): Baseline ~1-284, Flow Std ~0.5-119, Flow High ~1-135
- **All Phase 6 Baseline** (50k, 100k): Completed
- **All Phase 6 15k**: Baseline and Flow completed

### Failed/Timeout
- **Phase 6 Flow 50k** (4012536): All 9 TIMEOUT at 8h
- **Phase 6 Flow 100k** (4012538): All 8 TIMEOUT at 16h
- **Phase 8 WM Pretrain** (4012664/65/915/16): OmegaConf errors

### Actions Taken
- Fixed `pretrain_multitask_wm.py`: OmegaConf.set_struct
- Resubmitted WM pretrain: `4013702` (Flow), `4013703` (MLP)
- Cleaned weights for runs < 4h: 69GB → 40GB
- Updated all documentation

---

## 2026-01-05 03:18 – WM Pretrain Fix Attempt
- Fixed OmegaConf.open_struct (wrong API) → should be set_struct
- Resubmitted 4012915/16 - still failed

---

## 2026-01-04 21:50 – Phase 8 WM Pretraining Launched
- First attempt 4012664/65 - FAILED (ConfigAttributeError)

---

## 2026-01-04 20:25 – Phase 7 Launched
- Array job 4012601 (27 tasks) - All completed

---

## 2026-01-04 19:50 – Phase 6 Epoch Sweep Launched
- 72 jobs for 15k/50k/100k/150k epochs
- Baseline completed, Flow mostly timeout
