# Progress Log

> Newest entries at top.

---

## 2026-01-07 04:10 – Launched Phase 9 and Fixed Phase 6 Resume

### Phase 9: 2x2 Factorial Design Launched
- **Job ID**: `4018569` (36 runs, H200, 16h)
- **Goal**: Compare MLP/Flow WM + MLP/Flow Policy (4 conditions).
- **Checkpoints**: Using verified Phase 8 checkpoints.

### Phase 6 Resume Fixed (Attempt 4)
- **Issue**: Jobs `4018496` failed silently due to missing `general.data_dir` causing `TypeError`.
- **Fix**: Explicitly added `general.data_dir="/home/hice1/eliu354/scratch/Data/tdmpc2/mt30"` to all scripts.
- **Status**:
    - `4018554` (Flow 50k): 🟢 RUNNING (Logs Clean)
    - `4018563` (Flow 100k): 🟢 RUNNING (Logs Clean)
    - `4018564` (150k Sweep): ⏳ QUEUED

---

## 2026-01-07 03:55 – Jobs Successfully Started Verify (INACCURATE)
- Thought jobs were running, but they crashed after 45s due to missing data_dir.

---

## 2026-01-07 03:50 – Resubmitting Failed Jobs with Fixes
- Fixed Hydra connection/activation.
