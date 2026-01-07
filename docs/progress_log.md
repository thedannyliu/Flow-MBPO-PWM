# Progress Log

> Newest entries at top.

---

## 2026-01-07 03:55 – Jobs Successfully Started Verify ✅

### Status
- **Phase 6 Resume (Attempt 3)**:
    - `4018496` (Resume 50k) is **RUNNING** and logs show proper initialization (no Python/Hydra crashes).
    - `4018497` and `4018498` are submitted/queued.
- **Phase 8**: Confirmed Complete.

### Fix Summary
1.  **Conda Activation**: `eval "$(conda shell.bash hook)"` fixed Python codec error.
2.  **Hydra Overrides**: Added `+wandb.name` to ALL scripts (initially missed one).

---

## 2026-01-07 03:50 – Resubmitting Failed Jobs with Fixes
- Previous batch (`4018450`) failed because I missed adding `+` to `resume_flow_50k.sh`. Fixed and resubmitted.

---

## 2026-01-07 03:50 – Resubmitting Failed Jobs with Fixes
- Resubmitted jobs with `+wandb.project` fixed.
