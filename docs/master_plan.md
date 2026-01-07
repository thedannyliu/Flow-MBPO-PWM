# Master Plan: Flow-MBPO MT30 Experiments

## Current Status (2026-01-07 04:10)

### 🟢 Active Experiments (Total ~63 jobs)
| Phase | Job ID | Description | Status |
|-------|--------|-------------|--------|
| **Phase 9** | `4018569` | **2x2 Factorial Design** (36 runs) | ⏳ QUEUED |
| Phase 6 | `4018554` | Resume Flow 50k (9 runs) | 🟢 RUNNING |
| Phase 6 | `4018563` | Resume Flow 100k (9 runs) | 🟢 RUNNING |
| Phase 6 | `4018564` | 150k Baseline/Flow (18 runs) | ⏳ QUEUED |

### ✅ Completed Milestones
- **Phase 8 (WM Pretraining)**: Flow WM (Loss: 1.3040), MLP WM (Loss: 0.0009). Both checkpoints ready.
- **Phase 3-5, 7**: Complete.

---

## Phase 9: 2x2 Factorial Design (Executing)
**Conditions**:
1.  MLP WM + MLP Policy
2.  MLP WM + Flow Policy
3.  Flow WM + MLP Policy
4.  Flow WM + Flow Policy

**Hypothesis**: Full Flow (Cond 4) > Mixed (2/3) > Baseline (1).

---

## Next Steps
1.  **Monitor**: Ensure Phase 9 jobs start successfully when resources become available.
2.  **Analysis**:
    - Collect results from Phase 9.
    - Compare with Phase 6 (longer epochs).
    - Generate final paper plots.
