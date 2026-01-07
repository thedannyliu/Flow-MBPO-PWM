# Master Plan: Flow-MBPO MT30 Experiments

## Current Status (2026-01-07 03:55)

### 🟢 Active Experiments
| Phase | Job ID | Description | Status | Target |
|-------|--------|-------------|--------|--------|
| Phase 6 | `4018496` | Resume Flow 50k | 🟢 RUNNING | 50k epochs (from checkpoint) |
| Phase 6 | `4018497` | Resume Flow 100k | ⏳ QUEUED | 100k epochs (from checkpoint) |
| Phase 6 | `4018498` | 150k Baseline/Flow | ⏳ QUEUED | 150k epochs (Batch 128) |

### ✅ Completed Milestones
- **Phase 8 (WM Pretraining)**:
    - Flow WM: `outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt` (Loss: 1.3040)
    - MLP WM: `outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt` (Loss: 0.0009)

---

## Next Steps

### 1. Execute Phase 9: 2x2 Factorial Design
(Blocked on Queue/Job Completion)

**Hypothesis**: Flow WM + Flow Policy > Flow WM + MLP Policy > MLP WM + MLP Policy.

**Plan**:
Run 4 variants on H100/H200:
1.  `alg=pwm_48M_mt_baseline` + MLP WM Checkpoint
2.  `alg=pwm_48M_mt_flowpolicy` + MLP WM Checkpoint
3.  `alg=pwm_48M_mt_flowwm` + Flow WM Checkpoint
4.  `alg=pwm_48M_mt_fullflow` + Flow WM Checkpoint

**Script**: `scripts/mt30/submit_phase9_factorial.sh` (To be created)
