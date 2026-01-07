# Master Plan: Flow-MBPO MT30 Experiments

## Current Status (2026-01-07 03:50)

### ✅ Completed Milestones
- **Phase 3**: Pretrained WM + Policy (Best Performance)
- **Phase 4**: 10k Flow from Scratch (Undertrained)
- **Phase 5**: Flow Tuning
- **Phase 6 (Partial)**: Baseline Epoch Sweep (15k, 50k, 100k)
- **Phase 7**: Fine-tuning Pretrained WM
- **Phase 8 (WM Pretraining)**:
    - Flow WM: `outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt` (Loss: 1.3040)
    - MLP WM: `outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt` (Loss: 0.0009)

### 🟢 Active Experiments
| Phase | Job ID | Description | Status | Target |
|-------|--------|-------------|--------|--------|
| Phase 6 | `4018450` | Resume Flow 50k | ⏳ QUEUED | 50k epochs (from checkpoint) |
| Phase 6 | `4018451` | Resume Flow 100k | ⏳ QUEUED | 100k epochs (from checkpoint) |
| Phase 6 | `4018452` | 150k Baseline/Flow | ⏳ QUEUED | 150k epochs (Batch 128) |

---

## Next Steps

### 1. Execute Phase 9: 2x2 Factorial Design
**Goal**: Isolate the benefit of Flow WM vs. Flow Policy.
**Dependencies**: Phase 8 Checkpoints (Ready).

**Configurations**:
1.  **MLP WM + MLP Policy**: Control (Baseline)
    - `checkpoint=outputs/.../mlpwm_mt30_best.pt`
    - `alg=pwm_48M_mt_baseline`
2.  **MLP WM + Flow Policy**: Test Flow Policy benefit
    - `checkpoint=outputs/.../mlpwm_mt30_best.pt`
    - `alg=pwm_48M_mt_flowpolicy`
3.  **Flow WM + MLP Policy**: Test Flow WM benefit
    - `checkpoint=outputs/.../flowwm_mt30_best.pt`
    - `alg=pwm_48M_mt_flowwm`
4.  **Flow WM + Flow Policy**: Full Method
    - `checkpoint=outputs/.../flowwm_mt30_best.pt`
    - `alg=pwm_48M_mt_fullflow`

**Action**: Create submission script `scripts/mt30/submit_phase9_factorial.sh` once queue clears or is stable.

### 2. Analysis & Plotting
- Generate aggregate plots comparing all phases.
- Create final results table merging `all_experiments.csv` with new data.
