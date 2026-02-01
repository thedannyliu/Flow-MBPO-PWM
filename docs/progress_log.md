# Progress Log

> Newest entries at top.

---

## 2026-01-31 19:10 – Checkpoint Verification Complete

### Phase 8 WM Checkpoints Verified
Ran inspection job (`4044875`) to confirm Phase 8 checkpoints are valid and ready for Phase 9.

**Flow WM Checkpoint** (`flowwm_mt30_best.pt`):
- Iterations: 133,000
- Keys: `world_model` (39 keys), `world_model_opt`, `iter`, `obs_dim=24`, `act_dim=6`, `horizon=16`
- Status: `is_best=True`

**MLP WM Checkpoint** (`mlpwm_mt30_best.pt`):
- Iterations: 197,000
- Keys: Same structure as Flow WM
- Status: `is_best=True`

### Next Steps
1. Launch Phase 9 2x2 factorial experiment using these checkpoints.
2. Update documentation after successful launch.

---

## 2026-01-22 03:00 – Post-Failure Analysis & Fixes

### Status Check (Jan 7th Jobs)
- **Phase 9 (`4018569`)**: **FAILED** immediately.
    - **Cause**: `hydra.errors.ConfigCompositionException`. Keys `checkpoint_with_buffer` and `resume_training` were missing in `config_mt30.yaml`, requiring `+` prefix in overrides.
    - **Fix**: Updated `submit_phase9_factorial.sh` with `+general.checkpoint_with_buffer` and `+general.resume_training`.

- **Phase 6 (`4018554`, `4018563`)**: **FAILED** immediately.
    - **Cause**: `ValueError: 'dmcontrol-reacher-easy' is not in list`. The script passed `task=dmcontrol-${TASK}`, but `common.py` expects `reacher-easy` (no prefix).
    - **Fix**: Updated usage to `task=${TASK}` in all Phase 6 scripts.

---

## 2026-01-07 04:10 – Launched Phase 9 and Fixed Phase 6 Resume
- (These jobs failed shortly after launch)
