# Progress Log

> Newest entries at top.

---

## 2026-01-06 03:45 – Phase 8 Progress Check

### Status
- **`4013703` (MLP WM Pretrain)**: ✅ COMPLETED
  - Runtime: 2h28m
  - Best Loss: 0.0009
  - Checkpoint: `outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt`
  
- **`4013702` (Flow WM Pretrain)**: 🟢 RUNNING
  - Runtime: ~8h36m (ongoing)
  - Best Loss: 1.3040
  - Checkpoint (in progress): `outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt`

### Notes
- MLP WM trains much faster (2.5h vs 8h+)
- Flow WM loss (~1.30) is much higher than MLP WM (0.0009) - expected due to different architectures

### Next Steps
1. Wait for Flow WM to complete (~8h more expected)
2. Run 2×2 factorial policy training with both checkpoints

---

## 2026-01-05 19:00 – Final Comprehensive Audit
- All Phase 7 (27 jobs) completed
- Created CSV files for all experiment results
- Fixed WM pretrain script and submitted 4013702/03

---

## 2026-01-05 03:18 – WM Pretrain Fixes
- Fixed OmegaConf.set_struct error
- Cleaned weights for runs < 4h: 69GB → 40GB
