# Progress Log

> Purpose: Chronicle day-to-day development progress. Newest entries at top.

---

## 2026-01-04 22:00 – Documentation Update & Config Verification

### Actions
- Updated `experiment_log.md` with **all running/queued jobs** for Phase 5/6/7/8.
- Added individual job entries for Phase 7 (27 jobs) and Phase 6 (72 jobs total).
- Verified **config alignment** with original PWM README:
  - `horizon=16` ✅
  - `batch_size=1024` (WM pretraining) ✅
  - `wm_batch_size=256` (policy training) ✅

### Observations
- Phase 6 100k/150k jobs running on H200 (~2h runtime so far).
- Phase 8 WM pretraining jobs queued on H100.

---

## 2026-01-04 21:50 – Phase 8 WM Pretraining Launched

### Actions
- Submitted **Phase 8 WM pretraining jobs**:
  - Flow WM: `4012664` (200k iters, H100)
  - MLP WM: `4012665` (200k iters, H100)
- Created submission scripts:
  - `scripts/mt30/submit_pretrain_flow_wm.sh`
  - `scripts/mt30/submit_pretrain_mlp_wm.sh`

### Observations
- Phase 4/5/6 from-scratch training yielded poor results (cheetah-run ~0.2 reward).
- Confirms need for pretrained WM (Phase 8 approach).

---

## 2026-01-04 21:00 – Phase 8 Code Enhancements

### Actions
- Enhanced `pretrain_multitask_wm.py` with:
  - WandB logging for all training metrics
  - Best/last checkpoint saving (no intermediate spam)
  - `eval_every` configuration for best checkpoint tracking
- Added `pwm_48M_mt_flowwm.yaml` for 4th quadrant (Flow WM + MLP Policy)
- Fixed `buffer.py` episode ID collision bug
- Enhanced `pwm.py` `load_wm()` for dual checkpoint format support

---

## 2026-01-04 20:25 – Phase 7 Launched (Flow Fine-tuning)
- Submitted array job `4012601` (27 tasks)
- Comparing Baseline vs Flow Std vs Flow High with Pretrained MLP WM

---

## 2026-01-04 19:50 – Epoch Sweep Experiments Launched
- Submitted 72 jobs for Phase 6 (15k/50k/100k/150k epochs)
- Phase 6 100k/150k running on H200
