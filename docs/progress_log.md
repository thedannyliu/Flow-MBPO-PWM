# Progress Log

> Purpose: Chronicle day-to-day development progress. Newest entries at top.

---

## 2026-01-05 03:18 – Comprehensive Experiment Audit

### Actions
- Audited **all running/completed jobs** across Phase 3-8
- Extracted final rewards from completed runs
- Fixed `pretrain_multitask_wm.py` OmegaConf.open_struct error
- Resubmitted Phase 8 WM pretraining: `4012915` (Flow), `4012916` (MLP)

### Failed Jobs Identified
| Job ID | Phase | Failure Reason | Resolution |
|--------|-------|----------------|------------|
| `4012664/65` | Phase 8 | ConfigAttributeError | Fixed & resubmitted |
| `4012555/56` (mostly) | Phase 6 150k | CUDA OOM | Need smaller batch or different config |
| `4012538_1` | Phase 6 100k | 1min exit | Check logs |

### Resume Capability Analysis
- **`train_multitask.py`**: ✅ Supports resume via `general.resume_from=<ckpt>`
- **`pretrain_multitask_wm.py`**: ❌ No resume support (would need to add)

### Completed Results Summary
| Phase | Epochs | Reacher (best) | Walker (best) | Cheetah (best) |
|-------|--------|----------------|---------------|----------------|
| 3 (Pretrained) | 15k | ~982 | ~977 | ~135 |
| 6-100k (Baseline) | 100k | 438.50 | 213.53 | 2.50 |
| 6-50k (Baseline) | 50k | 188.60 | 147.34 | 44.35 |
| 6-15k (Baseline) | 15k | 153.40 | 156.05 | 0.19 |
| 7 (Fine-tune) | 15k | 54.00 | 284.19 | 56.44 |

---

## 2026-01-04 22:00 – Documentation Update & Config Verification
- Updated experiment_log.md with all running jobs
- Verified config alignment with original PWM README

---

## 2026-01-04 21:50 – Phase 8 WM Pretraining Launched
- Submitted Flow WM (`4012664`) and MLP WM (`4012665`) - FAILED
- Created submission scripts for WM pretraining
