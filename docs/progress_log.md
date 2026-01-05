# Progress Log

> Purpose: Chronicle day-to-day development progress. Newest entries at top.

---

## 2026-01-04 19:50 – Epoch Sweep Experiments Launched

### Actions
- **Created 8 submission scripts** for epoch sweep (15k, 50k, 100k, 150k) × (Baseline, Flow)
- **GPU Assignment**: H100 for shorter runs (15k, 50k), H200 for longer runs (100k, 150k)
- **Configuration Alignment**: All params match original PWM (`wm_batch_size=256`)
- **Flow High Precision**: `flow_substeps=8` (WM), `flow_substeps=4` (Policy)

### Submitted Jobs
| Epochs | Baseline Job | Flow Job |
|--------|--------------|----------|
| 15k | `4012533` | `4012534` |
| 50k | `4012535` | `4012536` |
| 100k | `4012537` | `4012538` |
| 150k | `4012555` | `4012556` |

**Total**: 72 jobs (8 configs × 3 tasks × 3 seeds)

---

## 2026-01-04 17:30 – Storage Cleanup & Resubmission
- Reduced storage from 98.4% to 47.8% (~145GB free)
- Resubmitted Full Flow (`4012433`) and Tuning (`4012434`) jobs
- Reorganized `experiment_log.md` for clarity

---

## 2026-01-04 01:00 – Phase 3 Complete
- All Baseline and Flow Policy experiments finished
- Results: Baseline wins on `walker-stand` (+14%), tie on easy tasks
- Weights cleaned up, metrics preserved in `mt30_results_summary.csv`
