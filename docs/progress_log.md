# Progress Log

> Purpose: Chronicle day-to-day development progress. Newest entries at top.

---

## 2026-01-05 00:30 – 2×2 Factorial Setup + Multitask WM Pretraining Added
- **Goal**: Enable fair 2×2 comparison: (MLP vs Flow) World Model × (MLP vs Flow) Policy, with WM pretraining and optional fine-tuning.
- **Added 4th quadrant config**: Flow WM + MLP Policy (`scripts/cfg/alg/pwm_48M_mt_flowwm.yaml`) to complete the 2×2.
- **Added WM pretraining entrypoint**: `scripts/pretrain_multitask_wm.py` (+ `scripts/cfg/pretrain_mt30_wm.yaml`) to pretrain Flow WM (and optionally MLP WM) on TD-MPC2 MT datasets.
- **Fixed offline buffer episode ID bug**: `src/flow_mbpo_pwm/utils/buffer.py` now assigns unique `episode` ids across multiple `add_batch()` calls (prevents SliceSampler grouping corruption).
- **Improved checkpoint compatibility**: `src/flow_mbpo_pwm/algorithms/pwm.py::load_wm()` now supports original PWM (`model`) and native (`world_model`) checkpoint formats.
- **Docs update**: `docs/experiment_log.md` now includes Phase 8 (2×2 factorial plan + checkpoint requirements).

---

## 2026-01-04 20:25 – Phase 7 Launched (Flow Fine-tuning)
- **Goal**: Test if Fine-tuning the Pretrained MLP WM (`finetune_wm=True`) helps Flow Policy performance.
- **Jobs**: Submitted array job `4012601` (27 tasks).
- **Setup**: 
    - **Baseline**: MLP Policy + MLP WM (Fine-tune)
    - **Flow Std**: Flow Policy (Substeps=2) + MLP WM (Fine-tune)
    - **Flow High**: Flow Policy (Substeps=4) + MLP WM (Fine-tune)
- **GPU**: H100 (15k epochs ~4h)

---

## 2026-01-04 20:15 – Documentation & Methodology Clarification

### Key Findings (Training Discrepancy)
- **Original PWM**: Uses a **2-week pretraining phase** for World Model (on RTX 3090) before Policy Training.
- **Our Current Full Flow**: Attempts **Joint Training** (WM + Policy) from scratch in just hours (15k-150k epochs).
- **Implication**: We are testing if Flow can learn dynamics significantly faster than MLP, or if "From Scratch" training is feasible without massive pretraining.

### Actions
- **Updated `experiment_log.md`**: Added explicit "Methodology" section for each phase (Policy Fine-tuning vs Joint Training).
- **Updated `master_plan.md`**: Clarified the "Critical Distinction" between phases and the "Training Duration Discrepancy".

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
