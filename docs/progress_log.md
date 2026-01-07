# Progress Log

> Newest entries at top.

---

## 2026-01-07 03:50 – Resubmitting Failed Jobs with Fixes

### Status Update
- **Phase 8 WM Pretraining**: Both Flow (`4013702`, 12.5h) and MLP (`4013703`, 2.5h) are **COMPLETED**. Checkpoints are ready.
- **Phase 6 Resume/New Jobs**: Previous attempts (`4015342`, etc.) failed due to a Hydra config error (`ConfigCompositionException`).
- **Action**: Fixed submission scripts to use `+wandb.name` and `+wandb.project` for appending configuration. Resubmitted as:
    - **Resume 50k**: `4015402` (H100)
    - **Resume 100k**: `4015403` (H200)
    - **150k Sweep**: `4015404` (H200, batch 128)

---

## 2026-01-06 04:20 – Jobs Running Successfully (Initially Thought)
- It appeared jobs were running, but they failed shortly after start due to configuration issues.
- **Lesson**: Double-check `slurm-*.out` logs even if the job state is `RUNNING` for a few seconds.

---

## 2026-01-06 04:00 – Submitted Resume and New Experiments
- Submitted initial resume scripts.
- Configured proper Conda activation.
