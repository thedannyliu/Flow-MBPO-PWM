# Experiment Log

> **Purpose**: This document is the authoritative registry for all training, evaluation, and verification jobs in the Flow-MBPO project. It tracks SLURM job IDs, checkpoint paths, configurations, seeds, WandB links, and experiment status.
>
> **Usage Guidelines**:
> - **Never delete entries** - only update status (Pending → Running → Completed/Failed → Evaluated)
> - **Newest entries at top** for each section
> - Each entry must include: Job ID, variant, seed, script path, checkpoint path, status
> - Update with eval results once available
>
> **Status Legend**:
> - 🕐 **PENDING** - Job submitted, waiting in queue
> - 🔄 **RUNNING** - Job currently executing
> - ✅ **COMPLETED** - Training finished, awaiting eval
> - ❌ **FAILED** - Job failed, needs investigation
> - 📊 **EVALUATED** - Eval complete, results recorded

---

## Resource Allocation (Standard)
| Setting | Value |
|---------|-------|
| Account | gts-agarg35-ideas_l40s |
| Partition | gpu-l40s |
| GPU | L40s × 1 |
| Memory | 400GB |
| Time Limit | 40 hours |
| Conda Env | pwm |

---

# NEWLY COMPLETED JOBS (Jan 3, 2026)

## Training Completed ✅
| Job ID | Task | Variant | Seed | Status | Checkpoint | Eval Result |
|--------|------|---------|------|--------|------------|-------------|
| 3138480 | Anymal | Baseline | 42 | � EVALUATED | `outputs/2026-01-03/02-00-06` | 24.84 |
| 3138481 | Anymal | Baseline | 123 | � EVALUATED | `outputs/2026-01-03/02-02-33` | 25.75 |
| 3139000 | Anymal | Baseline | 456 | � EVALUATED | `outputs/2026-01-03/03-30-22` | 35.12 |
| 3139001 | Humanoid | FlowWM K=8 | 456 | ✅ COMPLETED | `outputs/2026-01-03/10-39-37` | 26.67 |
| 3139002 | Humanoid | FlowPolicy | 42 | � EVALUATED | `outputs/2026-01-03/03-30-26` | 93.14 |
| 3139003 | Humanoid | FlowPolicy | 123 | ❌ FAILED | - | - |
| 3139004 | Humanoid | FlowPolicy | 456 | � EVALUATED | `outputs/2026-01-03/06-48-54` | 83.66 |
| 3139005 | Humanoid | FullFlow | 42 | � EVALUATED | `outputs/2026-01-03/07-23-25` | 42.40 |
| 3139006 | Humanoid | FullFlow | 123 | � EVALUATED | `outputs/2026-01-03/09-00-09` | 26.67 |
| 3139007 | Humanoid | FullFlow | 456 | 🔄 RUNNING | - | - |

## Still Missing
- Humanoid FlowPolicy s123 (failed - CUDA busy)

---

# COMPLETED EXPERIMENTS

## Humanoid (Dec 29-30, 2025)

### Baseline ✅ EVALUATED
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| 3101831 | 42 | `outputs/2025-12-29/20-17-36/logs/best_policy.pt` | 37.82 | 📊 EVALUATED |
| 3101832 | 123 | `outputs/2025-12-29/20-20-53/logs/best_policy.pt` | 65.52 | 📊 EVALUATED |
| 3101833 | 456 | `outputs/2025-12-29/20-31-30/logs/best_policy.pt` | 82.93 | 📊 EVALUATED |

### FlowWM K=4 ✅ EVALUATED
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| 3104842 | 42 | `outputs/2025-12-29/22-54-39/logs/best_policy.pt` | 55.09 | 📊 EVALUATED |
| 3104843 | 123 | `outputs/2025-12-29/22-54-42/logs/best_policy.pt` | 34.48 | 📊 EVALUATED |

### FlowWM K=2 ✅ EVALUATED
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| 3104846 | 123 | `outputs/2025-12-29/22-54-41/logs/best_policy.pt` | 34.09 | 📊 EVALUATED |

### FlowWM K=8 ✅ PARTIAL
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| 3107946 | 42 | `outputs/2025-12-30/04-20-56/logs/best_policy.pt` | 36.76 | 📊 EVALUATED |
| 3107947 | 123 | `outputs/2025-12-30/04-20-58/logs/best_policy.pt` | 30.70 | 📊 EVALUATED |
| - | 456 | - | - | ❌ FAILED (CUDA busy) |

### FlowPolicy ❌ PARTIAL
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| 3125488 | 42 | `outputs/2025-12-29/20-18-50/logs/best_policy.pt` | 58.90 | 📊 EVALUATED |
| - | 123 | - | - | ❌ FAILED |
| - | 456 | - | - | ❌ FAILED |

### FullFlow K=4 ✅ PARTIAL
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| - | 42 | `outputs/2025-12-29/20-17-31/logs/best_policy.pt` | 39.99 | 📊 EVALUATED |
| - | 123 | - | - | ❌ FAILED |
| - | 456 | - | - | ❌ FAILED |

---

## Ant (Dec 29, 2025) ✅ COMPLETE

### Baseline ✅ EVALUATED
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| - | 42 | `outputs/2025-12-29/07-10-46/logs/best_policy.pt` | 1170.49 | 📊 EVALUATED |
| - | 123 | `outputs/2025-12-29/08-18-33/logs/best_policy.pt` | 22.50 | 📊 EVALUATED |
| - | 456 | `outputs/2025-12-29/09-16-25/logs/best_policy.pt` | 85.45 | 📊 EVALUATED |

### FlowWM K=2 ✅ EVALUATED
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| - | 42 | `outputs/2025-12-29/11-03-10/logs/best_policy.pt` | 154.80 | 📊 EVALUATED |
| - | 123 | `outputs/2025-12-29/11-06-12/logs/best_policy.pt` | 1263.74 | 📊 EVALUATED |
| - | 456 | `outputs/2025-12-29/11-21-33/logs/best_policy.pt` | 871.15 | 📊 EVALUATED |

### FlowWM K=4 ✅ EVALUATED
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| - | 42 | `outputs/2025-12-29/09-17-18/logs/best_policy.pt` | 849.81 | 📊 EVALUATED |
| - | 123 | `outputs/2025-12-29/10-17-08/logs/best_policy.pt` | 1239.04 | 📊 EVALUATED |
| - | 456 | `outputs/2025-12-29/11-02-54/logs/best_policy.pt` | 823.06 | 📊 EVALUATED |

### FlowWM K=8 ✅ EVALUATED
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| - | 42 | `outputs/2025-12-29/11-22-48/logs/best_policy.pt` | 1107.08 | 📊 EVALUATED |
| - | 123 | `outputs/2025-12-29/13-55-00/logs/best_policy.pt` | 1197.42 | 📊 EVALUATED |
| - | 456 | `outputs/2025-12-29/14-27-09/logs/best_policy.pt` | 1244.37 | 📊 EVALUATED |

### FlowPolicy ✅ EVALUATED
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| - | 42 | `outputs/2025-12-29/14-28-54/logs/best_policy.pt` | 45.77 | 📊 EVALUATED |
| - | 123 | `outputs/2025-12-29/14-46-31/logs/best_policy.pt` | 19.73 | 📊 EVALUATED |
| - | 456 | `outputs/2025-12-29/14-56-22/logs/best_policy.pt` | 217.09 | 📊 EVALUATED |

### FullFlow K=4 ✅ EVALUATED
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| - | 42 | `outputs/2025-12-29/15-43-12/logs/best_policy.pt` | 245.34 | 📊 EVALUATED |
| - | 123 | `outputs/2025-12-29/16-34-47/logs/best_policy.pt` | 220.51 | 📊 EVALUATED |
| - | 456 | `outputs/2025-12-29/16-53-26/logs/best_policy.pt` | 1089.82 | 📊 EVALUATED |

---

## Anymal (Dec 28-29, 2025) ⚠️ MISSING BASELINE

> [!WARNING]
> **Anymal Baseline checkpoints were deleted during disk cleanup on Dec 31.**
> Original job IDs: 3080227, 3080228, 3080229
> Need to retrain.

### FlowPolicy ✅ EVALUATED
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| - | 42 | `outputs/2025-12-28/22-43-26/logs/best_policy.pt` | 33.13 | 📊 EVALUATED |
| - | 123 | `outputs/2025-12-29/00-10-02/logs/best_policy.pt` | 20.31 | 📊 EVALUATED |
| - | 456 | `outputs/2025-12-28/17-34-31/logs/best_policy.pt` | 47.05 | 📊 EVALUATED |

### FlowWM K=4 ✅ EVALUATED
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| - | 42 | `outputs/2025-12-28/06-54-04/logs/best_policy.pt` | 29.35 | 📊 EVALUATED |
| - | 123 | `outputs/2025-12-28/22-27-32/logs/best_policy.pt` | 15.33 | 📊 EVALUATED |
| - | 456 | `outputs/2025-12-28/22-28-56/logs/best_policy.pt` | 23.97 | 📊 EVALUATED |

### FlowWM K=8 ✅ EVALUATED
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| - | 42 | `outputs/2025-12-29/05-00-51/logs/best_policy.pt` | 14.76 | 📊 EVALUATED |
| - | 123 | `outputs/2025-12-29/03-24-05/logs/best_policy.pt` | 23.85 | 📊 EVALUATED |

### FullFlow K=4 ✅ EVALUATED
| Job ID | Seed | Checkpoint Path | Mean Reward | Status |
|--------|------|-----------------|-------------|--------|
| - | 42 | `outputs/2025-12-28/21-42-56/logs/best_policy.pt` | 22.40 | 📊 EVALUATED |
| - | 123 | `outputs/2025-12-28/23-27-25/logs/best_policy.pt` | 28.49 | 📊 EVALUATED |
| - | 456 | `outputs/2025-12-29/02-14-20/logs/best_policy.pt` | 17.28 | 📊 EVALUATED |

---

# EVALUATION RESULTS

## Final Aggregated CSV
**Path**: `/storage/scratch1/9/eliu354/flow_mbpo/eval_results/final_eval_results.csv`

### Summary by Task (as of Jan 1, 2026)

| Task | Variant | Avg Reward | Std | Seeds |
|------|---------|------------|-----|-------|
| **Ant** | FlowWM_K8_euler | **1182.96** | 69.78 | 3 |
| Ant | FlowWM_K4_heun | 970.64 | 232.83 | 3 |
| Ant | FlowWM_K2_heun | 763.23 | 562.29 | 3 |
| Ant | FullFlow_K4 | 518.56 | 494.88 | 3 |
| Ant | Baseline | 426.15 | 645.39 | 3 |
| **Anymal** | FlowPolicy | 33.50 | 13.37 | 3 |
| Anymal | FlowWM_K4_heun | 22.88 | 7.07 | 3 |
| Anymal | FullFlow_K4 | 22.72 | 5.61 | 3 |
| **Humanoid** | Baseline | **62.09** | 22.75 | 3 |
| Humanoid | FlowPolicy | 58.90 | - | 1 |
| Humanoid | FlowWM_K4_heun | 44.78 | 14.57 | 2 |

---

# SCRIPTS REFERENCE

## Training Scripts Location
- Anymal: `scripts/anymal/`
- Ant: `scripts/ant/`
- Humanoid: `scripts/humanoid/`

## Evaluation Scripts
- Main eval: `/storage/scratch1/9/eliu354/flow_mbpo/scripts/eval_pwm.py`
- Batch submit: `/storage/scratch1/9/eliu354/flow_mbpo/scripts/submit_all_evals.py`
- Aggregate: `/storage/scratch1/9/eliu354/flow_mbpo/scripts/aggregate_eval_results.py`
