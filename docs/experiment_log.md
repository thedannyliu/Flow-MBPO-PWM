# Experiment Log

> **Purpose**: The authoritative registry for all Flow-MBPO experiments.
> **Last Updated**: Jan 22, 2026

> [!CAUTION]
> **Hardware Alert**: Node **atl1-1-03-004** is defective (ECC Errors). All jobs/evals on this node fail immediately.
> **Training Alert**: Ant experiments show bifurcation (R~1200 vs R~20). Likely due to `rew_rms: True`.

> **Status Legend**:
> - 🕐 **PENDING**: Waiting in queue.
> - 🔄 **RUNNING**: Currently executing.
> - ✅ **COMPLETED**: Finished successully (Exit 0).
> - ❌ **FAILED**: Failed (OOM, Error, or Canceled).
> - 📊 **EVALUATED**: Evaluation complete.
> - **Hardware**: SLURM Node ID (e.g., `atl1-1-03-004`).
> - **Runtime**: Duration (HH:MM:SS).
> - **Storage**: Checkpoint path relative to project root.

---

# ALIGNED SINGLE-TASK EXPERIMENTS SUMMARY (Jan 22, 2026)

> **Full Report**: `scripts/eval/all_dflex_experiments.csv` (93 evaluated runs)
> **Config**: All aligned runs use `rew_rms: True` for fair comparison.

## Completion Status by Task/Variant

| Task | Variant | Completed | Failed | Evaluated | Notes |
|------|---------|-----------|--------|-----------|-------|
| **Ant** | Baseline | 10/10 | 0 | 10 | s0-9 complete |
| **Ant** | FlowWM_K8 | 9/10 | 1 | 8 | s8 (V7) complete, s0 failed |
| **Anymal** | Baseline | 3/10 | 7 | 1 | s3,s8 (V7) complete |
| **Anymal** | FlowPolicy | 8/10 | 2 | 6 | s6 (V7) complete |
| **Humanoid** | Baseline | 4/10 | 6 | 3 | s2,s5,s7 (V7) complete |
| **Humanoid** | FlowPolicy | 5/10 | 5 | 5 | s0,s9 (V7) complete |

## Aggregated Results (Aligned Phase Only)

| Task | Variant | N | Mean Reward | Std | Min | Max |
|------|---------|---|-------------|-----|-----|-----|
| **Ant** | Baseline | 10 | ~613 | 538 | 17 | 1265 |
| **Ant** | FlowWM_K8 | 8 | ~580 | - | - | - |
| **Anymal** | Baseline | 1 | 21.06 | - | - | - |
| **Anymal** | FlowPolicy | 6 | ~23 | - | - | - |
| **Humanoid** | Baseline | 3 | 59.45 | 16 | 42 | 72 |
| **Humanoid** | FlowPolicy | 5 | ~65 | - | - | - |

---

# DETAILED ALIGNED RUN REGISTRY

## Ant Aligned (20 runs total)

### Ant Baseline (10/10 complete)
| Seed | Timestamp | Status | Reward | Notes |
|------|-----------|--------|--------|-------|
| 0 | - | ❌ | - | V1 failed (CUDA busy) |
| 1 | 2026-01-03_22-11-18 | 📊 | 1134.56 | V2 |
| 2 | 2026-01-04_01-11-20 | 📊 | 23.63 | V2 (collapsed) |
| 3 | 2026-01-04_04-16-39 | 📊 | 16.99 | V2 (collapsed) |
| 4 | 2026-01-04_05-55-51 | 📊 | 327.84 | V2 |
| 5 | 2026-01-04_08-58-31 | 📊 | 1041.12 | V2 |
| 6 | 2026-01-04_11-58-00 | 📊 | 529.10 | V2 |
| 7 | 2026-01-04_15-01-48 | 📊 | 1234.27 | V2 |
| 8 | 2026-01-05_02-28-10 | 📊 | 1265.35 | V6 |
| 9 | - | ❌ | - | V6 failed |

### Ant FlowWM_K8 (9/10 complete)
| Seed | Timestamp | Status | Reward | Notes |
|------|-----------|--------|--------|-------|
| 0 | - | ❌ | - | V1 failed |
| 1 | 2026-01-03_22-13-21 | 📊 | ~1200 | V2 |
| 2 | 2026-01-04_01-11-49 | 📊 | 1211.57 | V2 |
| 3 | 2026-01-04_05-45-36 | 📊 | 25.88 | V2 (collapsed) |
| 4 | 2026-01-04_07-19-35 | 📊 | 19.29 | V2 (collapsed) |
| 5 | 2026-01-04_09-01-14 | 📊 | 50.77 | V2 (collapsed) |
| 6 | 2026-01-04_13-31-28 | 📊 | 55.56 | V2 (collapsed) |
| 7 | 2026-01-04_15-07-14 | 📊 | 1216.03 | V2 |
| 8 | 2026-01-07_04-27-05 | ✅ | - | V7 (Eval Pending) |
| 9 | 2026-01-05_23-36-26 | 📊 | - | V6 (Pending Eval) |

## Anymal Aligned (20 runs total)

### Anymal Baseline (3/10 complete)
| Seed | Timestamp | Status | Reward | Notes |
|------|-----------|--------|--------|-------|
| 0 | 2026-01-05_19-27-00 | 📊 | 21.06 | V6 |
| 1 | 2026-01-05_19-26-59 | 📊 | - | V6 |
| 2 | - | ❌ | - | V7 Failed |
| 3 | 2026-01-07_04-09-12 | ✅ | - | V7 (Eval Pending) |
| 4 | - | ❌ | - | V7 Failed |
| 5 | - | ❌ | - | V7 Failed |
| 6 | - | ❌ | - | V7 Failed |
| 7 | - | ❌ | - | V7 Failed |
| 8 | 2026-01-07_04-29-51 | ✅ | - | V7 (Eval Pending) |
| 9 | 2026-01-04_23-50-40 | 📊 | - | V6 Legacy |

### Anymal FlowPolicy (8/10 complete)
| Seed | Timestamp | Status | Reward | Notes |
|------|-----------|--------|--------|-------|
| 0 | 2026-01-04_21-18-10 | 📊 | - | V6 |
| 1 | 2026-01-06_04-28-13 | 📊 | - | V6 |
| 2 | - | ❌ | - | V7 Failed |
| 3 | 2026-01-04_21-18-11 | 📊 | - | V6 |
| 4 | 2026-01-04_22-51-07 | 📊 | - | V6 |
| 5 | 2026-01-05_07-25-01 | 📊 | - | V6 |
| 6 | 2026-01-07_04-50-48 | ✅ | - | V7 (Eval Pending) |
| 7 | 2026-01-05_09-05-05 | 📊 | - | V6 |
| 8 | - | ❌ | - | Missing |
| 9 | 2026-01-05_09-57-09 | 📊 | - | V6 |

## Humanoid Aligned (20 runs total)

### Humanoid Baseline (4/10 complete)
| Seed | Timestamp | Status | Reward | Notes |
|------|-----------|--------|--------|-------|
| 0 | - | ❌ | - | V7 Failed |
| 1 | - | ❌ | - | V7 Failed |
| 2 | 2026-01-07_04-10-01 | ✅ | - | V7 (Eval Pending) |
| 3 | 2026-01-05_12-29-30 | 📊 | 72.36 | V6 |
| 4 | 2026-01-05_01-24-41 | 📊 | 41.59 | V6 |
| 5 | 2026-01-07_06-29-49 | ✅ | - | V7 (Eval Pending) |
| 6 | 2026-01-05_01-24-41 | 📊 | 64.41 | V6 |
| 7 | 2026-01-07_07-18-58 | ✅ | - | V7 (Eval Pending) |
| 8 | - | ❌ | - | V7 Failed |
| 9 | - | ❌ | - | V7 Failed |

### Humanoid FlowPolicy (5/10 complete)
| Seed | Timestamp | Status | Reward | Notes |
|------|-----------|--------|--------|-------|
| 0 | 2026-01-07_07-39-16 | ✅ | - | V7 (Eval Pending) |
| 1 | 2026-01-05_11-36-47 | 📊 | - | V6 |
| 2 | - | ❌ | - | V7 Failed |
| 3 | - | ❌ | - | V7 Failed |
| 4 | 2026-01-05_14-12-35 | 📊 | - | V6 |
| 5 | - | ❌ | - | V7 Failed |
| 6 | 2026-01-05_15-05-04 | 📊 | - | V6 |
| 7 | - | ❌ | - | V7 Failed |
| 8 | 2026-01-05_15-31-00 | 📊 | - | V6 |
| 9 | 2026-01-07_08-55-52 | ✅ | - | V7 (Eval Pending) |

---

# COMPLETED EXPERIMENTS (Legacy)

## Humanoid (Dec 29-30, 2025)

### Baseline
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| 3101831 | 42 | ~2h20m | Unknown | 37.82 | `outputs/2025-12-29/20-17-36` |
| 3101832 | 123 | ~2h20m | Unknown | 65.52 | `outputs/2025-12-29/20-20-53` |
| 3101833 | 456 | ~2h20m | Unknown | 82.93 | `outputs/2025-12-29/20-31-30` |

### FlowWM K=4
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| 3104842 | 42 | ~5h | Unknown | 55.09 | `outputs/2025-12-29/22-54-39` |
| 3104843 | 123 | ~5h | Unknown | 34.48 | `outputs/2025-12-29/22-54-42` |

### FlowWM K=2
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| 3104846 | 123 | ~3h40m | Unknown | 34.09 | `outputs/2025-12-29/22-54-41` |

### FlowWM K=8
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| 3107946 | 42 | ~5h | Unknown | 36.76 | `outputs/2025-12-30/04-20-56` |
| 3107947 | 123 | ~5h | Unknown | 30.70 | `outputs/2025-12-30/04-20-58` |

### FlowPolicy
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| 3125488 | 42 | Unknown | Unknown | 58.90 | `outputs/2025-12-29/20-18-50` |

---

## Ant (Dec 29, 2025)

### Baseline
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| - | 42 | Unknown | Unknown | **1170.49** | `outputs/2025-12-29/07-10-46` |
| - | 123 | Unknown | Unknown | 22.50 | `outputs/2025-12-29/08-18-33` |
| - | 456 | Unknown | Unknown | 85.45 | `outputs/2025-12-29/09-16-25` |

### FlowWM K=8
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| - | 42 | Unknown | Unknown | 1107.08 | `outputs/2025-12-29/11-22-48` |
| - | 123 | Unknown | Unknown | **1197.42** | `outputs/2025-12-29/13-55-00` |
| - | 456 | Unknown | Unknown | **1244.37** | `outputs/2025-12-29/14-27-09` |

### FlowPolicy
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| - | 42 | Unknown | Unknown | 45.77 | `outputs/2025-12-29/14-28-54` |
| - | 123 | Unknown | Unknown | 19.73 | `outputs/2025-12-29/14-46-31` |
| - | 456 | Unknown | Unknown | 217.09 | `outputs/2025-12-29/14-56-22` |

---

## Anymal (Dec 28-29, 2025)

> **Note**: Baseline checkpoints were lost.

### FlowPolicy
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| - | 42 | Unknown | Unknown | 33.13 | `outputs/2025-12-28/22-43-26` |
| - | 123 | Unknown | Unknown | 20.31 | `outputs/2025-12-29/00-10-02` |
| - | 456 | Unknown | Unknown | 47.05 | `outputs/2025-12-28/17-34-31` |

### FlowWM K=4
| Job ID | Seed | Runtime | Hardware | Eval Reward | Storage |
|--------|------|---------|----------|-------------|---------|
| - | 42 | Unknown | Unknown | 29.35 | `outputs/2025-12-28/06-54-04` |
| - | 123 | Unknown | Unknown | 15.33 | `outputs/2025-12-28/22-27-32` |
| - | 456 | Unknown | Unknown | 23.97 | `outputs/2025-12-28/22-28-56` |
