# Experiment Log

> **Purpose**: The authoritative registry for all Flow-MBPO experiments.
> **Last Updated**: Jan 5, 2026

> [!CAUTION]
> **Hardware Alert**: Node **atl1-1-03-004** is defective (ECC Errors). All jobs/evals on this node fail immediately.
> **Training Alert**: Ant experiments show bifurcation (R~1200 vs R~20). Likely due to `rew_rms: True`.

---

# ACTIVE EXPERIMENTS (Jan 3-5, 2026)

## Aligned V5 (Resubmission - Recovery)
> **Goal**: Resubmit jobs that hung or OOM'd in V2/V4.
> **Config**: Packed 3 jobs/node (Safe).

| Batch | Content (Seeds) | Job ID | Status | Notes |
|-------|-----------------|--------|--------|-------|
| V5_Batch0 | Ant Flow s5, Ant Base s8, Ant Flow s8 | 3162639 | 🕐 PENDING | 3 seeds |
| V5_Batch1 | Ant Base s9, Ant Flow s9, Any Base s0 | 3162640 | 🕐 PENDING | 3 seeds |
| V5_Batch2 | Any Flow s0, Any Base s1, Any Flow s1 | 3162641 | 🕐 PENDING | 3 seeds |

## Aligned V4 (Packed - 4 jobs/node)
> **Goal**: Run remaining Anymal/Humanoid/Scaling jobs safely.
> **Status**: Mix of Completed, Running, and Pending.

| Batch | Job ID | Status | Runtime | Hardware | Notes |
|-------|--------|--------|---------|----------|-------|
| Packed_0 | 3153741 | ❌ FAILED (OOM) | 04:56:46 | atl1-1-03-004 | Resubmitted in V5 |
| Packed_1 | 3153742 | ✅ COMPLETED | 02:31:56 | atl1-1-03-004 | Evaluated |
| Packed_2 | 3153743 | ✅ COMPLETED | 02:31:23 | atl1-1-03-004 | Evaluated |
| Packed_3 | 3153744 | ✅ COMPLETED | 02:32:09 | atl1-1-03-004 | Evaluated |
| Packed_4 | 3153745 | ✅ COMPLETED | 02:31:41 | atl1-1-03-004 | Evaluated |
| Packed_5 | 3153746 | ✅ COMPLETED | 02:32:17 | atl1-1-03-004 | Evaluated |
| Packed_6 | 3153747 | ✅ COMPLETED | 02:35:47 | atl1-1-03-004 | Evaluated |
| Packed_7 | 3153748 | ✅ COMPLETED | 02:35:33 | atl1-1-03-004 | Evaluated |
| Packed_8 | 3153749 | ✅ COMPLETED | 02:37:06 | atl1-1-03-004 | Evaluated |
| Packed_9 | 3153750 | ✅ COMPLETED | 02:36:05 | atl1-1-03-004 | Evaluated |
| Packed_10 | 3153751 | ✅ COMPLETED | 02:35:41 | atl1-1-03-007 | Evaluated |
| Packed_11 | 3153752 | 🔄 RUNNING | 02:20:20+ | atl1-1-03-004 | Last batch |

## Aligned V2 (Ant Exclusive)
> **Goal**: Ant 10-seed ablation.
> **Status**: Completed (Mixed Success).

| Job ID | Variant | Seed | Status | Runtime | Hardware | Eval Reward |
|--------|---------|------|--------|---------|----------|-------------|
| 3143914 | Baseline | 0 | ✅ COMPLETED | 03:00:02 | atl1-1-03-004 | ~29.7 |
| 3143915 | FlowWM K=8 | 0 | ✅ COMPLETED | 07:44:34 | atl1-1-01-010 | ~25.2 |
| 3143916 | Baseline | 1 | ✅ COMPLETED | 03:00:30 | atl1-1-01-010 | - |
| 3143917 | FlowWM K=8 | 1 | ✅ COMPLETED | 07:32:14 | atl1-1-03-004 | - |
| 3143918 | Baseline | 2 | ✅ COMPLETED | 03:05:19 | atl1-1-03-004 | - |
| 3143919 | FlowWM K=8 | 2 | ✅ COMPLETED | 07:46:42 | atl1-1-03-007 | - |
| 3143920 | Baseline | 3 | ✅ COMPLETED | 03:02:55 | atl1-1-03-004 | - |
| 3143921 | FlowWM K=8 | 3 | ✅ COMPLETED | 07:45:52 | atl1-1-03-004 | - |
| 3143924 | Baseline | 4 | ✅ COMPLETED | 02:59:28 | atl1-1-03-007 | - |
| 3143925 | FlowWM K=8 | 5 | ❌ CANCELED | 34h+ | atl1-1-01-010 | HUNG (Resubmitted V5) |
| 3143926 | Baseline | 5 | ✅ COMPLETED | 03:03:46 | atl1-1-03-007 | - |
| 3143927 | FlowWM K=8 | 6 | ✅ COMPLETED | 07:46:42 | atl1-1-03-004 | - |
| 3143930 | Baseline | 7 | ✅ COMPLETED | - | - | - |
| 3143931 | FlowWM K=8 | 7 | ✅ COMPLETED | - | - | - |

---

# EVALUATION RESULTS (Aggregated)

**Latest CSV**: `/storage/scratch1/9/eliu354/flow_mbpo/eval_results/final_eval_results.csv`

| Task | Variant | Seed | Result Info |
|------|---------|------|-------------|
| **Ant** | **Baseline** | 123 | R=22.50 (Collapsed) |
| Ant | Baseline | 42 | **R=1170.49 (SOTA)** |
| Ant | Baseline | 456 | R=85.45 (Collapsed) |
| **Ant** | **FlowPolicy** | 7 | **R=1234.27 (SOTA)** |
| Ant | FlowPolicy | 42 | **R=1157.68 (SOTA)** |
| Ant | FlowPolicy | 2 | R=23.63 (Collapsed) |
| Ant | FlowPolicy | 3 | R=203.62 (Poor) |
| **Ant** | **FlowWM K=8** | 456 | **R=1244.37 (SOTA)** |
| Ant | FlowWM K=8 | 123 | **R=1197.42 (SOTA)** |

---

# ARCHIVE (Failures)

## Aligned V3 (Packed - 7 jobs/node)
> **Status**: ❌ FAILED (OOM). Too aggressive.
> **Resolution**: Reduced to 4 jobs/node in V4.

| Job ID | Status | Runtime | Outcome |
|--------|--------|---------|---------|
| 3150252 | ❌ FAILED | 05:25:22 | OOM (Replaced by V4/V5) |
| 3150258 | ❌ FAILED | 13:24:28 | OOM (Replaced by V4) |
| 3150253-57| ✅ COMPLETED | ~2.5h | Successful runs kept |

## Aligned V1
> **Status**: ❌ FAILED (CUDA Busy). Replaced by V2.
