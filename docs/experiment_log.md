# Experiment Log

Purpose: Registry for all training jobs with Slurm IDs, configs, seeds, wandb links, and status.

---

## Status Summary (Dec 29, 8:22 PM)

### Experiment Overview
| Environment | WandB Project | Total Jobs | Completed | Running |
|-------------|---------------|------------|-----------|---------|
| Anymal | flow-mbpo-single | 27 | 26 | 1 |
| Ant | flow-mbpo-single-task-ant | 21 | 17 | 4 |
| Humanoid | flow-mbpo-single-task-Humanoid | 22 | 4 (smoke) | 18 (queued) |

---

# HUMANOID EXPERIMENTS (Dec 29)

## WandB: flow-mbpo-single-task-Humanoid

### Smoke Tests ✅ COMPLETED
| Job ID | Variant | Status | Runtime |
|--------|---------|--------|---------|
| 3101822 | Baseline | ✅ | 0:00:46 |
| 3101823 | Flow WM | ✅ | 0:01:45 |
| 3101824 | Flow Policy | ✅ | 0:00:33 |
| 3101825 | Full Flow | ✅ | 0:01:45 |

### Training Jobs (18 jobs submitted)

#### Baseline (MLP WM + MLP Policy)
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3101831 | 42 | Humanoid_Baseline_MLP_s42 |
| 3101832 | 123 | Humanoid_Baseline_MLP_s123 |
| 3101833 | 456 | Humanoid_Baseline_MLP_s456 |

#### Flow WM K=4 Heun
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3101834 | 42 | Humanoid_FlowWM_K4Heun_s42 |
| 3101835 | 123 | Humanoid_FlowWM_K4Heun_s123 |
| 3101836 | 456 | Humanoid_FlowWM_K4Heun_s456 |

#### Flow WM K=2 Heun
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3101837 | 42 | Humanoid_FlowWM_K2Heun_s42 |
| 3101838 | 123 | Humanoid_FlowWM_K2Heun_s123 |
| 3101839 | 456 | Humanoid_FlowWM_K2Heun_s456 |

#### Flow WM K=8 Euler
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3101840 | 42 | Humanoid_FlowWM_K8Euler_s42 |
| 3101841 | 123 | Humanoid_FlowWM_K8Euler_s123 |
| 3101842 | 456 | Humanoid_FlowWM_K8Euler_s456 |

#### MLP WM + Flow Policy
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3101843 | 42 | Humanoid_FlowPolicy_MLPWM_s42 |
| 3101844 | 123 | Humanoid_FlowPolicy_MLPWM_s123 |
| 3101845 | 456 | Humanoid_FlowPolicy_MLPWM_s456 |

#### Full Flow
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3101846 | 42 | Humanoid_FullFlow_FlowWM_FlowPol_s42 |
| 3101847 | 123 | Humanoid_FullFlow_FlowWM_FlowPol_s123 |
| 3101848 | 456 | Humanoid_FullFlow_FlowWM_FlowPol_s456 |

---

# ANT EXPERIMENTS (Dec 29)

## WandB: flow-mbpo-single-task-ant

### Completed ✅ (17 jobs)
- Smoke tests: 4 ✅ (3093139-3093142)
- Baseline: 3 ✅ (3093165-3093167)
- Flow WM K=4: 3 ✅ (3093168-3093170)
- Flow WM K=2: 3 ✅ (3093171-3093173)
- Flow WM K=8: 3 ✅ (3093174-3093176)
- Flow Policy: 3 ✅ (3093177-3093179)

### Running (4 jobs)
- Full Flow: 3093180 (s42), 3099532 (s123), 3099534 (s456)

---

# ANYMAL EXPERIMENTS (Dec 28-29)

## WandB: flow-mbpo-single

### Completed ✅ (26 jobs)
All core experiments and ablations completed.

### Running (1 job)
- 3091999: Flow WM HighLR s456 (~15h elapsed)

---

## Resource Allocation
- **Memory**: 400GB
- **Time**: 40 hours
- **Partition**: gpu-l40s
