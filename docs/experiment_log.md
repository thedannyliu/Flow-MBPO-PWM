# Experiment Log

Purpose: Registry for all training jobs with Slurm IDs, configs, seeds, wandb links, and status.

---

## Session: 2025-12-29 - Ant Experiments

### WandB Dashboards
- **Anymal**: `flow-mbpo-single` - https://wandb.ai/danny010324/flow-mbpo-single
- **Ant**: `flow-mbpo-single-task-ant` - https://wandb.ai/danny010324/flow-mbpo-single-task-ant

---

# ANT EXPERIMENTS (Dec 29)

## Smoke Tests
| Job ID | Variant | Status |
|--------|---------|--------|
| 3093139 | Baseline | PENDING |
| 3093140 | Flow WM | PENDING |
| 3093141 | Flow Policy | PENDING |
| 3093142 | Full Flow | PENDING |

## Full Training (18 jobs)

### Baseline (MLP WM + MLP Policy)
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3093165 | 42 | Ant_Baseline_MLP_s42 |
| 3093166 | 123 | Ant_Baseline_MLP_s123 |
| 3093167 | 456 | Ant_Baseline_MLP_s456 |

### Flow WM K=4 Heun
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3093168 | 42 | Ant_FlowWM_K4Heun_s42 |
| 3093169 | 123 | Ant_FlowWM_K4Heun_s123 |
| 3093170 | 456 | Ant_FlowWM_K4Heun_s456 |

### Flow WM K=2 Heun
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3093171 | 42 | Ant_FlowWM_K2Heun_s42 |
| 3093172 | 123 | Ant_FlowWM_K2Heun_s123 |
| 3093173 | 456 | Ant_FlowWM_K2Heun_s456 |

### Flow WM K=8 Euler
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3093174 | 42 | Ant_FlowWM_K8Euler_s42 |
| 3093175 | 123 | Ant_FlowWM_K8Euler_s123 |
| 3093176 | 456 | Ant_FlowWM_K8Euler_s456 |

### MLP WM + Flow Policy
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3093177 | 42 | Ant_FlowPolicy_MLPWM_s42 |
| 3093178 | 123 | Ant_FlowPolicy_MLPWM_s123 |
| 3093179 | 456 | Ant_FlowPolicy_MLPWM_s456 |

### Full Flow (Flow WM + Flow Policy)
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3093180 | 42 | Ant_FullFlow_FlowWM_FlowPol_s42 |
| 3093181 | 123 | Ant_FullFlow_FlowWM_FlowPol_s123 |
| 3093182 | 456 | Ant_FullFlow_FlowWM_FlowPol_s456 |

---

# ANYMAL EXPERIMENTS (Dec 28-29)

## Completed ✅
| Variant | Seeds | Job IDs |
|---------|-------|---------|
| Baseline | 3 | 3080227-3080229 |
| Flow WM K=4 Heun | 3 | 3082681-3082683 |
| Flow WM K=2 Heun | 3 | 3082684-3082686 |
| Flow WM K=8 Euler | 3 | 3084837-3084839 |
| MLP WM + Flow Policy | 3 | 3084840-3084842 |
| Full Flow s42 | 1 | 3084843 |
| Flow WM H=8 | 3 | 3087697-3087699 |
| Flow Policy H=8 | 3 | 3087703-3087705 |
| Flow WM LowLR | 3 | 3087700-3087702 |
| Full Flow s123, s456 | 2 | 3087706-3087707 |

## Running/Pending
| Variant | Job IDs | Status |
|---------|---------|--------|
| Flow WM StrongReg | 3093114-3093116 | Running |
| Flow WM HighLR | 3091997-3091999 | Running |
| Full Flow H=8 | 3091993-3091996 | Running |

---

## Resource Allocation
- **Memory**: 400GB
- **Time**: 40 hours
- **Partition**: gpu-l40s
- **Account**: gts-agarg35-ideas_l40s
