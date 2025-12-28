# Experiment Log

Purpose: Registry for all training jobs with Slurm IDs, configs, seeds, wandb links, and status.

---

## Session: 2025-12-28 Evening - Complete Experiment Status

### Overall Summary
| Variant | Seeds | Status |
|---------|-------|--------|
| Baseline (MLP WM + MLP Policy) | 42, 123, 456 | ✅ COMPLETED (Dec 27) |
| Flow WM K=4 Heun + MLP Policy | 42, 123, 456 | ✅ COMPLETED |
| Flow WM K=2 Heun + MLP Policy | 42, 123, 456 | ✅ COMPLETED |
| Flow WM K=8 Euler + MLP Policy | 42, 123, 456 | 🔄 RUNNING |
| MLP WM + Flow Policy | 42, 123, 456 | 🔄 RUNNING |
| Full Flow (Flow WM + Flow Policy) | 42, 123, 456 | 🔄 RUNNING |

### WandB Dashboard
- **Project**: `flow-mbpo-single`
- **URL**: https://wandb.ai/danny010324/flow-mbpo-single

---

## Completed Experiments

### Baseline (MLP WM + MLP Policy) - Dec 27
| Job ID | Seed | Status | Runtime |
|--------|------|--------|---------|
| 3080227 | 42 | ✅ COMPLETED | ~2h |
| 3080228 | 123 | ✅ COMPLETED | ~2h |
| 3080229 | 456 | ✅ COMPLETED | ~2h |

### Flow WM K=4 Heun - Dec 28
| Job ID | Seed | Status | Runtime |
|--------|------|--------|---------|
| 3082681 | 42 | ✅ COMPLETED | 4:55:05 |
| 3082682 | 123 | ✅ COMPLETED | 4:54:37 |
| 3082683 | 456 | ✅ COMPLETED | 4:53:58 |

### Flow WM K=2 Heun - Dec 28
| Job ID | Seed | Status | Runtime |
|--------|------|--------|---------|
| 3082684 | 42 | ✅ COMPLETED | 3:36:07 |
| 3082685 | 123 | ✅ COMPLETED | 3:38:12 |
| 3082686 | 456 | ✅ COMPLETED | 4:27:55 |

---

## Currently Running (Dec 28 Evening)

### Flow WM K=8 Euler (400GB memory)
| Job ID | Seed | WandB Name | Status |
|--------|------|------------|--------|
| 3084837 | 42 | Anymal_FlowWM_K8Euler_s42 | RUNNING |
| 3084838 | 123 | Anymal_FlowWM_K8Euler_s123 | RUNNING |
| 3084839 | 456 | Anymal_FlowWM_K8Euler_s456 | RUNNING |

### MLP WM + Flow Policy (400GB memory)
| Job ID | Seed | WandB Name | Status |
|--------|------|------------|--------|
| 3084840 | 42 | Anymal_FlowPolicy_MLPWM_s42 | RUNNING |
| 3084841 | 123 | Anymal_FlowPolicy_MLPWM_s123 | RUNNING |
| 3084842 | 456 | Anymal_FlowPolicy_MLPWM_s456 | RUNNING |

### Full Flow: Flow WM + Flow Policy (400GB memory)
| Job ID | Seed | WandB Name | Status |
|--------|------|------------|--------|
| 3084843 | 42 | Anymal_FullFlow_FlowWM_FlowPol_s42 | RUNNING |
| 3084847 | 123 | Anymal_FullFlow_FlowWM_FlowPol_s123 | RUNNING |
| 3084848 | 456 | Anymal_FullFlow_FlowWM_FlowPol_s456 | PENDING |

---

## Configuration Reference
| Variant | Config File | WM Type | Policy Type |
|---------|-------------|---------|-------------|
| Baseline | pwm_5M_baseline_final | MLP | MLP (Gaussian) |
| Flow WM K=2 | pwm_5M_flow_v1_substeps2 | Flow (Heun, K=2) | MLP |
| Flow WM K=4 | pwm_5M_flow_v2_substeps4 | Flow (Heun, K=4) | MLP |
| Flow WM K=8 | pwm_5M_flow_v3_substeps8_euler | Flow (Euler, K=8) | MLP |
| Flow Policy | pwm_5M_flowpolicy | MLP | Flow ODE |
| Full Flow | pwm_5M_fullflow | Flow (Heun, K=4) | Flow ODE |

---

## Resource Configuration
- **Memory**: 400GB per job
- **Time**: 40 hours
- **Partition**: gpu-l40s
- **Account**: gts-agarg35-ideas_l40s
