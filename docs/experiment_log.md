# Experiment Log

Purpose: Registry for all training jobs with Slurm IDs, configs, seeds, wandb links, and status.

---

## Session: 2025-12-28 - All Experiments (400GB Memory)

### Summary
- **Baseline**: COMPLETED (3080227-3080229) ✅
- **Flow WM + MLP Policy**: 9 jobs submitted (3082681-3082689)
- **MLP WM + Flow Policy**: 3 jobs submitted (3082690-3082692)
- **Full Flow (Flow WM + Flow Policy)**: 3 jobs submitted (3082693-3082695)
- **TOTAL**: 15 new jobs with 400GB memory

### WandB Dashboard
- **Project**: `flow-mbpo-single`
- **URL**: https://wandb.ai/danny010324/flow-mbpo-single

---

## Completed: Baseline (MLP WM + MLP Policy)
| Job ID | Seed | Status |
|--------|------|--------|
| 3080227 | 42 | ✅ COMPLETED |
| 3080228 | 123 | ✅ COMPLETED |
| 3080229 | 456 | ✅ COMPLETED |

---

## Flow WM + MLP Policy (400GB memory)
| Job ID | Config | Seed | WandB Name | Status |
|--------|--------|------|------------|--------|
| 3082681 | K=4 Heun | 42 | Anymal_FlowWM_K4Heun_s42 | RUNNING |
| 3082682 | K=4 Heun | 123 | Anymal_FlowWM_K4Heun_s123 | RUNNING |
| 3082683 | K=4 Heun | 456 | Anymal_FlowWM_K4Heun_s456 | RUNNING |
| 3082684 | K=2 Heun | 42 | Anymal_FlowWM_K2Heun_s42 | RUNNING |
| 3082685 | K=2 Heun | 123 | Anymal_FlowWM_K2Heun_s123 | RUNNING |
| 3082686 | K=2 Heun | 456 | Anymal_FlowWM_K2Heun_s456 | RUNNING |
| 3082687 | K=8 Euler | 42 | Anymal_FlowWM_K8Euler_s42 | RUNNING |
| 3082688 | K=8 Euler | 123 | Anymal_FlowWM_K8Euler_s123 | PENDING |
| 3082689 | K=8 Euler | 456 | Anymal_FlowWM_K8Euler_s456 | PENDING |

---

## MLP WM + Flow Policy (400GB memory)
| Job ID | Seed | WandB Name | Status |
|--------|------|------------|--------|
| 3082690 | 42 | Anymal_FlowPolicy_MLPWM_s42 | PENDING |
| 3082691 | 123 | Anymal_FlowPolicy_MLPWM_s123 | PENDING |
| 3082692 | 456 | Anymal_FlowPolicy_MLPWM_s456 | PENDING |

---

## Full Flow: Flow WM + Flow Policy (400GB memory)
| Job ID | Seed | WandB Name | Status |
|--------|------|------------|--------|
| 3082693 | 42 | Anymal_FullFlow_FlowWM_FlowPol_s42 | PENDING |
| 3082694 | 123 | Anymal_FullFlow_FlowWM_FlowPol_s123 | PENDING |
| 3082695 | 456 | Anymal_FullFlow_FlowWM_FlowPol_s456 | PENDING |

---

## Configuration Summary
| Variant | Config File | WM | Policy |
|---------|-------------|-----|-------|
| Baseline | pwm_5M_baseline_final | MLP | MLP |
| Flow WM K=2 | pwm_5M_flow_v1_substeps2 | Flow (Heun, K=2) | MLP |
| Flow WM K=4 | pwm_5M_flow_v2_substeps4 | Flow (Heun, K=4) | MLP |
| Flow WM K=8 | pwm_5M_flow_v3_substeps8_euler | Flow (Euler, K=8) | MLP |
| Flow Policy | pwm_5M_flowpolicy | MLP | Flow ODE |
| Full Flow | pwm_5M_fullflow | Flow (Heun, K=4) | Flow ODE |

---

## Memory: 400GB per job | Time: 40 hours | Partition: gpu-l40s
