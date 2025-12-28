# Experiment Log

Purpose: Registry for all training jobs with Slurm IDs, configs, seeds, wandb links, and status.

---

## Session: 2025-12-28 - Flow Policy Experiments

### Summary
- **Baseline jobs**: COMPLETED (3080227-3080229)
- **Flow WM jobs**: RUNNING with 256GB memory (3082648-3082656)
- **Flow Policy jobs**: SUBMITTED (3082664-3082669)

### WandB Dashboard
- **Project**: `flow-mbpo-single`
- **URL**: https://wandb.ai/danny010324/flow-mbpo-single

---

## Completed: Baseline (MLP WM + MLP Policy)
| Job ID | Seed | WandB Name | Status |
|--------|------|------------|--------|
| 3080227 | 42 | Anymal_Baseline_MLP_s42 | ✅ COMPLETED |
| 3080228 | 123 | Anymal_Baseline_MLP_s123 | ✅ COMPLETED |
| 3080229 | 456 | Anymal_Baseline_MLP_s456 | ✅ COMPLETED |

---

## Running: Flow WM + MLP Policy (256GB memory)
| Job ID | Config | Seed | WandB Name | Status |
|--------|--------|------|------------|--------|
| 3082648 | K=4 Heun | 42 | Anymal_FlowWM_K4Heun_s42 | RUNNING |
| 3082649 | K=4 Heun | 123 | Anymal_FlowWM_K4Heun_s123 | RUNNING |
| 3082650 | K=4 Heun | 456 | Anymal_FlowWM_K4Heun_s456 | RUNNING |
| 3082651 | K=2 Heun | 42 | Anymal_FlowWM_K2Heun_s42 | RUNNING |
| 3082652 | K=2 Heun | 123 | Anymal_FlowWM_K2Heun_s123 | RUNNING |
| 3082653 | K=2 Heun | 456 | Anymal_FlowWM_K2Heun_s456 | RUNNING |
| 3082654 | K=8 Euler | 42 | Anymal_FlowWM_K8Euler_s42 | PENDING |
| 3082655 | K=8 Euler | 123 | Anymal_FlowWM_K8Euler_s123 | PENDING |
| 3082656 | K=8 Euler | 456 | Anymal_FlowWM_K8Euler_s456 | PENDING |

---

## Submitted: Flow Policy (MLP WM + Flow ODE Policy)
| Job ID | Seed | WandB Name | Status |
|--------|------|------------|--------|
| 3082664 | 42 | Anymal_FlowPolicy_MLPWM_s42 | SUBMITTED |
| 3082665 | 123 | Anymal_FlowPolicy_MLPWM_s123 | SUBMITTED |
| 3082666 | 456 | Anymal_FlowPolicy_MLPWM_s456 | SUBMITTED |

---

## Submitted: Full Flow (Flow WM + Flow ODE Policy)
| Job ID | Seed | WandB Name | Status |
|--------|------|------------|--------|
| 3082667 | 42 | Anymal_FullFlow_FlowWM_FlowPol_s42 | SUBMITTED |
| 3082668 | 123 | Anymal_FullFlow_FlowWM_FlowPol_s123 | SUBMITTED |
| 3082669 | 456 | Anymal_FullFlow_FlowWM_FlowPol_s456 | SUBMITTED |

---

## New Code Components

### ActorFlowODE (src/pwm/models/flow_actor.py)
- ODE-based policy using learned velocity field
- Integrates from noise to action using Heun's method
- 2 flow substeps by default (configurable)
- Compatible with existing actor interface

### Configurations
- `pwm_5M_flowpolicy.yaml`: MLP WM + Flow Policy
- `pwm_5M_fullflow.yaml`: Flow WM + Flow Policy

---

## Memory Configuration
- L40s nodes: 515GB max, 8 GPUs
- Using 256GB per job (128GB caused OOM for Flow WM)

## Expected Completion
- ~8-12 hours per job for 15k epochs
