# Experiment Log

Purpose: Registry for all training jobs with Slurm IDs, configs, seeds, wandb links, and status.

---

## Session: 2025-12-29 - Complete Experiment Matrix

### WandB Dashboard
- **Project**: `flow-mbpo-single`
- **URL**: https://wandb.ai/danny010324/flow-mbpo-single

---

## Status Summary (Dec 29, 2:15 AM)

### COMPLETED ✅ (21 jobs)
| Variant | Seeds | Job IDs |
|---------|-------|---------|
| Baseline (MLP WM + MLP Policy) | 3 | 3080227-3080229 |
| Flow WM K=4 Heun | 3 | 3082681-3082683 |
| Flow WM K=2 Heun | 3 | 3082684-3082686 |
| Flow WM K=8 Euler | 3 | 3084837-3084839 |
| MLP WM + Flow Policy | 3 | 3084840-3084842 |
| Full Flow s42 | 1 | 3084843 |
| Flow WM H=8 | 3 | 3087697-3087699 |
| Flow Policy H=8 | 3 | 3087703-3087705 |

### RUNNING 🔄 (14 jobs)
| Variant | Seeds | Job IDs | Elapsed |
|---------|-------|---------|---------|
| Flow WM LowLR | 3 | 3087700-3087702 | ~4h |
| Full Flow s123, s456 | 2 | 3087706-3087707 | ~2h |
| **Flow WM StrongReg** | 3 | 3091990-3091992 | just started |
| **Full Flow H=8** | 3 | 3091993-3091996 | just started |
| **Flow WM HighLR** | 3 | 3091997-3091999 | just started |

---

## Hyperparameter Ablation Matrix

### Flow WM Ablations
| Variant | Config | LR | H | Reg | Status |
|---------|--------|-----|---|-----|--------|
| K=4 Heun (base) | pwm_5M_flow_v2_substeps4 | 5e-4 | 16 | base | ✅ |
| K=2 Heun | pwm_5M_flow_v1_substeps2 | 5e-4 | 16 | base | ✅ |
| K=8 Euler | pwm_5M_flow_v3_substeps8_euler | 5e-4 | 16 | base | ✅ |
| H=8 | pwm_5M_flow_H8 | 5e-4 | 8 | base | ✅ |
| LowLR | pwm_5M_flow_lowLR | 3e-4 | 16 | base | 🔄 |
| StrongReg | pwm_5M_flow_strongReg | 5e-4 | 16 | 3e-4 | 🔄 |
| HighLR | (CLI override) | 7e-4 | 16 | base | 🔄 |

### Flow Policy Ablations
| Variant | WM | Policy | H | Status |
|---------|-----|--------|---|--------|
| Flow Policy (base) | MLP | Flow ODE | 16 | ✅ |
| Flow Policy H=8 | MLP | Flow ODE | 8 | ✅ |
| Full Flow (base) | Flow | Flow ODE | 16 | 🔄 |
| Full Flow H=8 | Flow | Flow ODE | 8 | 🔄 |

---

## Configuration Files
| Config | Description |
|--------|-------------|
| pwm_5M_baseline_final | Baseline |
| pwm_5M_flow_v1_substeps2 | K=2 Heun |
| pwm_5M_flow_v2_substeps4 | K=4 Heun |
| pwm_5M_flow_v3_substeps8_euler | K=8 Euler |
| pwm_5M_flowpolicy | Flow Policy |
| pwm_5M_fullflow | Full Flow |
| pwm_5M_flow_H8 | H=8 |
| pwm_5M_flow_lowLR | LR=3e-4 |
| pwm_5M_flowpolicy_H8 | Flow Policy H=8 |
| **pwm_5M_flow_strongReg** | Strong reg (wd=3e-4) |
| **pwm_5M_fullflow_H8** | Full Flow H=8 |

---

## Resource Allocation
- **Memory**: 400GB per job
- **Time**: 40 hours
- **Partition**: gpu-l40s
