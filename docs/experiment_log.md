# Experiment Log

Purpose: Registry for all training jobs with Slurm IDs, configs, seeds, wandb links, and status.

---

## Session: 2025-12-28 - Complete Experiment Matrix

### WandB Dashboard
- **Project**: `flow-mbpo-single`
- **URL**: https://wandb.ai/danny010324/flow-mbpo-single

---

## Overall Status Summary
| Category | Variant | Seeds | Status |
|----------|---------|-------|--------|
| Baseline | MLP WM + MLP Policy | 3 | ✅ COMPLETED |
| Core | Flow WM K=4 Heun | 3 | ✅ COMPLETED |
| Core | Flow WM K=2 Heun | 3 | ✅ COMPLETED |
| Core | Flow WM K=8 Euler | 3 | 🔄 RUNNING |
| Core | MLP WM + Flow Policy | 3 | ✅ COMPLETED |
| Core | Full Flow | 1+2 | 🔄 s42 RUNNING, s123/s456 PENDING |
| **Ablation** | Flow WM H=8 | 3 | 🔄 RUNNING |
| **Ablation** | Flow WM LowLR | 3 | ⏳ PENDING |
| **Ablation** | Flow Policy H=8 | 3 | ⏳ PENDING |

---

## Completed Experiments ✅

### Baseline (MLP WM + MLP Policy) - Dec 27
| Job ID | Seed | Runtime |
|--------|------|---------|
| 3080227 | 42 | ~2h |
| 3080228 | 123 | ~2h |
| 3080229 | 456 | ~2h |

### Flow WM K=4 Heun - Dec 28
| Job ID | Seed | Runtime |
|--------|------|---------|
| 3082681 | 42 | 4:55:05 |
| 3082682 | 123 | 4:54:37 |
| 3082683 | 456 | 4:53:58 |

### Flow WM K=2 Heun - Dec 28
| Job ID | Seed | Runtime |
|--------|------|---------|
| 3082684 | 42 | 3:36:07 |
| 3082685 | 123 | 3:38:12 |
| 3082686 | 456 | 4:27:55 |

### MLP WM + Flow Policy - Dec 28
| Job ID | Seed | Runtime |
|--------|------|---------|
| 3084840 | 42 | 2:22:20 |
| 3084841 | 123 | 2:22:48 |
| 3084842 | 456 | 2:22:34 |

---

## Currently Running 🔄

### Flow WM K=8 Euler (~4h elapsed)
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3084837 | 42 | Anymal_FlowWM_K8Euler_s42 |
| 3084838 | 123 | Anymal_FlowWM_K8Euler_s123 |
| 3084839 | 456 | Anymal_FlowWM_K8Euler_s456 |

### Full Flow (Flow WM + Flow Policy)
| Job ID | Seed | Status |
|--------|------|--------|
| 3084843 | 42 | RUNNING (~4h) |
| 3087706 | 123 | PENDING |
| 3087707 | 456 | PENDING |

### Ablation: Flow WM H=8 (just started)
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3087697 | 42 | Anymal_FlowWM_K4_H8_s42 |
| 3087698 | 123 | Anymal_FlowWM_K4_H8_s123 |
| 3087699 | 456 | Anymal_FlowWM_K4_H8_s456 |

---

## Pending ⏳

### Ablation: Flow WM LowLR (actor_lr=3e-4)
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3087700 | 42 | Anymal_FlowWM_K4_LR3e4_s42 |
| 3087701 | 123 | Anymal_FlowWM_K4_LR3e4_s123 |
| 3087702 | 456 | Anymal_FlowWM_K4_LR3e4_s456 |

### Ablation: Flow Policy H=8
| Job ID | Seed | WandB Name |
|--------|------|------------|
| 3087703 | 42 | Anymal_FlowPolicy_H8_s42 |
| 3087704 | 123 | Anymal_FlowPolicy_H8_s123 |
| 3087705 | 456 | Anymal_FlowPolicy_H8_s456 |

---

## Configuration Files
| Config | Description |
|--------|-------------|
| pwm_5M_baseline_final | Baseline MLP WM + MLP Policy |
| pwm_5M_flow_v1_substeps2 | Flow WM K=2 Heun |
| pwm_5M_flow_v2_substeps4 | Flow WM K=4 Heun |
| pwm_5M_flow_v3_substeps8_euler | Flow WM K=8 Euler |
| pwm_5M_flowpolicy | MLP WM + Flow Policy |
| pwm_5M_fullflow | Flow WM + Flow Policy |
| **pwm_5M_flow_H8** | Flow WM K=4 with H=8 |
| **pwm_5M_flow_lowLR** | Flow WM K=4 with LR=3e-4 |
| **pwm_5M_flowpolicy_H8** | Flow Policy with H=8 |

---

## Resource Allocation
- **Memory**: 400GB per job
- **Time**: 40 hours
- **Partition**: gpu-l40s
- **Account**: gts-agarg35-ideas_l40s
