# Experiment Log

Purpose: Registry for all training and evaluation jobs with Slurm IDs, configs, seeds, phases, metrics, wandb links, status, and log paths.

---

## 2025-12-27 – Anymal Single-Task Training (Phase 1)

### Experiment Overview
- **Environment**: Anymal (medium difficulty locomotion)
- **WandB Project**: `flow-mbpo-single`
- **Submission Time**: 2025-12-27 05:42 EST
- **GPU**: L40S (1 per job)
- **Memory**: 384GB
- **Time Limit**: 40 hours
- **Account**: gts-agarg35-ideas_l40s

### Jobs Submitted

| Job ID | Config | Seed | WandB Name | Status |
|--------|--------|------|------------|--------|
| 3076501 | `pwm_5M_baseline_final` | 42 | anymal_MLPWM_MLPpol_s42 | RUNNING |
| 3076502 | `pwm_5M_baseline_final` | 123 | anymal_MLPWM_MLPpol_s123 | RUNNING |
| 3076503 | `pwm_5M_baseline_final` | 456 | anymal_MLPWM_MLPpol_s456 | RUNNING |
| 3076504 | `pwm_5M_flow_v2_substeps4` (K=4) | 42 | anymal_FlowWM_K4_s42 | RUNNING |
| 3076505 | `pwm_5M_flow_v2_substeps4` (K=4) | 123 | anymal_FlowWM_K4_s123 | RUNNING |
| 3076506 | `pwm_5M_flow_v2_substeps4` (K=4) | 456 | anymal_FlowWM_K4_s456 | RUNNING |
| 3076507 | `pwm_5M_flow_v1_substeps2` (K=2) | 42 | anymal_FlowWM_K2_s42 | PENDING |
| 3076508 | `pwm_5M_flow_v1_substeps2` (K=2) | 123 | anymal_FlowWM_K2_s123 | PENDING |
| 3076509 | `pwm_5M_flow_v1_substeps2` (K=2) | 456 | anymal_FlowWM_K2_s456 | PENDING |
| 3076510 | `pwm_5M_flow_v3_substeps8_euler` (K=8) | 42 | anymal_FlowWM_K8Euler_s42 | PENDING |
| 3076511 | `pwm_5M_flow_v3_substeps8_euler` (K=8) | 123 | anymal_FlowWM_K8Euler_s123 | PENDING |
| 3076512 | `pwm_5M_flow_v3_substeps8_euler` (K=8) | 456 | anymal_FlowWM_K8Euler_s456 | PENDING |

### Smoke Test Results (Pre-submission Validation)
| Job ID | Type | Epochs | Result |
|--------|------|--------|--------|
| 3076495 | Baseline | 100 | ✅ PASSED (R~17, ~1920 FPS) |
| 3076500 | Flow WM K=4 | 100 | ✅ PASSED (FlowWorldModel loaded) |

### Key Hyperparameters (Aligned for Fair Comparison)
| Parameter | Value | Notes |
|-----------|-------|-------|
| `wm_batch_size` | 256 | Original PWM value |
| `wm_buffer_size` | 1,000,000 | Original PWM value |
| `num_envs` | 64 | Anymal default |
| `max_epochs` | 15,000 | Original PWM value |
| `horizon` | 16 | Default PWM value |
| `latent_dim` | 512 | - |

### Flow-Specific Parameters
| Variant | Integrator | Substeps K | Config File |
|---------|------------|------------|-------------|
| v1 | Heun | 2 | `pwm_5M_flow_v1_substeps2.yaml` |
| v2 | Heun | 4 | `pwm_5M_flow_v2_substeps4.yaml` |
| v3 | Euler | 8 | `pwm_5M_flow_v3_substeps8_euler.yaml` |

### Expected Completion
- ~8-12 hours per job for 15k epochs
- Check status: `squeue -u $USER`
- Logs: `logs/slurm/anymal/`
