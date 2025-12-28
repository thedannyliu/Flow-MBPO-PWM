# Experiment Log

Purpose: Registry for all training and evaluation jobs with Slurm IDs, configs, seeds, phases, metrics, wandb links, status, and log paths.

---

## 2025-12-27 – Session 2 – Fixed WandB Naming and Resubmitted Jobs

### Issues Fixed
1. **PWM.py duplicate WandBLogger**: Removed internal WandBLogger creation that was overriding run names with generic 'env_flow'/'env_baseline'
2. **visualization._smooth() array mismatch**: Fixed to return same-length array using np.convolve with 'same' mode
3. **train_dflex.py naming**: Updated `create_wandb_run()` to use `wandb.name` from CLI override when provided

### Validation Test
- **Job ID**: 3080225
- **Type**: Smoke test (50 epochs)
- **Result**: ✅ PASSED
- **WandB Run Name**: `TEST_Anymal_Baseline_s42` (verified naming fix works)
- **Final Reward**: R=28.57

### Jobs Resubmitted (2025-12-27 20:43 EST)

| Job ID | Config | Seed | WandB Name | Status |
|--------|--------|------|------------|--------|
| 3080227 | `pwm_5M_baseline_final` | 42 | Anymal_Baseline_MLP_s42 | SUBMITTED |
| 3080228 | `pwm_5M_baseline_final` | 123 | Anymal_Baseline_MLP_s123 | SUBMITTED |
| 3080229 | `pwm_5M_baseline_final` | 456 | Anymal_Baseline_MLP_s456 | SUBMITTED |
| 3080230 | `pwm_5M_flow_v2_substeps4` (K=4) | 42 | Anymal_FlowWM_K4Heun_s42 | SUBMITTED |
| 3076505 | `pwm_5M_flow_v2_substeps4` (K=4) | 123 | (previous, running) | RUNNING |
| 3080231 | `pwm_5M_flow_v2_substeps4` (K=4) | 456 | Anymal_FlowWM_K4Heun_s456 | SUBMITTED |
| 3080232 | `pwm_5M_flow_v1_substeps2` (K=2) | 42 | Anymal_FlowWM_K2Heun_s42 | SUBMITTED |
| 3080233 | `pwm_5M_flow_v1_substeps2` (K=2) | 123 | Anymal_FlowWM_K2Heun_s123 | SUBMITTED |
| 3080234 | `pwm_5M_flow_v1_substeps2` (K=2) | 456 | Anymal_FlowWM_K2Heun_s456 | SUBMITTED |
| 3080235 | `pwm_5M_flow_v3_substeps8_euler` (K=8) | 42 | Anymal_FlowWM_K8Euler_s42 | SUBMITTED |
| 3080236 | `pwm_5M_flow_v3_substeps8_euler` (K=8) | 123 | Anymal_FlowWM_K8Euler_s123 | SUBMITTED |
| 3080237 | `pwm_5M_flow_v3_substeps8_euler` (K=8) | 456 | Anymal_FlowWM_K8Euler_s456 | SUBMITTED |

### WandB Dashboard
- **Project**: `flow-mbpo-single`
- **URL**: https://wandb.ai/danny010324/flow-mbpo-single

### Hyperparameters (All Aligned for Fair Comparison)
| Parameter | Baseline | Flow WM |
|-----------|----------|---------|
| `wm_batch_size` | 256 | 256 |
| `wm_buffer_size` | 1,000,000 | 1,000,000 |
| `max_epochs` | 15,000 | 15,000 |
| `horizon` | 16 | 16 |
| `num_envs` | 64 | 64 |

### Previous Session (2025-12-27 Morning) – Failed Jobs Analysis
Previous batch (jobs 3076501-3076512) had issues:
- WandB names were generic ('env_flow', 'env_baseline') due to duplicate logger
- Several jobs failed with "CUDA device busy" error due to GPU contention
- Completed runs logged to wrong project ('pwm-flow-matching' instead of 'flow-mbpo-single')

---

## Expected Completion
- ~8-12 hours per job for 15k epochs
- Check status: `squeue -u $USER`
- Logs: `logs/slurm/anymal/`
