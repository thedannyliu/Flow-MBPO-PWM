# Experiment Log

Purpose: Registry for all training and evaluation jobs with Slurm IDs, configs, seeds, wandb links, status, and log paths.

---

## 2025-12-27 – Anymal Single-Task Training (ACTIVE)

### Status Summary
- **Baseline jobs**: 3080227, 3080228, 3080229 – RUNNING ✅
- **Flow K=4 jobs**: 3080239, 3076505, 3080240 – RUNNING ✅
- **Flow K=2 jobs**: 3080241, 3080242, 3080243 – RUNNING ✅
- **Flow K=8 jobs**: 3080244, 3080245, 3080246 – RUNNING ✅

### WandB Dashboard
- **Project**: `flow-mbpo-single`
- **URL**: https://wandb.ai/danny010324/flow-mbpo-single

### Issues Fixed This Session
1. **PWM.py duplicate WandBLogger**: Removed internal WandBLogger that overwrote run names with 'env_flow'/'env_baseline'
2. **visualization._smooth() array mismatch**: Fixed to return same-length array using np.convolve
3. **train_dflex.py naming**: Updated to use wandb.name from CLI when provided
4. **Hydra argument quoting**: Removed spaces from wandb.notes to avoid parsing errors

### Current Job Registry

| Job ID | Config | Seed | WandB Name | Status |
|--------|--------|------|------------|--------|
| 3080227 | Baseline | 42 | Anymal_Baseline_MLP_s42 | RUNNING |
| 3080228 | Baseline | 123 | Anymal_Baseline_MLP_s123 | RUNNING |
| 3080229 | Baseline | 456 | Anymal_Baseline_MLP_s456 | RUNNING |
| 3080239 | Flow K=4 | 42 | Anymal_FlowWM_K4Heun_s42 | RUNNING |
| 3076505 | Flow K=4 | 123 | (prior run) | RUNNING |
| 3080240 | Flow K=4 | 456 | Anymal_FlowWM_K4Heun_s456 | RUNNING |
| 3080241 | Flow K=2 | 42 | Anymal_FlowWM_K2Heun_s42 | RUNNING |
| 3080242 | Flow K=2 | 123 | Anymal_FlowWM_K2Heun_s123 | RUNNING |
| 3080243 | Flow K=2 | 456 | Anymal_FlowWM_K2Heun_s456 | RUNNING |
| 3080244 | Flow K=8 | 42 | Anymal_FlowWM_K8Euler_s42 | RUNNING |
| 3080245 | Flow K=8 | 123 | Anymal_FlowWM_K8Euler_s123 | RUNNING |
| 3080246 | Flow K=8 | 456 | Anymal_FlowWM_K8Euler_s456 | RUNNING |

### Validation Test Result
- **Job ID**: 3080225
- **Type**: 50-epoch smoke test for WandB naming fix
- **Result**: ✅ PASSED
- **WandB Run Name**: `TEST_Anymal_Baseline_s42`
- **Final Reward**: R=28.57

### Training Configuration
| Parameter | Value |
|-----------|-------|
| max_epochs | 15,000 |
| wm_batch_size | 256 |
| wm_buffer_size | 1,000,000 |
| horizon | 16 |
| num_envs | 64 |

### Expected Completion
- ~8-12 hours per job
- Check status: `squeue -u $USER`
- Logs: `logs/slurm/anymal/`
