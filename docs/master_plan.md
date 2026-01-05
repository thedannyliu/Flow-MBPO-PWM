# Master Plan: Flow-MBPO MT30 Experiments

## Overview
Compare Flow World Model + Flow Policy against MLP Baseline on MT30 multitask benchmarks.

---

## Experiment Phases

### Phase 3: Policy Comparison (✅ COMPLETED)
- **Goal**: Compare MLP vs Flow Policy with frozen, pretrained WM
- **Result**: Baseline wins on `walker-stand` (+14%), tie on simple tasks

### Phase 4: Full Flow From Scratch (✅ COMPLETED)
- **Goal**: Train Flow WM + Flow Policy from scratch (10k epochs)
- **Result**: Severely undertrained. Need more epochs.

### Phase 5: Flow Tuning (🟢 RUNNING)
- **Job**: `4012434`
- **Goal**: Optimize Flow parameters (substeps, integrator)

### Phase 6: Epoch Sweep (🟢 RUNNING)
- **Goal**: Determine optimal training duration
- **Jobs**:
  | Epochs | Baseline | Flow |
  |--------|----------|------|
  | 15k | `4012533` | `4012534` |
  | 50k | `4012535` | `4012536` |
  | 100k | `4012537` | `4012538` |
  | 150k | `4012555` | `4012556` |

---

## Configuration Alignment (Original PWM)
All experiments match `baselines/original_pwm`:
- `wm_batch_size: 256`
- `wm_iterations: 8`
- `wm_buffer_size: 1_000_000`
- `horizon: 16`

## Flow High Precision Settings
- `flow_substeps: 8` (WM)
- `actor_config.flow_substeps: 4` (Policy)
- `flow_integrator: heun`

---

## Resources
- **Pretrained WM**: `checkpoints/multitask/mt30_48M_4900000.pt`
- **Data**: `/home/hice1/eliu354/scratch/Data/tdmpc2/mt30/`
- **WandB**: `MT30-Detailed`
