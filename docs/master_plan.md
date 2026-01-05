# Master Plan: Flow-MBPO MT30 Experiments

## Overview
Compare Flow World Model + Flow Policy against MLP Baseline on MT30 multitask benchmarks.
**Critical Distinction**: Experiments differ significantly in whether the World Model is **Pretrained** or **trained from scratch**.

---

## Experiment Phases Structure

### Phase 3: Policy Comparison (Frozen Pretrained WM)
- **Methodology**: "Policy Fine-tuning"
    - Load **Pretrained MLP World Model** (trained for weeks on MT30).
    - **Freeze WM** (`finetune_wm=False`).
    - Train Policy Only (15k epochs).
- **Goal**: Isolate policy performance (Flow vs MLP) on a perfect model.
- **Outcome**: ✅ **COMPLETED**. Baseline wins on hard tasks, tie on easy tasks.

### Phase 4: Full Flow Feasibility (Joint Training)
- **Methodology**: "Joint Training From Scratch"
    - Initialize pure Flow WM and Flow Policy.
    - Train **BOTH** simultaneously (`finetune_wm=True`) for 10k epochs.
- **Goal**: Can Flow learn dynamics + policy in short horizons?
- **Outcome**: ❌ **FAILED**. 10k epochs is insufficient (Original PWM uses millions of steps for pretraining).

### Phase 6: Epoch Sweep (Current Focus)
- **Methodology**: "Joint Training From Scratch" (Scaling Epochs)
    - Train WM + Policy simultaneously (`finetune_wm=True`).
    - Test horizons: **15k, 50k, 100k, 150k**.
- **Hypothesis**: Flow might learn faster than MLP, allowing "From Scratch" training to work within reasonable time (150k epochs ~16h).
- **Jobs**:
  | Epochs | Baseline | Flow | Status |
  |--------|----------|------|--------|
  | 15k | `4012533` | `4012534` | QUEUED |
  | 50k | `4012535` | `4012536` | QUEUED |
  | 100k | `4012537` | `4012538` | RUNNING |
  | 150k | `4012555` | `4012556` | RUNNING |

---

## Technical Details

### Training Duration Discrepancy
- **Original PWM Pretraining**: ~2 weeks on RTX 3090 (Millions of steps).
- **Our Phase 4 (Full Flow)**: ~1.5 hours on H200 (10k epochs).
- **Implication**: We are attempting to compress weeks of pretraining into hours of joint training.

### Configuration Alignment (Original PWM)
All experiments match `baselines/original_pwm`:
- `wm_batch_size: 256` (Fair comparison)
- `wm_iterations: 8`
- `wm_buffer_size: 1_000_000`
- `horizon: 16`

### Flow High Precision Settings
- `flow_substeps: 8` (WM)
- `actor_config.flow_substeps: 4` (Policy)
- `flow_integrator: heun`

---

## Resources
- **Pretrained WM**: `checkpoints/multitask/mt30_48M_4900000.pt`
- **Data**: `/home/hice1/eliu354/scratch/Data/tdmpc2/mt30/`
- **WandB**: `MT30-Detailed`
