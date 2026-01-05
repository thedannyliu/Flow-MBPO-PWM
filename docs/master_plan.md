# Master Plan: Flow-MBPO MT30 Experiments

## Overview
Compare Flow World Model + Flow Policy against MLP Baseline on MT30 multitask benchmarks.
**Critical Distinction**: Experiments differ significantly in whether the World Model is **Pretrained** or **trained from scratch**.

---

## Experiment Phases Structure

### Phase 3: Policy Comparison (Frozen Pretrained WM)
- **Methodology**: "Policy Fine-tuning" (`finetune_wm=False`)
- **Goal**: Isolate policy performance (Flow vs MLP) on a perfect model.
- **Outcome**: ✅ **COMPLETED**. Baseline wins on hard tasks.

### Phase 7: Policy Comparison (Fine-tuned Pretrained WM)
- **Methodology**: "Policy + WM Tuning" (`finetune_wm=True`)
    - Load **Pretrained MLP World Model**.
    - **Unfreeze and Fine-tune WM** alongside Policy.
- **Goal**: Does allowing the WM to adapt help Flow Policy performance?
- **Status**: 🟢 **RUNNING** (Job `4012601`).

### Phase 6: Epoch Sweep (Joint Training From Scratch)
- **Methodology**: "Joint Training From Scratch" (`finetune_wm=True`)
    - Train WM + Policy simultaneously from random initialization.
    - Test horizons: **15k, 50k, 100k, 150k**.
- **Goal**: Can Flow learn dynamics fast enough without pretraining?
- **Status**: 🟢 **RUNNING** (100k/150k jobs).

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
