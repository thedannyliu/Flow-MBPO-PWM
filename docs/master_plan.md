# Master Plan: Flow-MBPO MT30 Experiments (2×2 Factorial)

## Overview
Primary objective is a fair **2×2 factorial** on MT30:
- **World Model**: `MLP` vs `Flow`
- **Policy**: `MLP` vs `Flow ODE`

**Critical Distinction**: results are not comparable if you mix **pretrained+frozen WM** with **from-scratch joint training** without stating it explicitly.

---

## Experiment Phases Structure

### Phase 8: 2×2 Factorial (Pretrained WM; Primary)
- **Methodology**: **Pretrain WM → Train Policy**, with two settings:
  - **Frozen WM**: `finetune_wm=False` (primary comparison)
  - **Fine-tuned WM**: `finetune_wm=True` (ablation)
- **Why**: isolates where Flow helps (WM vs policy) without confounding with “who had a better model”.
- **Quadrants (configs)**:
  - MLP WM + MLP policy: `alg=pwm_48M_mt_baseline`
  - MLP WM + Flow policy: `alg=pwm_48M_mt_flowpolicy`
  - Flow WM + MLP policy: `alg=pwm_48M_mt_flowwm`
  - Flow WM + Flow policy: `alg=pwm_48M_mt_fullflow`
- **Checkpoint rule**: WM checkpoints must match the instantiated WM architecture:
  - MLP WM configs require an **MLP WM** checkpoint
  - Flow WM configs require a **Flow WM** checkpoint

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

### Recommended Execution Order (Phase 8)
1. **Pretrain Flow WM** on MT30 offline dataset using `scripts/pretrain_multitask_wm.py` (produces a native `world_model` checkpoint).
2. **(Optional) Pretrain MLP WM** with the same script to remove reliance on the original PWM checkpoint.
3. Run the 4 quadrants with **frozen WM** (`finetune_wm=False`) across tasks/seeds.
4. Repeat the 4 quadrants with **fine-tuning** (`finetune_wm=True`) as ablation.

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
- **Flow WM Pretraining**: `scripts/pretrain_multitask_wm.py` + `scripts/cfg/pretrain_mt30_wm.yaml` (output under Hydra run dir, `logs/<out_name>_<iters>.pt`)
- **Data**: `/home/hice1/eliu354/scratch/Data/tdmpc2/mt30/`
- **WandB**: `MT30-Detailed`
