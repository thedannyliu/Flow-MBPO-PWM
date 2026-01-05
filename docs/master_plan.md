# Master Plan: Flow-MBPO MT30 Experiments (2×2 Factorial)

## Overview
Primary objective is a fair **2×2 factorial** on MT30:
- **World Model**: `MLP` vs `Flow`
- **Policy**: `MLP` vs `Flow ODE`

**Critical Distinction**: Results are not comparable if you mix **pretrained+frozen WM** with **from-scratch joint training** without stating it explicitly.

---

## Experiment Phases Structure

### Phase 8: 2×2 Factorial (Pretrained WM; Primary)
**Methodology**: **Pretrain WM → Train Policy**

**Step 1: WM Pretraining** (Current)
| Job ID | WM Type | Status | Output |
|--------|---------|--------|--------|
| `4012664` | Flow WM | ⏳ QUEUED | `flowwm_mt30_best.pt` |
| `4012665` | MLP WM | ⏳ QUEUED | `mlpwm_mt30_best.pt` |

**Step 2: Policy Training** (After WM pretraining completes)
Use pretrained checkpoints for 2×2 factorial policy training:

| WM | Policy | Config | Checkpoint |
|---|---|---|---|
| MLP | MLP | `pwm_48M_mt_baseline` | `mlpwm_mt30_best.pt` |
| MLP | Flow | `pwm_48M_mt_flowpolicy` | `mlpwm_mt30_best.pt` |
| Flow | MLP | `pwm_48M_mt_flowwm` | `flowwm_mt30_best.pt` |
| Flow | Flow | `pwm_48M_mt_fullflow` | `flowwm_mt30_best.pt` |

**Settings**:
- **Frozen WM**: `finetune_wm=False` (primary comparison)
- **Fine-tuned WM**: `finetune_wm=True` (ablation)

---

### Phase 7: Policy Comparison (Fine-tuned Pretrained WM)
- **Status**: ⏳ QUEUED (`4012601`)
- **Methodology**: Load original PWM checkpoint → Enable Fine-tuning → Train Policy

### Phase 6: Epoch Sweep (From Scratch)
- **Status**: 🟢 RUNNING (100k/150k on H200)
- **Methodology**: Joint Training From Scratch

### Phase 3: Policy Comparison (Frozen Pretrained WM)
- **Status**: ✅ COMPLETED
- **Result**: Baseline wins on walker-stand, tie on others

---

## Technical Details

### Recommended Execution Order (Phase 8)
1. **Pretrain Flow WM** on MT30 offline dataset (Job `4012664`)
2. **Pretrain MLP WM** for control (Job `4012665`)
3. Run the 4 quadrants with **frozen WM** (`finetune_wm=False`)
4. Repeat the 4 quadrants with **fine-tuning** (`finetune_wm=True`) as ablation

### Training Duration Discrepancy
- **Original PWM Pretraining**: ~2 weeks on RTX 3090 (Millions of steps)
- **Our Phase 8 Pretraining**: ~16h on H100 (200k iterations)
- **Note**: Our budget is smaller but still substantial for fair comparison

### Configuration Alignment (Original PWM)
All experiments match `baselines/original_pwm`:
- `wm_batch_size: 256` (Fair comparison)
- `wm_iterations: 8`
- `wm_buffer_size: 1_000_000`
- `horizon: 16`

---

## Resources
- **Original PWM Checkpoint**: `checkpoints/multitask/mt30_48M_4900000.pt`
- **Flow WM Pretraining**: `scripts/pretrain_multitask_wm.py` + `scripts/cfg/pretrain_mt30_wm.yaml`
- **Data**: `/home/hice1/eliu354/scratch/Data/tdmpc2/mt30/`
- **WandB**: `WM-Pretrain` (pretraining), `MT30-Detailed` (policy training)
