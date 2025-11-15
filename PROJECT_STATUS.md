# Flow-MBPO-PWM: Project Status & Overview

**Last Updated**: November 14, 2025  
**Branch**: `dev/flow-dynamics`  
**Status**: 🔴 Training Issues - Debugging Phase

---

## 🎯 Project Goal

Implementing and comparing **Flow Matching** dynamics models against standard **Deterministic** world models in the PWM (Policy learning with World Models) framework for continuous control tasks.

**Key Question**: Can flow-matching based world models improve sample efficiency and policy performance compared to deterministic models?

---

## 📊 Current Status

### Recent Training Results (Nov 10-14, 2025)

| Model | Size | GPU | Status | Final Reward | Expected | Notes |
|-------|------|-----|--------|--------------|----------|-------|
| **Baseline** | 5M | H200 | ❌ Failed | R~25 | R~1200+ | Collapsed completely |
| **Flow** | 5M | H200 | ❌ Failed | R~164 | R~900+ | Unstable, low performance |
| **Flow** | 5M | L40s | ⏸️ Preempted | - | R~900+ | Interrupted after 4h50m |
| **Baseline** | 5M | L40s | ❓ Unknown | - | R~1200+ | Needs analysis |

**Key Finding**: All November 10th H200 training runs failed catastrophically despite completing full iteration counts.

### Best Available Checkpoints

**From Previous Successful Runs (Nov 8-9)**:
- ✅ **5M Baseline** (Nov 8): `outputs/2025-11-08/23-48-46/` - R~1200
- ⚠️ **5M Flow** (Nov 9): `outputs/2025-11-09/06-18-53/` - R~900 (25% below baseline)

---

## 🔴 Critical Issues Identified

### 1. Configuration Errors
- **task_dim**: Set to 96 for single-task training (should be 0)
- **wm_batch_size**: 2048 too aggressive, causing instability
- **lr_schedule**: Wrong schedule (`constant` vs `linear`)

### 2. Training Failures
- **H200 Baseline**: Total collapse, losses zeroed out, gradient vanishing
- **H200 Flow**: Never learned effectively, extremely unstable rewards
- **Root Cause**: Accumulated configuration errors from multiple debugging attempts

### 3. Evaluation Issues
- Evaluation script has environment naming bugs
- Output buffering prevents real-time result logging
- CSV comparison files not being generated properly

---

## 🛠️ Technical Stack

### Environment
- **Task**: DeepMind Control Suite - Ant locomotion
- **Framework**: dflex (differentiable physics)
- **Observations**: 37-dim proprioceptive state
- **Actions**: 8-dim continuous
- **Episode Length**: 1000 steps

### Model Architecture
- **Actor**: 3-layer MLP (400→200→100 units, ELU activation)
- **World Model**: 
  - Baseline: Deterministic dynamics (5M or 48M params)
  - Flow: Flow-matching dynamics with substeps=4
- **Training**: Model-based RL with learned world model

### Compute Resources
- **GPU**: NVIDIA H200 (primary), L40s (backup)
- **SLURM**: PACE cluster (Georgia Tech)
- **Account**: gts-agarg35

---

## 📁 Repository Structure

```
Flow-MBPO-PWM/
├── PWM/                          # Main codebase
│   ├── src/pwm/                  # Core PWM implementation
│   │   ├── algorithms/           # PWM algorithm
│   │   ├── models/               # Actor, Critic, World Model
│   │   │   └── world_model.py    # Flow vs Deterministic models
│   │   └── utils/                # Training utilities
│   ├── scripts/                  # Training & evaluation scripts
│   │   ├── train_dflex.py        # Main training script
│   │   ├── evaluate_policy.py    # Policy evaluation
│   │   ├── cfg/alg/              # Algorithm configs
│   │   │   ├── pwm_5M.yaml           # 5M Baseline config
│   │   │   ├── pwm_5M_flow.yaml      # 5M Flow config
│   │   │   ├── pwm_5M_improved.yaml  # Attempted fixes
│   │   │   └── pwm_5M_flow_improved.yaml
│   │   └── submit_*.sh           # SLURM submission scripts
│   ├── outputs/                  # Training checkpoints (gitignored)
│   └── logs/                     # Training logs (gitignored)
├── docs/                         # Project documentation (gitignored)
└── README.md                     # This file
```

---

## 🔄 Workflow

### Training Pipeline
1. **Configuration**: Set up YAML config with hyperparameters
2. **Submission**: Submit SLURM job via `submit_*.sh` scripts
3. **Monitoring**: Check logs in `PWM/logs/` for progress
4. **Checkpointing**: Best models saved to `PWM/outputs/`
5. **Evaluation**: Run `evaluate_policy.py` on checkpoints

### Key Scripts
```bash
# Submit training job
cd PWM
sbatch scripts/submit_retrain_5M_baseline.sh

# Monitor progress
tail -f logs/retrain_5M_baseline_*.out

# Evaluate checkpoint
python scripts/evaluate_policy.py \
  --baseline outputs/.../best_policy.pt \
  --env dflex_ant \
  --num-episodes 100
```

---

## 📝 Configuration Files

### Baseline Model (`pwm_5M.yaml`)
```yaml
task_dim: 0                    # Single-task
wm_hidden_dims: [512, 512]     # 5M parameters
wm_activation: relu
wm_batch_size: 512
lr_schedule: linear
num_iters: 15000
```

### Flow Model (`pwm_5M_flow.yaml`)
```yaml
task_dim: 0
use_flow_model: true
flow_substeps: 4               # Flow matching integration steps
wm_hidden_dims: [512, 512]
wm_activation: relu
wm_batch_size: 512
lr_schedule: linear
num_iters: 20000               # More iterations for flow
```

---

## 🎯 Next Steps

### Immediate (High Priority)
1. ✅ **Fix Configuration**: Revert to proven Nov 8-9 configs
2. ✅ **Analyze L40s Baseline**: Check Job 2199152 results
3. ✅ **Fix Evaluation Script**: Resolve environment naming and output issues
4. 🔄 **Re-train Models**: Use validated configurations only

### Short-term
1. **Debug Flow Model**: Investigate 25% performance gap vs baseline
2. **Hyperparameter Tuning**: Systematic search once training is stable
3. **Ablation Studies**: Test flow_substeps (2, 4, 8, 16)

### Long-term
1. **Scale to 48M Parameters**: Once 5M models are stable
2. **Multi-task Training**: Test on MT30/MT80 benchmarks
3. **Paper Experiments**: Reproduce results for publication

---

## 📊 Performance Targets

| Model | Current | Target | Status |
|-------|---------|--------|--------|
| 5M Baseline | R~1200 | R~1200+ | ✅ Achieved (Nov 8) |
| 5M Flow | R~900 | R~1200+ | ⚠️ 25% gap |
| 48M Baseline | - | R~1500+ | 📅 Future work |
| 48M Flow | - | R~1500+ | 📅 Future work |

---

## 🐛 Known Issues

1. **Configuration Management**: Manual YAML editing error-prone
2. **Evaluation Pipeline**: Script needs refactoring for robustness
3. **Flow Model Underperformance**: Root cause unclear
4. **Training Instability**: Recent runs showing collapse patterns
5. **Checkpoint Management**: No automatic cleanup of failed runs

---

## 📚 References

### Key Papers
- **PWM**: Policy learning with World Models
- **Flow Matching**: Lipman et al. (2023)
- **TD-MPC2**: Hansen et al. (2024)
- **DreamerV3**: Hafner et al. (2023)

### Related Repos
- PWM: [Original implementation]
- TD-MPC2: `PWM/external/tdmpc2/`
- dflex: Differentiable physics engine

---

## 👥 Team & Contact

**Project**: Flow-MBPO-PWM  
**Institution**: Georgia Tech  
**Cluster**: PACE (Partnership for Advanced Computing Environment)  
**Account**: gts-agarg35

---

## 🔖 Version History

- **v0.3** (Nov 14, 2025): Training failures diagnosed, configuration fixes identified
- **v0.2** (Nov 8-9, 2025): Successful 5M baseline and flow checkpoints
- **v0.1** (Oct-Nov 2025): Initial implementation and experiments

---

**Status Legend**:
- ✅ Completed/Working
- ⚠️ Partial/Issues
- ❌ Failed
- 🔄 In Progress
- 📅 Planned
- ⏸️ Paused
- ❓ Unknown
