# Unified Branch Merge Documentation

> **Branch**: `dev/unified-pwm`  
> **Created**: January 31, 2026  
> **Base**: `origin/master`

---

## Overview

This document summarizes the merge of three parallel development branches into a single unified codebase (`dev/unified-pwm`):

| Branch | Purpose | Source of Truth For |
|--------|---------|---------------------|
| `dev/flow-dynamics` | Single-task flow-matching dynamics experiments | Single-task training logic |
| `dev/multitask` | Multi-task PWM training and evaluation | Batching & task routing |
| `frank/mjwarp` | Extended mjwarp environments | Environment extensions |

---

## Branch Summaries

### 1. dev/flow-dynamics

**Focus**: Single-task flow-matching dynamics experiments on dflex environments (Ant, Anymal, Humanoid).

**Key Additions**:
- `scripts/eval/` - Evaluation pipeline (`eval_pwm.py`, `aggregate_eval_results.py`, report generators)
- `scripts/cfg/alg/pwm_5M_flow_v3_aligned.yaml` - Aligned flow WM config
- `scripts/cfg/alg/pwm_5M_flowpolicy_aligned.yaml` - Aligned flow policy config
- `scripts/humanoid/` - Humanoid-specific submission scripts
- `scripts/submit_aligned_*.sh` - Aligned experiment submission scripts
- `scripts/submit_v*_recovery.py` - Job recovery scripts
- `eval_results/final_eval_results.csv` - Consolidated evaluation results
- `docs/experiment_log.md` - Single-task experiment registry
- `docs/progress_log.md` - Single-task progress tracking
- `docs/master_plan.md` - Research planning

**Experiments Tracked**:
- Ant Baseline, FlowWM_K8 (10 seeds each)
- Anymal Baseline, FlowPolicy (10 seeds each)
- Humanoid Baseline, FlowPolicy (10 seeds each)

---

### 2. dev/multitask

**Focus**: Multi-task PWM training with MT30/MT80 task sets.

**Key Additions**:
- `scripts/cfg/alg/pwm_48M_mt_*.yaml` - Multi-task algorithm configs:
  - `pwm_48M_mt_baseline.yaml` - MLP world model baseline
  - `pwm_48M_mt_flowwm.yaml` - Flow-matching world model
  - `pwm_48M_mt_flowpolicy.yaml` - Flow-matching policy
  - `pwm_48M_mt_fullflow.yaml` - Full flow (WM + policy)
- `scripts/cfg/pretrain_mt30_wm.yaml` - WM pretraining config
- `scripts/pretrain_multitask_wm.py` - WM pretraining script
- `scripts/mt30/` - Multi-task submission scripts:
  - `submit_baseline*.sh` - Baseline experiments
  - `submit_flow*.sh` - Flow experiments
  - `submit_phase*.sh` - Phased experiments
  - `submit_pretrain_*.sh` - WM pretraining
  - `collect_results.py` - Results aggregation
  - `cleanup_weights.py` - Checkpoint management
- `results/` - Multi-task experiment CSVs
- Updated `scripts/train_multitask.py`:
  - Resume training support (`resume_from` config key)
  - Random start delay for cluster stability
  - Fixed OmegaConf handling
  - Removed IPython debug hook

**Changes to Core Files**:
- `src/flow_mbpo_pwm/algorithms/pwm.py`:
  - Added training progress tracking (`iter_count`, `step_count`, `best_reward`)
  - Enhanced save/load with `resume_training` flag
  - Tensor-safe metric logging (`.item()` for tensors)
  - Improved `load_wm()` with format detection
- `src/flow_mbpo_pwm/utils/buffer.py`:
  - Minor stability fixes

---

### 3. frank/mjwarp

**Focus**: Extended MuJoCo Playground (mjwarp) environment support for online training.

**Key Additions**:
- `frank/` - PWM environment adapters:
  - `pwm_env_adapter.py` - Wraps RSLRLBraxWrapper for PWM compatibility
  - `dmcontrol_pwm_adapter.py` - DMControl-to-PWM adapter
  - `batched_tdmpc2.py` - Batched TD-MPC2 implementation
  - `batched_trainer.py` - Batched training utilities
  - `episode_tracker.py` - Episode tracking for parallel envs
- `mujoco_playground/` - Git submodule for MuJoCo Playground
- `scripts/test_pwm_playground.py` - Test script for mjwarp integration
- `pyproject.toml` - Package configuration for uv
- `requirements.txt` - Additional dependencies
- `.gitmodules` - Submodule configuration

---

## Conflict Resolution

### Files with Conflicts

| File | Resolution Strategy |
|------|---------------------|
| `.gitignore` | Merged both exclusion lists |
| `docs/experiment_log.md` | Concatenated (Part 1: Single-task, Part 2: Multi-task) |
| `docs/progress_log.md` | Concatenated (Multi-task updates first, then single-task history) |
| `src/flow_mbpo_pwm/algorithms/pwm.py` | **multitask wins** - Training progress tracking essential for resume |

### Resolution Details

#### pwm.py (Critical)
The `dev/multitask` version was preserved because it includes:
1. Training progress tracking (`iter_count`, `step_count`, `best_reward`)
2. Enhanced checkpoint save/load with `resume_training` flag
3. Tensor-safe metric logging (prevents CUDA tensor serialization issues)
4. Improved `load_wm()` that auto-detects checkpoint format

#### Documentation
Both branches maintained separate experiment logs. The merged documentation:
- **experiment_log.md**: Part 1 contains single-task aligned experiments, Part 2 contains multi-task phase experiments
- **progress_log.md**: Multi-task updates at top (newest), single-task history below

---

## API Changes

### New Config Keys

| Config File | New Key | Type | Description |
|-------------|---------|------|-------------|
| `config_mt30.yaml` | `general.resume_from` | `str|null` | Path to checkpoint for resuming training |
| `pwm_48M_mt_*.yaml` | `use_flow_dynamics` | `bool` | Enable flow-matching world model |
| `pwm_48M_mt_*.yaml` | `flow_substeps` | `int` | Number of flow integration substeps |

### Modified Functions

#### `PWM.save()`
```python
# Now includes training progress
{
    "iter_count": getattr(self, 'iter_count', 0),
    "step_count": getattr(self, 'step_count', 0),
    "best_reward": getattr(self, 'best_reward', -float('inf')),
    "best_policy_loss": getattr(self, 'best_policy_loss', float('inf')),
    "mean_horizon": getattr(self, 'mean_horizon', 0.0),
}
```

#### `PWM.load(path, buffer=False, resume_training=False)`
```python
# New resume_training parameter
# When True: restores iter_count, step_count, best_reward
# When False: loads weights only (for fine-tuning/eval)
```

#### `PWM.load_wm(path)`
```python
# Auto-detects checkpoint format:
# - Native format: looks for "world_model" key
# - Original PWM format: looks for "model" key with _pi/_Qs filtering
```

---

## New Configs Required

### For Multi-task Training

```yaml
# config_mt30.yaml
general:
  resume_from: null  # or path/to/checkpoint.pt
  epochs: 10_000
  finetune_wm: False

# With flow world model
alg: pwm_48M_mt_flowwm  # or pwm_48M_mt_baseline, pwm_48M_mt_flowpolicy
```

### For Single-task Training

```yaml
# config_dflex_ant.yaml (existing)
alg: pwm_5M_flow_v3_aligned  # or pwm_5M_flowpolicy_aligned
```

### For mjwarp Environments

```python
# scripts/test_pwm_playground.py
from frank.pwm_env_adapter import PWMEnvAdapter
env = PWMEnvAdapter(brax_wrapper)
```

---

## Verification Results

### Syntax Validation
All critical files pass Python syntax check:
- ✅ `src/flow_mbpo_pwm/algorithms/pwm.py`
- ✅ `scripts/train_multitask.py`
- ✅ `src/flow_mbpo_pwm/utils/buffer.py`
- ✅ `scripts/eval/eval_pwm.py`
- ✅ `frank/batched_tdmpc2.py`
- ✅ `frank/pwm_env_adapter.py`
- ✅ `frank/dmcontrol_pwm_adapter.py`

### Available Entry Points

| Script | Purpose | Branch Origin |
|--------|---------|---------------|
| `scripts/train_multitask.py` | Multi-task training | multitask |
| `scripts/pretrain_multitask_wm.py` | WM pretraining | multitask |
| `scripts/eval/eval_pwm.py` | Checkpoint evaluation | flow-dynamics |
| `scripts/test_pwm_playground.py` | mjwarp integration test | mjwarp |

---

## Git Commit History

```
7183a54 merge(mjwarp): Integrate extended mjwarp environments and adapters
32f2c87 merge(multitask): Integrate multi-task PWM training and evaluation
3e9df30 merge(flow-dynamics): Integrate single-task flow-matching dynamics experiments
50ec194 (origin/master) remove api key
```

---

## Known Limitations

1. **No automated integration tests**: Runtime verification requires cluster GPU access
2. **Documentation overlap**: Both branches had independent experiment tracking; merged by concatenation
3. **mjwarp submodule**: Requires `git submodule update --init` after clone

---

## Usage Examples

### Single-task Training (flow-dynamics)
```bash
cd scripts
python train.py alg=pwm_5M_flow_v3_aligned task=dflex_ant seed=42
```

### Multi-task Training (multitask)
```bash
cd scripts
python train_multitask.py alg=pwm_48M_mt_flowwm general.epochs=5000
```

### Resume Multi-task Training
```bash
cd scripts
python train_multitask.py alg=pwm_48M_mt_flowwm \
    general.resume_from=/path/to/checkpoint.pt
```

### mjwarp Environment Test
```bash
cd scripts
python test_pwm_playground.py --num_envs=64 --max_epochs=100
```

### Evaluation (single-task checkpoints)
```bash
cd scripts/eval
python eval_pwm.py /path/to/checkpoint.pt dflex_ant --n_runs=10
```

---

## Future Work

1. Add automated CI tests for each branch's functionality
2. Consolidate experiment logging format across single-task and multi-task
3. Create unified training script that auto-detects single vs multi-task
4. Add mjwarp configs to Hydra config system
