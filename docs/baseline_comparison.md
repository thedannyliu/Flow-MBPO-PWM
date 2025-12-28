# PWM Baseline Comparison Report

Comparison between our modified PWM (`PWM/`) and original imgeorgiev/PWM (`baselines/original_pwm/`).

**Date**: 2025-12-26

---

## Summary

| File | Status | Impact |
|------|--------|--------|
| `world_model.py` | ✅ IDENTICAL | No concerns |
| `critic.py` | ✅ IDENTICAL | No concerns |
| `mlp.py` | ✅ IDENTICAL | No concerns |
| `model_utils.py` | ✅ IDENTICAL | No concerns |
| `actor.py` | ⚠️ MODIFIED | Added FlowActor class, code style changes |
| `pwm.py` | ⚠️ MODIFIED (437 lines) | Flow-matching support, monitoring, evaluation changes |
| `dflex_ant.yaml` | ⚠️ MODIFIED | num_envs: 128 → 256 |
| Config files | ⚠️ DIFFERENT | See config comparison below |

---

## Critical Config Differences (Potential Performance Impact)

Comparing `baselines/original_pwm/scripts/cfg/alg/pwm_5M.yaml` vs `PWM/scripts/cfg/alg/pwm_5M_baseline_final.yaml`:

| Parameter | Original | Ours | Notes |
|-----------|----------|------|-------|
| `wm_batch_size` | 256 | 1024 | **4x larger** - may affect training dynamics |
| `wm_buffer_size` | 1,000,000 | 2,000,000 | 2x larger replay buffer |
| `task_dim` | 96 | 0 | We use 0 for single-task (correct) |
| `multitask` | True | False | We use False for single-task (correct) |
| `planning` | False | (removed) | Minor, defaults to False |

### Environment Config (`dflex_ant.yaml`)

| Parameter | Original | Ours | Notes |
|-----------|----------|------|-------|
| `num_envs` | 128 | 256 | **2x more parallel envs** - affects batch size |

---

## Algorithm Code Changes (`pwm.py`)

### New Parameters Added

```python
# Flow-matching specific parameters
use_flow_dynamics: bool = False,
flow_integrator: str = "heun",       # 'heun' or 'euler'
flow_substeps: int = 2,
flow_tau_sampling: str = "uniform",  # 'uniform' or 'midpoint'
wm_weight_decay: float = 0.0,        # New regularization option
```

### Key Behavioral Changes

1. **World Model Optimizer**: Added `weight_decay=self.wm_weight_decay` and supports Flow velocity network
2. **Actor/Eval Step**: Branching for flow vs MLP dynamics with `self.use_flow_dynamics`
3. **Enhanced Monitoring**: Added `TrainingMonitor`, `WandBLogger`, gradient stats
4. **Evaluation**: Changed to use true environment reward instead of predicted reward (was using wm predicted reward)

### Evaluation Logic Change (Potential Issue)

**Original code:**
```python
z, rew, trunc = self.wm.step(z, actions, task=None)
_, _, done, _ = self.env.step(actions)
```

**Our code:**
```python
# ... flow/mlp branching ...
# Get true environment reward instead of using world model's predicted reward
_, env_rew, done, _ = self.env.step(actions)  
```

> [!WARNING]
> We modified evaluation to use real environment reward. This is correctfix but means our metrics may differ from original PWM paper metrics if they used predicted rewards.

---

## Actor Changes (`actor.py`)

1. **FlowActor class added** (new, ~120 lines): ODE-based policy for Phase 2 experiments
2. **Code quality improvements**: Type hints, docstrings, style cleanup
3. **No behavioral changes to original ActorStochasticMLP/ActorDeterministicMLP**

---

## New Files in Our Codebase

1. `flow_world_model.py` - Flow-matching world model implementation
2. Multiple flow-specific config files (`pwm_5M_flow_*.yaml`, etc.)
3. Enhanced utilities: `monitoring.py`, `reproducibility.py`, `integrators.py`

---

## Recommendations

### For Fair Baseline Comparison

1. **Use original config values** when comparing baseline to original paper:
   - `wm_batch_size: 256` (not 1024)
   - `num_envs: 128` (not 256)
   - `wm_buffer_size: 1_000_000` (not 2M)

2. **Create a pure baseline config** that exactly matches `baselines/original_pwm/scripts/cfg/alg/pwm_5M.yaml`

3. **Verify evaluation logic** - our env-reward change is correct but means different metric collection

### Preserved Original PWM Configs

We have copies in `PWM/scripts/cfg/alg/original_pwm/` that match the original repo exactly.
