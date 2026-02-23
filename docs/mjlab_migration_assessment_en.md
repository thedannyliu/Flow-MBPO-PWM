# Detailed Assessment and Experiment Plan for Migrating Flow-MBPO-PWM to mjlab

## 1) Conclusion First

- Migrating to `mjlab` is feasible, and is most likely to speed up iteration for single-task online training (`scripts/train_dflex.py`).
- The project already has partial migration groundwork (`frank/pwm_env_adapter.py`, `scripts/test_pwm_playground.py`), but **it is not integrated into the official Hydra training path yet**, so it cannot directly support main experiments.
- The largest technical risk is that PWM currently depends on `info["obs_before_reset"]` to avoid replay contamination from auto-reset; `mjlab` API returns Gymnasium-style `(obs, reward, terminated, truncated, info)` by default, so additional adapter logic is required for alignment.

## 2) Current Codebase and Environment Coupling

Your current main training loop (`scripts/train_dflex.py` + `src/flow_mbpo_pwm/algorithms/pwm.py`) is tightly coupled to environment API requirements:

- `env = instantiate(cfg.env.config, ...)` (Hydra instantiates env directly)
- `env` must provide:
  - `num_envs`, `num_obs`, `num_actions`, `episode_length`
  - `observation_space.shape`, `action_space.shape`
  - `reset(grads=True/False)`
  - `step(actions) -> (obs, reward, done, info)`
  - `info` must include `termination`, `truncation`, `obs_before_reset`, `primal`

Critical dependency is in `src/flow_mbpo_pwm/algorithms/pwm.py` `compute_actor_loss()`:
after `done`, replay writes `obs_before_reset` to avoid mistaking reset observations for terminal observations.

## 3) Alignment Between Official mjlab Capabilities and Your Requirements

Based on official docs (Installation / Migration / API):

- `mjlab` is PyTorch-centric and supports GPU acceleration + vectorized environments.
- Installation/runtime recommendation is the `uv` workflow (`uv sync`).
- Migration guide emphasizes an interface close to Gymnasium / Isaac Lab.
- `ManagerBasedRlEnv.step()` returns `(obs_dict, reward, terminated, truncated, info)`, and `reset()` returns `(obs_dict, info)`.
- Distributed training recommendation is one process per GPU with `torch.distributed` DDP.

This is "connectable but non-zero cost" with your current PWM training logic:

- Directly alignable: `terminated/truncated`, vectorized `num_envs`, PyTorch tensor pipeline.
- Adapter needed: obs-dict flattening, `reset(grads=True)` behavior, and `obs_before_reset` compatibility layer.

## 4) Required Changes (minimal viable migration target)

## 4.1 Add a mjlab-specific adapter (required)

Recommended new file:

- `src/flow_mbpo_pwm/envs/mjlab_pwm_adapter.py`

Responsibilities:

- Wrap `mjlab` env into PWM interface:
  - expose `num_envs / num_obs / num_actions / episode_length`
  - support `reset(grads=True)` returning cached obs (without resetting)
  - map Gymnasium outputs in `step()` to PWM-expected format
- Flatten `obs_dict` into one state vector deterministically (state-only first to reduce initial complexity)
- Unify `done = terminated | truncated`

## 4.2 Add Hydra env config (required)

Recommended new files:

- `scripts/cfg/env/mjlab_<task>.yaml` (start with one locomotion task)

Should include:

- `_target_`: points to `create_mjlab_pwm_env(...)` factory
- `task_id` / `num_envs` / `device` / `episode_length` / `action_repeat`, etc.

## 4.3 Training entry compatibility (required)

You can keep `scripts/train_dflex.py` unchanged for now, but recommended:

- add comments or an alias script (e.g., `scripts/train_online.py`) to avoid misleading naming (it is no longer dFlex-only)

## 4.4 Evaluation script compatibility (recommended)

Current `scripts/eval/eval_pwm.py` and `scripts/evaluate_policy.py` hardcode `dflex_*`.
Recommended minimum:

- `scripts/eval/eval_pwm_mjlab.py` (or refactor env creation in existing script to support `mjlab_*`)

Otherwise post-training evaluation will break.

## 4.5 Job scripts and experiment management (recommended)

Most current `scripts/*submit*.sh` use `env=dflex_*`.
Recommended:

- create `scripts/mjlab/` with new smoke/submit scripts first, without replacing old dflex scripts.

## 5) Main Risks and Mitigations

## 5.1 Highest risk: `obs_before_reset` semantic mismatch

Risk:

- If adapter cannot provide the correct terminal-pre-reset observation, replay may mix in post-reset observations, hurting WM training quality and fairness.

Mitigation:

1. Check whether `mjlab info` already contains terminal observation fields (if yes, map directly).
2. If not, implement a "correctness-first adapter":
   - explicitly capture pre-reset observation in the env-step flow (may require wrapper/subclass).
3. Add smoke-test assertions:
   - when `done`, `obs_before_reset` must be distinguishable from post-reset `obs` (not always equal).

## 5.2 Dependency conflicts

Risk:

- Your `environment.yaml` is oriented toward an older stack (Torch 2.3 / CUDA 11.8 + dFlex);
- `mjlab` ecosystem (newer JAX/MJX/warp/mujoco) may conflict.

Mitigation:

- Create an **independent environment** (e.g., `flow-mbpo-mjlab`), do not overwrite current `pwm` env.
- Keep old dflex path as rollback.

## 5.3 Speed gain may be non-linear

Risk:

- Even with a faster simulator, end-to-end wall-clock may still be dominated by WM updates and Python logic.

Mitigation:

- Profile first after migration:
  - `env.step` time share
  - `world model training` time share
  - end-to-end FPS (already present in your logs)
- If env share is < 30%, simulator acceleration alone gives limited gain; optimize training code in parallel.

## 6) Recommended Migration Path (three phases)

## Phase A: Make it runnable (1-2 days)

Goal: run single-task smoke end-to-end; no fairness conclusion yet.

- Create `mjlab_pwm_adapter.py`
- Add one `mjlab` env config
- Run baseline smoke with `max_epochs=100~300`, `num_envs=32~64`
- Validate:
  - no NaN
  - replay has data (`buffer.num_eps > 0`)
  - eval completes

## Phase B: Correctness alignment (2-4 days)

Goal: fix `obs_before_reset` semantics and evaluation chain.

- Complete correct done/reset boundary handling
- Add `mjlab` evaluation path
- Run 2-3 seed short experiments to verify stable learning curves

## Phase C: Formal experiments (1-2 weeks)

Goal: fair baseline-vs-flow comparisons in `mjlab`.

- Fix same task, same seed set, same budget
- Start with the minimum 4-group matrix:
  1. MLP-WM + MLP-Policy
  2. Flow-WM + MLP-Policy
  3. MLP-WM + Flow-Policy
  4. Flow-WM + Flow-Policy
- Report:
  - final reward / success
  - convergence speed (wall-clock and env steps)
  - FPS and cost (GPU hours)

## 7) How to Run Experiments on mjlab (recommended workflow)

## 7.1 Environment setup

- Use official `uv` recommendation or an isolated conda+pip environment.
- Run `mjlab` official quick-start/play examples first to verify GPU and rendering backend.

## 7.2 First in-repo smoke command (recommended shape)

```bash
python scripts/train_dflex.py \
  env=mjlab_<task> \
  alg=pwm_5M_baseline_final \
  general.seed=0 \
  alg.max_epochs=300 \
  env.config.num_envs=64 \
  general.run_wandb=true
```

Then switch `alg` to a flow version under the same conditions and compare:

- FPS (steps per second)
- reward curves at equal epochs
- whether done/reset boundary anomalies appear

## 7.3 Formal controlled-comparison rules

To keep results interpretable:

- Fix everything except the target factor (seed, epochs, horizon, batch, lr, eval_freq)
- Use the same task set and initialization protocol for baseline and flow
- If you keep both dflex and mjlab experiments:
  - do not compare absolute reward directly across simulators
  - only compare method deltas **within the same simulator**

## 8) Execution Recommendations After This Assessment

Recommended direct actions:

1. Start with **Phase A**, target: first usable learning curve within 48 hours.
2. Enter **Phase B** only after confirming `obs_before_reset`; do not scale resources before this.
3. After B is complete, start seed expansion and ablations.

---

## References (official)

- mjlab GitHub: https://github.com/mujocolab/mjlab
- mjlab docs: https://mujocolab.github.io/mjlab/
- Installation Guide: https://mujocolab.github.io/mjlab/source/installation_guide.html
- Migration Guide: https://mujocolab.github.io/mjlab/source/migration_guide.html
- Environment API (`ManagerBasedRlEnv`): https://mujocolab.github.io/mjlab/source/api/envs.html#mjlab.envs.manager_based_rl_env.ManagerBasedRlEnv
- Distributed Training Guide: https://mujocolab.github.io/mjlab/source/distributed_training_guide.html
- PyPI (versions and release timeline): https://pypi.org/project/mjlab/
