# Single-Task Online RL Experiment Spec (PACE-ICE, Single GPU)

Version: `v1.0`  
Date: `2026-02-27`  
Owner: `Flow-MBPO-PWM`  
Scope: `single-task only`, `online RL only`, `train-from-scratch only`, `single GPU per run`

## 1. Objective

Build a reproducible, high-throughput experiment system to compare PWM-style baselines and Flow variants across mixed environments:

- dFlex tasks (PWM-compatible, GPU accelerated): Hopper / Ant / Anymal / Humanoid / SNU Humanoid
- mjlab tasks (MuJoCo Warp backend, GPU accelerated):  
  `Mjlab-Velocity-Flat-Unitree-Go2`, `Mjlab-Velocity-Flat-Unitree-G1`,  
  `Mjlab-Tracking-Flat-Unitree-G1` (motion imitation), `Mjlab-Leap-Left-HandCube-Rotate` (in-hand manipulation)

Primary comparison target is PWM family only:

- MLP WM + MLP Policy
- Flow WM + MLP Policy
- MLP WM + Flow Policy
- Flow WM + Flow Policy

## 2. Platforms and Resource Constraints

- Cluster: Georgia Tech PACE-ICE
- GPU mode: **single GPU per job only**
- Supported GPU pools: `H100`, `H200`, `L40s`
- Login nodes have no GPU; all train/eval/rollout jobs must run through Slurm (`sbatch`)

## 3. Task Panel (Frozen)

| Suite | Task key | Hydra env | Complexity | Episode length |
|---|---|---|---|---:|
| dFlex | hopper | `dflex_hopper` | Low | 1000 |
| dFlex | ant | `dflex_ant` | Medium | 1000 |
| dFlex | anymal | `dflex_anymal` | Medium | 1000 |
| dFlex | humanoid | `dflex_humanoid` | Medium-High | 1000 |
| dFlex | snu_humanoid | `dflex_snu_humanoid` | High | 1000 |
| mjlab | velocity_flat_unitree_go2 | `mjlab_velocity_flat_unitree_go2` | Medium | 1000 |
| mjlab | velocity_flat_unitree_g1 | `mjlab_velocity_flat_unitree_g1` | Medium | 1000 |
| mjlab | tracking_flat_unitree_g1 | `mjlab_tracking_flat_unitree_g1` | Medium-High | 1000 |
| mjlab | leap_left_handcube_rotate | `mjlab_leap_left_handcube_rotate` | High | 500 |

Notes:
- mjlab task IDs can change between releases. Smoke stage is mandatory for ID/API validation before pilot/confirm stages.

## 4. Method Matrix (2x2 Factorial)

| Method key | Algorithm config | WM | Policy |
|---|---|---|---|
| `mlpwm_mlppolicy` | `pwm_5M_baseline_final` | MLP | MLP |
| `flowwm_mlppolicy` | `pwm_5M_flow_v2_substeps4` | Flow | MLP |
| `mlpwm_flowpolicy` | `pwm_5M_flowpolicy` | MLP | Flow |
| `flowwm_flowpolicy` | `pwm_5M_fullflow` | Flow | Flow |

## 5. Staged Budget Plan

| Stage | Goal | Seeds | Hparam profiles | Epoch budget | Eval games | Rollout episodes |
|---|---|---:|---|---:|---:|---:|
| Smoke | API + training + eval sanity | 1 (`seed=0`) | `default` | 200 | 8 | 2 |
| Pilot | trend + hparam filtering | 3 (`0,1,2`) | `default`, `lr_x05`, `lr_x20` | 3000 | 20 | 3 |
| Confirm | reportable statistics | 10 (`0..9`) | `default` (or selected pilot winner) | 15000 | 40 | 5 |

Learning-rate profiles:
- `default`: config default
- `lr_x05`: actor/critic/model LR × 0.5
- `lr_x20`: actor/critic/model LR × 2.0

## 6. WandB Logging Spec (Train + Eval)

## 6.1 Required Run Metadata

Each train run must include:
- `stage`, `suite`, `task`, `method`, `seed`, `hparam_profile`, `run_key`
- Slurm metadata: job id, array job id, array task id, node, partition
- Git metadata (logged externally in experiment report): commit hash + dirty state

## 6.2 Required Training Metrics

Core RL metrics:
- `rewards`, `policy_loss`, `policy_discounted_loss`, `episode_lengths`
- `actor_loss`, `value_loss`, `wm_loss`, `reward_loss`, `dynamics_loss`
- `actor_grad_norm`, `critic_grad_norm`, `wm_grad_norm`, `actor_std`

Throughput and timing:
- `fps`, `profile/epoch_wall_seconds`
- `profile/*_seconds`, `profile/*_pct`
- `profile/training_time_total_seconds`, `profile/training_time_avg_epoch_seconds`

Buffer and GPU:
- `buffer/num_episodes`, `buffer/fill_ratio_estimate`
- `gpu/memory_allocated_bytes`, `gpu/memory_reserved_bytes`
- `gpu/memory_allocated_ratio`, `gpu/memory_reserved_ratio`

Adapter diagnostics (mjlab path):
- `env/terminal_obs_from_info`, `env/terminal_obs_from_fallback`
- `env/terminal_obs_equal_next_obs_ratio`
- `env/done_events`, `env/done_terminated`, `env/done_truncated`

Final eval (post-train):
- `eval/final_episode_loss`, `eval/final_discounted_loss`
- `eval/final_episode_length`, `eval/final_mean_reward_proxy`

## 6.3 Required Evaluation Artifacts

For each run:
- `eval_summary.json`
- `episode_metrics.csv`
- `rollout_steps.csv` (step-level traces)
- `rollout_summary.csv` (episode-level rollout summaries)

Evaluation metrics:
- return: mean/std/median/min/max/p25/p75/IQM
- discounted return mean
- episode length mean/std
- success rate (if env exposes success signal)

## 7. Profiling Spec

Default profiling comes from internal section timers (already instrumented):
- actor compute, forward sim, env step, backward sim, critic prep, critic training, WM training

Analysis rules:
- If `profile/env_step_pct > 50%`, prioritize simulator-side and env batching optimizations.
- If `profile/world_model_training_pct > 40%`, prioritize WM batch/iterations/model-size tuning.
- If `profile/critic_training_pct` dominates, reduce critic iterations/batches in pilot stage.

## 8. Evaluation and Rollout Protocol

1. Use `best_policy.pt` if available, otherwise fallback to `final_policy.pt`.
2. Evaluate deterministically by default.
3. Log at least `N=20` episodes (pilot) and `N=40` episodes (confirm).
4. Collect rollout traces for human inspection:
   - per-step reward, action norm, observation norm
   - per-episode return and length
5. Keep rendering/video optional; prioritize numeric rollout comparability across all tasks.

## 9. Slurm Execution Spec (PACE-ICE)

- Submission style: Slurm array (`--array`) with concurrency cap (`%K`)
- One array index = one experiment row from manifest
- One GPU per row (`--gres=gpu:*:1`)
- Supported partitions:
  - `gpu-h100`
  - `gpu-h200`
  - `gpu-l40s`

Recommended defaults:
- smoke: `8h`, `128G`, `16 CPU`
- pilot: `24h`, `128G`, `16 CPU`
- confirm: `48h`, `128G~256G`, `16 CPU`

## 10. Reproducibility Rules

- Fixed seed lists per stage; same seed list across all methods
- Hydra resolved configs stored in run directory
- Save both `best_policy.pt` and `final_policy.pt`
- Keep all scripts/spec/changelogs in English
- Use English commit messages only

## 11. Ready-for-Smoke Acceptance Checklist

A stage is considered "ready" only if all conditions are true:

1. Manifest generation works (`smoke_v1.csv` created).
2. Slurm array submission works for selected GPU type.
3. At least one dFlex and one mjlab task finish train+eval.
4. WandB receives train + eval metrics and metadata tags.
5. Eval artifacts are generated under run `eval/` directory.
6. No terminal-observation contamination alarms in mjlab adapter diagnostics.

## 12. Implemented Pipeline in This Repo

Manifest and launcher:
- `scripts/experiments/single_task_online/build_manifest.py`
- `scripts/experiments/single_task_online/run_manifest_job.py`
- `scripts/experiments/single_task_online/submit_manifest_array.sh`
- `scripts/experiments/single_task_online/build_default_manifests.sh`
- `scripts/experiments/single_task_online/submit_profile_single.sh`

Task configs:
- `scripts/cfg/env/mjlab_velocity_flat_unitree_go2.yaml`
- `scripts/cfg/env/mjlab_velocity_flat_unitree_g1.yaml`
- `scripts/cfg/env/mjlab_tracking_flat_unitree_g1.yaml`
- `scripts/cfg/env/mjlab_leap_left_handcube_rotate.yaml`

Evaluation:
- `scripts/eval/eval_online_single_task.py`

## 13. Next Step

Run smoke stage first, then freeze pilot shortlist based on:
- return IQM
- wall-clock to threshold
- profile bottleneck share
- crash/NaN rate
