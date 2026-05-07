# Original PWM Code Audit and MJLab-QS Adapter Plan

Date: 2026-05-07
Branch: dev/mjlab

## Purpose

The goal of this pass is to answer two questions:

1. What does the upstream `baselines/PWM` pipeline actually do?
2. Can we run the original PWM pipeline on the same MJLab-QS offline dataset to compare against our state-space PWM/Flow-MBPO runner?

## What the upstream PWM code supports

The upstream repository has two official entrypoints:

- `baselines/PWM/scripts/train_dflex.py`
  - Single-task DFlex environments.
  - Supports loading a pretrained DFlex world model with `general.checkpoint`.
  - Supports pretraining a DFlex world model from DFlex-format offline data with `general.pretrain` and `general.pretrain_steps`.

- `baselines/PWM/scripts/train_multitask.py`
  - TD-MPC2 MT30/MT80 task setup.
  - Loads TD-MPC2-style offline `.pt` trajectories into `pwm.utils.buffer.Buffer`.
  - Loads a pretrained TD-MPC2/PWM world model checkpoint.
  - Runs single-task policy extraction by repeatedly sampling H-step sequences from the replay buffer and calling `PWM.update(...)`.

There is no upstream entrypoint that directly consumes our MJLab-QS windows:

```text
phys_obs, command, policy_action, reward, done, quality_bin, split_id
```

Therefore, a byte-identical upstream `train_dflex.py` or `train_multitask.py` run on the MJLab-QS dataset is not plug-and-play.

## Original PWM algorithm details from `baselines/PWM/src/pwm/algorithms/pwm.py`

The original `PWM` class uses:

- World model: `M_phi = (encoder, dynamics, reward)`.
- Actor: stochastic MLP, then `tanh` action squashing.
- Critic: ensemble of 3 MLP critics, using min value for actor targets.
- Actor objective: first-order gradients through imagined rollouts in the learned WM.
- Critic target: TD(lambda), default `gamma=0.99`, `lambda=0.95`.
- Critic updates: default 8 iterations and 4 minibatches in the paper configs.
- Return RMS: enabled in paper configs.
- DFlex scalar reward path can use reward RMS.
- Multitask TD-MPC2 path uses discrete reward bins and no scalar reward RMS.

Paper-style config values checked in `baselines/PWM/scripts/cfg/alg/pwm_5M.yaml`:

```text
actor hidden: [400, 200, 100]
critic hidden: [400, 200]
latent_dim: 512
world model hidden: [512, 512]
encoder hidden: [256]
actor_lr: 5e-4
critic_lr: 5e-4
model_lr: 3e-4
gamma: 0.99
lambda: 0.95
horizon: 16
num_critics: 3
critic_iterations: 8
critic_batches: 4
actor_grad_norm: 1.0
critic_grad_norm: 100.0
ret_rms: True
max_epochs: 15000
```

## Adapter implemented

New files:

- `scripts/experiments/mjlab_qs/run_original_pwm_adapter.py`
- `scripts/experiments/mjlab_qs/run_original_pwm_adapter_row.py`

The adapter keeps the original PWM implementation intact:

```text
from pwm.algorithms.pwm import PWM
```

It uses the original:

- `PWM.compute_wm_loss`
- `PWM.update`
- original stochastic MLP actor
- original critic ensemble
- original SimNorm world model
- original TD(lambda) actor/critic loop

Only the boundary is adapted:

```text
MJLab-QS window -> original PWM obs/action/reward tensors
```

Canonical packing for this run:

```text
obs_t = concat(normalized_phys_obs_t, normalized_command_t)
reward_t = normalized_reward_t
rew_rms = false
ret_rms = true
```

This choice keeps the comparison aligned with our existing MJLab-QS state-space WM runs, where both physical state and reward are normalized before training. It is not byte-identical to upstream DFlex scalar reward normalization, but it is the most controlled same-data comparison.

## Why this is not fully identical to upstream PWM

The upstream scripts assume DFlex or TD-MPC2 data/env interfaces. Our task is MJLab command-conditioned locomotion. The differences are:

- MJLab command is packed into observation for the original PWM model, because upstream PWM does not expose separate command conditioning for dynamics vs reward.
- The adapter uses fixed MJLab-QS windows instead of TD-MPC2 trajectory files.
- The adapter evaluates in MJLab using our MJLab eval wrapper.
- The adapter uses normalized state/reward by default for direct same-data comparison.

So the correct label is:

```text
original PWM algorithm adapter on MJLab-QS data
```

not:

```text
byte-identical upstream PWM train_multitask.py run
```

## Submitted jobs

Smoke, no W&B, no real-env eval:

```text
job 5272000: original_pwm_adapter_smoke_20260507, L40S, pretrain=2, policy=2
```

Formal 3-seed run, W&B project `flow-mbpo-mjlab-original-pwm-adapter`:

```text
job 5272001: seed 0, H100, pretrain=50k, policy=15k
job 5272003: seed 1, H200, pretrain=50k, policy=15k
job 5272002: seed 2, L40S, pretrain=50k, policy=15k
```

Output root:

```text
scripts/outputs/mjlab_qs/original_pwm_adapter/original_pwm_adapter_g1_20260507/
```

Slurm logs:

```text
logs/slurm/mjlab_qs/original_pwm_adapter/
```

## Methods worth trying next after this audit

### 1. Original PWM adapter with paper-like raw reward + reward RMS

Run the same adapter with:

```text
reward_mode = raw
rew_rms = true
```

Reason: DFlex PWM uses scalar reward RMS. Our first adapter run uses normalized reward to match current MJLab-QS state-space runs. This ablation tests whether reward scaling is hurting policy extraction.

### 2. Remove action L2 from current state-space runner

Our state-space runner adds `action_l2=1e-4` inside imagined reward. Original PWM does not add this explicit penalty in the same location. Run:

```text
action_l2 = 0
```

Reason: If current returns are close to random/weak baseline, an extra action penalty may suppress gait generation.

### 3. Actor warm-start / behavior cloning before imagined PWM extraction

Pretrain the actor on expert or expert-noisy windows:

```text
min_theta E[||pi_theta(z_t, c_t) - a_t||^2]
```

then run PWM actor-critic extraction.

Reason: Original PWM starts from strong pretrained WMs and tasks where scratch extraction can work. MJLab G1 may need a policy prior to avoid poor local optima in imagined rollout optimization.

### 4. Longer policy extraction budget

We already submitted a 50k policy-iteration ablation for the state-space runner. Apply the same idea to original PWM adapter if the 15k run is still improving.

Reason: Original `max_epochs=15000` may be insufficient for MJLab G1, especially with a newly pretrained WM and command-conditioned locomotion.

### 5. More critic training or critic stabilization

Try:

```text
critic_iterations = 16
critic_batches = 8
```

Reason: PWM policy extraction quality depends on TD(lambda) critic targets from imagined rollouts. If value loss remains unstable, the actor may optimize a bad value landscape.

### 6. Cumulative online replay instead of latest-window online finetune

Our current online finetune uses newly collected windows per round. Original online PWM maintains a replay buffer from real interactions. Modify online finetune to keep cumulative replay:

```text
D_online <- D_offline union D_round1 union ... union D_roundK
```

Reason: latest-only finetune can overfit a narrow policy distribution and destabilize the WM.

### 7. Termination-aware WM and imagined rollout

Original PWM handles termination when an environment is attached. Our offline state-space runner does not yet have a termination head. Add termination prediction or done-aware imagined rollout masking.

Reason: G1 locomotion has short/fall episodes; ignoring termination during imagined policy extraction can make the policy exploit invalid model states.

## Current interpretation rule

If the original PWM adapter significantly outperforms our state-space PWM runner, the issue is likely in our reimplementation details: reward scaling, actor/critic loop, action penalty, or critic target handling.

If the original PWM adapter performs similarly poorly, then the problem is more likely the dataset, MJLab task difficulty, WM quality, or the difficulty of offline-only policy extraction for this G1 setup.

## Smoke bugfix and resubmission

Initial submitted jobs `5272000` to `5272003` failed at the adapter boundary:

```text
phys_obs has H+1 endpoints, command has H transition values.
Original PWM expects H+1 observation endpoints.
```

Fix applied:

```text
command_obs = concat(command[0:H], command[H-1:H])
obs = concat(phys_obs[0:H+1], command_obs[0:H+1])
```

This keeps commands aligned to the final endpoint by carrying the last transition command forward.

Resubmitted jobs:

```text
job 5272013: smoke, L40S, pretrain=2, policy=2, no W&B, no real eval
job 5272012: formal seed 0, H100, pretrain=50k, policy=15k
job 5272014: formal seed 1, H200, pretrain=50k, policy=15k
job 5272015: formal seed 2, L40S, pretrain=50k, policy=15k
```

## Smoke result after bugfix

Resubmitted smoke job `5272013` completed successfully in 11 seconds.

Observed smoke metrics:

```text
pretrain iter 1: val/wm_loss=0.14249, val/dynamics_loss=0.02285, val/reward_loss=0.11964
pretrain iter 2: val/wm_loss=0.08952, val/dynamics_loss=0.01895, val/reward_loss=0.07056
policy iter 1: actor_loss=2.6539, value_loss=0.3513
policy iter 2: actor_loss=3.1435, value_loss=0.2507
```

The adapter now passes the minimal original-PWM algorithm smoke path:

```text
load MJLab-QS windows -> pack PWM observations -> original WM update -> original actor/critic update
```

## Formal run initial health check

Formal jobs `5272012`, `5272014`, and `5272015` all started and are logging to W&B. Initial pretrain curves are decreasing under the original PWM WM loss.

Examples from the first ~7k pretrain iterations:

```text
seed0/H100: val/wm_loss 0.0846 at iter 1 -> 0.0239 at iter 1000 -> 0.0084 at iter 7000
seed1/H200: val/wm_loss 0.0996 at iter 1 -> 0.0335 at iter 1000 -> 0.0234 at iter 7000
seed2/L40S: val/wm_loss 0.0963 at iter 1 -> 0.0295 at iter 1000 -> 0.0094 at iter 7000
```

This indicates the original PWM adapter is not stuck at startup. Final policy extraction and MJLab eval still need to finish before comparing against the state-space PWM/Flow endpoint runners.
