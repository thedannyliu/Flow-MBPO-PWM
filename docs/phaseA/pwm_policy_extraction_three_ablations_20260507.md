# MJLab-QS PWM Policy Extraction: Three Ablations

Date: 2026-05-07
Branch: dev/mjlab

## Goal

Return to the main MJLab-QS state-space PWM policy extraction runner and test three targeted ablations that can explain why the offline PWM-style extracted policies currently get weak real-environment returns.

These are not new datasets or new WM checkpoints. They use the same fixed G1 QS dataset and the same A2.5 WM checkpoints:

```text
dataset: scripts/outputs/mjlab_qs/windows/a25_native_qs_g1stage4/velocity_flat_unitree_g1/d_qs_core_h16.pt
MLP WM checkpoints: scripts/outputs/mjlab_qs/results/a25_native_qs_g1stage4/velocity_flat_unitree_g1/mlp_ref/seed_*/best.pt
Flow endpoint WM checkpoints: scripts/outputs/mjlab_qs/results/a25_native_qs_g1stage4/velocity_flat_unitree_g1/flow_endpoint/seed_*/best.pt
```

Each ablation runs the 2x2 policy extraction table:

```text
MLP WM + MLP policy
MLP WM + Flow policy
Flow endpoint WM + MLP policy
Flow endpoint WM + Flow policy
```

with 3 seeds.

## Ablation 1: no action penalty

Stage:

```text
offline_pwm_ablate_no_action_l2_20260507
```

Change:

```text
action_l2: 1e-4 -> 0
```

Reason:

The current state-space PWM runner subtracts an explicit action L2 penalty from imagined reward:

```text
r_model(z, a, c) - action_l2 * ||a||^2
```

The original PWM actor loss does not add this exact penalty in the same location. If G1 locomotion requires large actions to initiate gait, this penalty may suppress useful behavior and produce short, low-return episodes.

W&B:

```text
flow-mbpo-mjlab-ablation-no-action-l2
```

Submitted jobs:

```text
5272057: H100, seed 0, 4-row 2x2
5272058: H200, seed 1, 4-row 2x2
5272059: L40S, seed 2, 4-row 2x2
```

## Ablation 2: stronger critic

Stage:

```text
offline_pwm_ablate_strong_critic_20260507
```

Change:

```text
critic_iterations: 8 -> 16
critic_batches: 4 -> 8
```

Reason:

PWM policy extraction depends heavily on TD(lambda) critic targets computed from imagined rollouts. If the critic is underfit or unstable, the actor may optimize a bad value landscape even if the WM is good enough. This ablation increases critic optimization without changing the WM.

W&B:

```text
flow-mbpo-mjlab-ablation-strong-critic
```

Submitted jobs:

```text
5272060: H100, seed 0, 4-row 2x2
5272061: H200, seed 1, 4-row 2x2
5272062: L40S, seed 2, 4-row 2x2
```

## Ablation 3: actor behavior-cloning warm start

Stage:

```text
offline_pwm_ablate_bc_warmstart_20260507
```

Change:

Before imagined PWM actor-critic extraction, warm-start the actor on offline collector actions:

```text
min_theta E[ || pi_theta(z_t, c_t) - a_t ||^2 ]
```

Settings:

```text
bc_warmstart_iters: 10000
bc_lr: 5e-4
bc_batch_size: 256
then policy_iters: 15000
```

Reason:

The current offline PWM extraction starts policy optimization from a random actor. In MJLab G1, random actors may quickly enter bad imagined regions, and the real-env policy can remain close to random/weak behavior. A BC warm start tests whether giving the actor a data-manifold prior is necessary before first-order imagined-return optimization.

W&B:

```text
flow-mbpo-mjlab-ablation-bc-warmstart
```

Smoke:

```text
5272053: BC warm-start smoke, L40S, no W&B, no real eval
```

Submitted formal jobs:

```text
5272063: H100, seed 0, 4-row 2x2
5272064: H200, seed 1, 4-row 2x2
5272065: L40S, seed 2, 4-row 2x2
```

## Code changes

Updated files:

```text
scripts/experiments/mjlab_qs/run_offline_pwm_policy_extraction.py
scripts/experiments/mjlab_qs/run_policy_extraction_row.py
```

New runner args:

```text
--action-l2
--bc-warmstart-iters
--bc-lr
--bc-batch-size
--bc-eval-every
```

The manifest row runner now forwards these fields to the training script.

## Interpretation plan

Primary comparison:

```text
real MJLab eval return_mean, return_std, episode_length_mean
```

Secondary diagnostics:

```text
imagined_return
actor_loss
critic_loss
action_norm
actor_std
BC action MSE for BC runs
```

Decision rules:

- If `action_l2=0` improves return, the previous imagined reward penalty was likely suppressing useful locomotion actions.
- If stronger critic improves return, policy extraction is critic-limited rather than WM-limited.
- If BC warm-start improves return, random-start imagined PWM extraction is too brittle for this MJLab G1 dataset, and future PWM/Flow comparisons should include policy prior or BC initialization.
- If none improve return, the bottleneck is more likely WM reward/dynamics mismatch, dataset quality/coverage, or the mismatch between offline imagined extraction and real MJLab deployment.
