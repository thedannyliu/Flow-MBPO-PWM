# MJLab QS Flow-MBPO High-Value Next Goal

Date: 2026-05-29  
Target file to update: `docs/goals/mjlab_qs_rollout_policy_improvement_20260528.md`

## Purpose

Update the current MJLab QS rollout-policy-improvement goal so the next agent does not spend most of its effort on low-value data/BC micro-optimizations. Keep the existing reproducibility and rollout-evidence rules, but redirect the research objective toward a high-value Flow-MBPO method.

The next phase should answer:

> Can Flow-based world models provide useful short-horizon synthetic rollouts for real policy improvement in Velocity Flat Unitree G1, beyond better imagined return or lower world-model loss?

## Current state to preserve

The current evidence protocol is good and should not be weakened.

- Collector/reference policies have real MJLab rollout evidence and videos.
- Expert and expert-noisy collectors are the real targets.
- BC/IL baselines are useful sanity checks and warm starts.
- Current PWM-style extracted policies remain far below BC and collector behavior.
- Imagined return and world-model MSE are diagnostic only.
- Final and true-best actors must both be saved and evaluated.
- All formal runs must use W&B.
- All GPU jobs must use `embers` QOS unless `inferno` is explicitly approved.
- Meaningful changes must be committed with clear English git messages.

## New research stance

Do not frame the next phase as another architecture-swap PWM sweep. Frame it as:

> Flow-MBPO for contact-rich humanoid locomotion: use Flow world models to generate conservative short-horizon synthetic rollouts, then improve a BC-warmstarted policy using model-free offline/online RL updates, with real-environment rollout evidence as the only success criterion.

This is higher value than continuing to tune small BC/data details because the current bottleneck is not only data cleanliness. The observed failure mode is model exploitation: imagined metrics improve while real rollout remains collapsed.

## Stop doing by default

Do not spend major time on these unless a fresh diagnosis directly requires them:

- more yaw-bin sampling variants;
- small action-smoothing coefficient sweeps;
- naive medium-data mixing;
- one-step world-model MSE comparisons without downstream value;
- longer unconstrained PWM policy extraction;
- claiming "best" checkpoints from best imagined return without real rollout evidence.

## Core implementation track: Flow-MBPO v0

Implement a new path separate from the current long-horizon PWM-style actor optimization.

### World models

Train and compare:

1. `mlp_one_step_ensemble`
2. `flow_endpoint_ensemble`
3. `flow_residual_ensemble`
4. `flow_trajectory_chunk_ensemble`

The Flow trajectory model should predict short chunks, not only one-step endpoints:

```text
p(s_{t+1:t+H}, r_{t:t+H}, done/fall_{t:t+H} | s_t, a_{t:t+H-1}, command)
```

Start with `H = 1, 3, 5, 10`.

Prefer residual/chunk modeling over pure Flow replacement if pure Flow is unstable.

### Synthetic rollout generation

Generate short model rollouts from real dataset states:

```text
real dataset state -> current BC-warmstarted policy -> model rollout for H steps
```

Do not start with long free-running imagination. Do not backpropagate an unconstrained actor through long imagined horizons.

### Policy update

Start from the strongest BC/IL checkpoint. Update with a model-free objective using mixed real/synthetic data:

- SAC-style update, or
- AWAC/AWR-style advantage-weighted update, or
- PPO-style clipped update.

The policy update should consume synthetic transitions, not directly exploit raw long-horizon model gradients.

### Conservatism

Add uncertainty-aware pessimism from the beginning:

```text
r_conservative = r_model - lambda_uncertainty * uncertainty
```

Uncertainty candidates:

- ensemble next-state disagreement;
- ensemble trajectory-chunk disagreement;
- reward/done/fall disagreement;
- latent/state-action OOD score;
- rollout-horizon error proxy.

Also support:

- synthetic rollout early termination when uncertainty is high;
- early termination when predicted fall probability is high;
- synthetic:real batch ratio sweep;
- horizon curriculum;
- real-eval-based early stopping.

## Minimal experiment matrix

Run the smallest matrix that can answer the research question:

| Method | Purpose |
|---|---|
| BC only | warm-start and minimum baseline |
| MLP one-step MBPO | non-Flow MBPO baseline |
| Flow endpoint MBPO | current Flow WM as MBPO model |
| Flow residual/chunk MBPO | higher-value Flow use |
| Flow residual/chunk MBPO + uncertainty | main proposed method |
| optional Flow action-chunk prior + advantage weighting | policy-side Flow test |

Do not expand to many seeds until one-seed rollout videos show non-degenerate behavior. Once a method clears BC on one seed, run 3 seeds and 40-episode eval.

## Evaluation and claim policy

A result is useful only if it includes:

- real MJLab eval;
- rollout MP4s and W&B videos;
- return;
- episode length;
- fall rate;
- comparison against expert collector, expert-noisy collector, medium collector, random/reference, and best BC;
- final and true-best actor evaluation;
- W&B URL and git SHA.

Success means:

```text
real return > best BC baseline
episode length > best BC baseline
fall rate < best BC baseline
videos visibly no worse than BC
```

A method that improves imagined return but reduces real rollout quality is a failure or diagnostic, not policy improvement.

## SigReg track

Keep SigReg, but narrow its role.

Use it as a world-model/latent regularization ablation for:

- Flow endpoint;
- Flow residual/chunk;
- image-based future tasks.

Judge SigReg by:

- long-horizon prediction;
- reward/done/fall calibration;
- expert-in-model behavior;
- downstream Flow-MBPO real rollout.

Do not prioritize SigReg if it only improves MSE or latent statistics.

## Image-based / LeWorldModel track

Do not move to image-based NEWT/Newt until the state-based Flow-MBPO protocol is credible.

Move only if one of these holds:

1. Flow-MBPO beats BC in real MJLab rollout.
2. Flow-MBPO preserves BC and clearly reduces model exploitation.
3. The failure mode is well documented enough that image tasks are a representation-learning study rather than unresolved action/evaluation debugging.

When moving to image tasks, repeat the same evidence protocol and compare with LeWorldModel only under matched dataset, compute, seeds, and real-eval/video conditions.

## Immediate next actions for the agent

1. Update `docs/goals/mjlab_qs_rollout_policy_improvement_20260528.md` to point to this high-value Flow-MBPO track.
2. Create or update a short design doc: `docs/design/flow_mbpo_v0.md`.
3. Add an implementation checklist for:
   - model ensemble interface;
   - synthetic rollout buffer;
   - uncertainty penalty;
   - SAC/AWAC/PPO-style update path;
   - real/synthetic ratio sweep;
   - horizon sweep;
   - eval/render pipeline reuse.
4. Implement the smallest smoke test with W&B disabled.
5. Commit the smoke infrastructure.
6. Run formal one-seed Flow-MBPO v0 with W&B enabled on `embers`.
7. Evaluate and render final + true-best actors.
8. Only expand if real rollout is at least BC-preserving.

## Final instruction

The next deliverable is not another table where Flow has slightly better imagined return. The next deliverable is a reproducible Flow-MBPO pipeline showing whether Flow-generated short-horizon synthetic rollouts can improve or at least preserve real humanoid locomotion relative to BC.
