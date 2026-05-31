# Flow-MBPO Top-Conference Research Plan

Date: 2026-05-31

This is the canonical research-plan entry point for the MJLab QS Flow-MBPO track.
The dated source notes are under `docs/goals/0531/`.

## Thesis

The project should test Pessimistic Flow-MBPO for contact-rich humanoid
locomotion:

```text
Flow residual/trajectory world model ensemble
-> conservative short synthetic rollouts from real dataset states
-> BC-warmstarted conservative policy update
-> real MJLab eval, rollout MP4/W&B video, return, length, and fall-rate gates
```

Flow models should be treated as distributional short-horizon trajectory
generators, not as drop-in MLP replacements for unconstrained long-horizon actor
optimization.

## Current Evidence

- Collector/reference policies remain the true target.
- BC is a useful warm start and minimum baseline, but remains far below the
  expert collector.
- PWM-style imagined optimization and weakly constrained Flow-MBPO updates can
  improve scalar diagnostics while failing the rollout-video fall-rate gate.
- Trajectory/chunk Flow-MBPO now produces useful return/length gains on seed0,
  but both the `r224/s32` and `r240/s16` AWR variants tie matched BC video fall
  rate instead of reducing it.
- The next bottleneck is not another small BC/data tweak or plain synthetic
  ratio sweep; it is missing pessimism, fall calibration, and support control.

## Method Scope

Compare only the smallest matrix needed to answer whether Flow helps:

| Method | Purpose |
|---|---|
| Best BC | minimum warm-start baseline |
| MLP one-step MBPO + AWR | non-Flow MBPO baseline |
| Flow endpoint MBPO + AWR | current Flow baseline |
| Flow residual/chunk MBPO + AWR | distributional trajectory Flow |
| Flow residual/chunk + uncertainty/fall penalty | main method |
| Flow residual/chunk + conservative Q | fallback if explicit uncertainty is weak |
| Flow action-chunk prior + advantage weighting | optional policy-side Flow test |

## Required Pessimism

Every formal v1 candidate should expose or approximate:

- ensemble uncertainty;
- done/fall probability;
- uncertainty/fall early termination;
- conservative reward
  `reward_model - lambda_uncertainty * uncertainty - lambda_fall * fall_prob`;
- support/OOD gating for synthetic state-action pairs;
- final and true-best actor evaluation.

If the done/fall head collapses near zero, treat the synthetic MDP as unsafe and
fix calibration before using longer or less constrained synthetic rollouts.

## Claim Gate

No policy-improvement claim is allowed without:

- W&B run with git SHA, command, dataset/version, seed, checkpoint paths, and
  config;
- final actor and true-best actor checkpoints;
- 40-episode real MJLab eval;
- 10-episode 1000-step rollout MP4/W&B videos;
- comparison to expert, expert-noisy, medium, random/reference, and best BC;
- return and episode length above best BC, with fall rate below best BC.

## Immediate Next Step

Implement the smallest Flow-MBPO v1 pessimistic slice. Current status:

1. Done: add fall-aware replay fields and reward penalty plumbing.
2. Still required: calibrate or repair the trajectory/chunk done/fall signal before relying on
   model termination.
3. Done: add a KL/action-deviation constraint to the AWR policy update or a
   conservative-Q fallback.
4. Done for current slice: W&B-disabled trajectory/chunk H3 AWR action-deviation
   smoke passed mechanically in job `9354631`, then one formal W&B seed ran on
   `embers` in jobs `9354764` and `9354806`.
5. Done for current slice: final and true-best actors were evaluated/rendered.
   The action-deviation variant failed the strict gate because scalar fall
   regressed versus matched BC and rollout fall still tied BC. Do not expand it.
   Next add a real fall/support/OOD risk signal or conservative-Q penalty before
   another formal policy update.
6. Partly done: state/action support-OOD replay penalty plumbing is implemented
   and passed W&B-disabled replay/AWR smokes. The first q90 support setting is a
   mild penalty on the current H3 replay, so it needs calibration or a stronger
   local-support variant before any formal run.
7. Partly done: q50 support gating is a stronger pessimistic replay and passes
   AWR smoke. Existing rollout logs lack full state/action vectors, so support
   distance still cannot be tied to real fall events without adding richer
   rollout logging or a separate calibration collection.
