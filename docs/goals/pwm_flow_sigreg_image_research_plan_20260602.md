# PWM, Flow-PWM, SIGReg, and Image-Based World-Model Research Plan

Date: 2026-06-02

## Objective

Turn the current observation that a Flow-based world model plus Flow-based policy architecture in a PWM-style method may outperform faithful PWM into a controlled, well-documented research program. Do not replace the existing PWM fidelity gate; extend it. Claims still require real MJLab eval, final and true-best actors, MP4/W&B videos, return, episode length, and fall rate.

References:

- PWM paper: https://arxiv.org/abs/2407.02466
- LeWorldModel / LeWM: https://arxiv.org/abs/2603.19312
- LeWM code: https://github.com/lucas-maes/le-wm
- NEWT: https://arxiv.org/abs/2511.19584
- NEWT code/project: https://github.com/nicklashansen/newt and https://www.nicklashansen.com/NewtWM/

Source note: LeWM is a pixel-based JEPA-style world model using a next-embedding prediction loss plus SIGReg, a Gaussian latent regularizer. Use that as inspiration for regularizing our state/image latent world models; do not claim LeWM parity until we run their code/baselines.

## Non-Negotiable Evidence Gates

1. Record the status and results of current no-dependency jobs in `docs/goals/pwm_fidelity_mjlab_flow_migration_20260601.md`.
2. Preserve original PWM parity evidence: Hopper locked-env parity is already established; Ant true-env eval and Phase 2 probes are supplemental.
3. Any MJLab claim must include:
   - real eval for final actor and true-best actor;
   - 40 episodes for eval;
   - 10-episode 1000-step MP4/W&B videos;
   - return, episode length, fall rate;
   - comparison against expert, expert-noisy, medium, random/reference, and best BC.
4. Imagined return, WM loss, or short smoke output is diagnostic only.
5. Flow replacement rows must change one variable at a time unless explicitly labeled exploratory.

## Phase A: Consolidate Current Flow-PWM Evidence

Goal: verify whether current Flow WM + Flow policy architecture is actually better than faithful PWM under the same MJLab real-eval protocol.

Tasks:

1. Inventory current best Flow-PWM runs:
   - config path;
   - git SHA;
   - dataset/window version;
   - seed;
   - checkpoint paths;
   - W&B links;
   - final and best actor availability;
   - whether best is true real-eval best or training/imagined best.
2. Run or locate final/best real eval and videos.
3. Create a table:
   - faithful PWM adapter;
   - previous PWM-style runner;
   - Flow WM only if available;
   - Flow policy only if available;
   - Flow WM + Flow policy;
   - best BC and expert references.
4. If Flow-PWM advantage is only imagined-return-based, mark it unverified.

Deliverable:

- Update the active goal doc with a result table and a conservative diagnosis:
  - Flow-PWM verified better;
  - Flow-PWM promising but unverified;
  - Flow-PWM exploits model;
  - faithful PWM still strongest.

## Phase B: Controlled State-Based A/B Rows

Run a clean matrix with fixed MJLab QS data, fixed horizon, fixed actor/critic update budget, fixed eval protocol, and fixed seed first.

Rows:

```text
R0: faithful original PWM WM + original PWM policy/update
R1: Flow WM + original PWM policy/update
R2: original PWM WM + Flow policy architecture
R3: Flow WM + Flow policy architecture
R4: best current Flow-PWM config, exact reproduction
```

Log for each row:

```text
WM loss
reward calibration
done/fall calibration
gradient smoothness
imagined return
actor gradient norm
action drift
action saturation
support/OOD distance
real return
episode length
fall rate
final/best checkpoint paths
video paths/W&B links
```

Conclusion rules:

- R3 > R0 in real eval and fall rate does not worsen: Flow architecture is useful.
- R3 improves imagined return but worsens fall/length: model exploitation.
- R1 improves but R2 does not: Flow WM is the important variable.
- R2 improves but R1 does not: policy architecture/update is the important variable.
- R4 beats matrix rows only because it changes multiple variables: treat as exploratory, not causal.

## Phase C: Add SIGReg-Style Latent Regularization

Goal: test whether LeWM-inspired SIGReg stabilizes Flow/PWM latent dynamics and reduces exploitation.

Implementation order:

1. Read LeWM paper/code enough to identify the SIGReg objective and expected input shape.
2. Implement a small local SIGReg-style latent regularizer for state-based latent batches:
   - random projections over latent embeddings;
   - penalty encouraging projected latents to match an isotropic Gaussian;
   - single configurable weight;
   - logged as a separate loss term.
3. Add unit/smoke tests for:
   - finite loss;
   - gradients flow to latent encoder/model;
   - zero/constant embeddings receive nonzero anti-collapse penalty;
   - no change when weight is zero.
4. Test on the best Flow-PWM row:
   - no SIGReg;
   - low SIGReg;
   - medium SIGReg;
   - high SIGReg only if low/medium are stable.

Metrics:

```text
latent variance
latent covariance/isotropy proxy
WM loss
reward/fall calibration
gradient smoothness
action drift
support/OOD distance
real eval/video metrics
```

Stop condition: if SIGReg improves WM diagnostics but worsens real rollout/fall, do not tune it blindly; treat it as representation regularization that still needs pessimism/support gating.

## Phase D: Pessimistic Short-Horizon Flow-MBPO

If PWM/Flow-PWM still exploits learned dynamics, prioritize pessimistic synthetic rollouts over long-horizon differentiable policy extraction.

Rows:

```text
M0: best BC
M1: faithful PWM
M2: best Flow-PWM
M3: Flow endpoint MBPO + AWR/AWAC
M4: Flow residual/chunk MBPO + AWR/AWAC
M5: M4 + uncertainty penalty
M6: M5 + support/OOD gate
M7: M6 + fall-risk early termination
M8: M7 + conservative Q
M9: M8 + SIGReg-style WM regularization
```

Keep synthetic horizon short first: `H=1,3,5`. Start rollouts from real dataset states. Require real eval/video evidence before any claim.

## Phase E: Image-Based Tasks, NEWT, and LeWorldModel

Only start image-based work after the state-based story is clean enough to know what we are testing.

Entry conditions:

```text
state-based Flow-PWM beats faithful PWM or BC under real eval;
or pessimistic Flow-MBPO preserves/improves BC and reduces fall;
or state-based failures are cleanly diagnosed as representation/observation limitations.
```

NEWT track:

1. Clone/reference NEWT externally, not inside core code unless vendor policy is approved.
2. Reproduce one official image-based task smoke.
3. Define a small image-based benchmark matrix:
   - NEWT baseline;
   - our Flow WM;
   - our Flow WM + SIGReg;
   - pessimistic Flow-MBPO variant if applicable.
4. Use the same discipline:
   - real eval;
   - videos;
   - final/best checkpoints;
   - W&B logs;
   - fixed seeds and configs.

LeWorldModel comparison:

1. Run LeWM official code on one supported task if feasible.
2. Compare against our image-based Flow/SIGReg model on matched task/eval where possible.
3. Do not claim superiority over LeWM unless we run their baseline under comparable data, compute, and evaluation.

## Documentation and Commit Rules

Commit meaningful docs/config/scripts with English messages. Record every formal run with:

```text
git SHA
command
config
env/dataset/version
seed
GPU/QOS
W&B run
checkpoint paths
eval/video paths
results
failure reason
next action
```

Keep smoke tests W&B-disabled. Use `embers` for formal GPU jobs unless `inferno` is explicitly approved. Multiple GPU jobs may be submitted at once, prioritizing H200, H100, A100, L40S, then lower-tier GPUs.

## Scheduling Policy

Submit aggressively when the plan has several useful independent experiments. Do not block broad experiment submission behind Slurm dependencies unless there is a hard data artifact requirement that makes the downstream job impossible to start. If a submitted job later proves invalid because of a config, environment, dataset, checkpoint, or wrapper mistake, cancel it and record the failure reason, affected job IDs, and replacement job IDs.

Agent execution rules:

1. Before submitting, list the candidate jobs and classify each as smoke, diagnostic, eval, formal, or exploratory.
2. Submit every candidate that already has all required input files. Do not wait for earlier phases only for sequencing aesthetics.
3. Do not use `--dependency` / `afterok` unless the command needs an artifact that does not exist yet.
4. Prefer GPU queues in this order: H200, H100, A100, L40S, then lower-tier GPUs.
5. Use `embers` for GPU jobs. Do not use `inferno` unless explicitly approved.
6. Keep new-code-path smoke tests W&B-disabled.
7. Use W&B for formal jobs and include notes with git SHA, command, config, env/dataset/version, seed, checkpoint paths, and expected evidence.
8. If a submitted job is wrong, cancel it with `scancel <job_id>` as soon as the mistake is known.
9. For every failed/canceled/replacement job, update the goal doc in English with job ID, status, root cause, fix, replacement job ID, and whether the result is usable.
10. Commit meaningful doc/config/script updates in English after each scheduling or result milestone.

Default behavior summary:

```text
submit many useful GPU jobs in parallel
avoid dependency chains by default
submit first, inspect/cancel/fix/resubmit if wrong
record all results and failures in English
commit meaningful docs/config/scripts with English git messages
```

This policy is intentionally throughput-oriented. Incorrect jobs may be canceled after inspection; the durable requirement is that every failure and resubmission is recorded in English with clear git commits and doc updates.
