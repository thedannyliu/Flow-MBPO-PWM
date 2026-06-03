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

PWM comparator note: the current MJLab row called `original_pwm_adapter` is an
adapter-level comparator, not a byte-identical upstream PWM pipeline run. It
imports upstream `baselines/PWM/src/pwm.algorithms.pwm.PWM` and uses the
upstream actor, critic, SimNorm world model, `compute_wm_loss`, `update`,
TD(lambda), return RMS, and LR schedule, but it owns the MJLab-QS window
sampling, pretrain-loop orchestration, policy-update-loop orchestration, and
MJLab eval bridge. It should be treated as "upstream PWM algorithm/model/update
adapted to MJLab-QS", not as proof that `baselines/PWM/scripts/train_dflex.py`
or `train_multitask.py` fails on MJLab.

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

MJLab reference points to keep in every result table:

```text
expert collector: return 82.6090, length 1000, fall 0
best aggregate BC: return about 45.8491, length about 594.97, fall about 0.625
```

## Execution Layers

Use three concurrent layers, not a strict serial pipeline:

1. Immediate gates: finish and document the currently submitted PWM/Flow/MJLab jobs.
2. Evidence-based branching: as soon as results show a reasonable signal, launch the next useful diagnostic or comparison without waiting for perfect closure.
3. Long-horizon research tracks: start SIGReg, pessimistic Flow-MBPO, NEWT, and LeWorldModel preparation when there is enough signal to justify the test, even if other jobs are still running.

Reasonable signal means at least one of:

```text
PWM algorithm adapter or Flow-PWM smoke runs without runtime issues;
Flow-PWM has higher real eval than a PWM baseline on the same dataset/protocol;
Flow-PWM improves imagined return while diagnostics show possible exploitation;
the PWM algorithm adapter collapses in MJLab despite original DFlex parity;
fall/support/OOD diagnostics identify a specific failure boundary;
an image-based setup can be reproduced cheaply enough to prepare the next comparison.
```

Early signals are enough to submit more jobs, build infrastructure, and run smokes. They are not enough for performance claims; claims still require final/best eval, videos, and baseline comparisons.

## Environment Policy

Maintain two separate execution environments and record which one is used for every submitted job.

```text
project/current environment:
  conda env `pwm`
  purpose: current Flow-MBPO code, MJLab QS runners, policy eval/render tools,
    manifest utilities, and new-code diagnostics.
  audit on 2026-06-02 login node: Python 3.10.19, torch 2.10.0+cu128,
    Hydra 1.3.2, OmegaConf 2.3.0, W&B 0.23.0; CUDA unavailable on login node.

locked original-PWM reproduction environment:
  /storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4
  purpose: original PWM/DFlex parity and faithful PWM reproduction evidence.
    This is the more credible environment for PWM reproduction claims.
  audit on 2026-06-02 login node: Python 3.10.14, torch 2.3.1/cu118,
    Hydra 1.2.0, OmegaConf 2.2.3, W&B 0.12.21; CUDA unavailable on login node.
  caveat: direct login-node DFlex import tries to rebuild kernels in read-only
    site-packages and fails. Use the repaired Slurm wrappers with a job-local
    DFlex sandbox and locked compiler/CUDA exports for DFlex jobs.

hybrid locked MJLab bridge:
  scripts/experiments/mjlab_qs/locked_mjlab_python.py
  purpose: load locked torch/tensordict/torchrl/PWM first, then expose project
    MJLab packages for short faithful-adapter MJLab checks without modifying
    either conda environment.
```

## Agent Preflight And History Inventory

Run this section before any new GPU submission. The purpose is to prevent duplicate, stale, or non-reproducible jobs. Do not submit new jobs until the preflight inventory has been written and committed.

1. Inspect repository state:
   - current branch;
   - current git SHA;
   - `git status --short`;
   - recent commits relevant to PWM, Flow, MJLab, SIGReg, Slurm, W&B, configs, and docs.
2. Inspect prior GPU submission records:
   - use `squeue -u $USER`, `sacct`, and `seff` when available;
   - check the known gate job IDs: `9387942`, `9387949`, `9387896`, and `9387895`;
   - search `docs/git`, `docs/goals`, scripts, configs, Slurm outputs, W&B/offline directories, checkpoint directories, and video/eval artifacts for job IDs, sbatch commands, W&B links, config paths, checkpoint paths, and result summaries.
3. Write an English inventory table before launching new jobs. Each row should include:

```text
job ID
purpose
status
command or sbatch script
git SHA
config
env/dataset/version
seed
GPU/QOS
W&B link or offline directory
checkpoint paths
eval/video paths
return, episode length, fall rate when available
failure reason when failed
whether the result is usable
next action
```

4. Commit the inventory update with an English message.

Only after this preflight commit should the agent list candidate jobs and begin broad no-dependency submission.

## Immediate Work: Current Gate Jobs

Record and interpret the active no-dependency jobs first. If any job ID is replaced, record the replacement in the active fidelity doc.

```text
9387942  Ant final/best true DFlex eval
9387949  Hopper final/best WM-vs-real probe
9387896  MJLab faithful original PWM adapter smoke
9387895  MJLab faithful original PWM adapter formal
```

Judgment:

```text
Ant true eval pass:
  strengthens original PWM parity and reduces the chance that Hopper was a lucky task.

Hopper WM-vs-real probe pass:
  learned-WM reward, action behavior, and termination diagnostics do not show an obvious mismatch.

MJLab smoke pass:
  adapter/runtime is usable enough to submit broader MJLab diagnostics and Flow comparisons.

MJLab formal pass or fail:
  triggers final/best eval and videos; it does not by itself prove success or collapse.
```

After any Phase 3 MJLab formal run writes checkpoints, immediately submit:

```text
final_policy_extraction.pt -> 40-episode real MJLab eval
best_policy_extraction.pt  -> 40-episode real MJLab eval
final/best                 -> 10-episode, 1000-step MP4/W&B videos
metrics                    -> return, episode length, fall rate
comparisons                -> expert, expert-noisy, medium, random/reference, best BC
```

Without these eval/video artifacts, do not make a MJLab performance claim.

## Phase A: Consolidate Current Flow-PWM Evidence

Goal: verify whether current Flow WM + Flow policy architecture is actually better than the upstream PWM algorithm adapter under the same MJLab real-eval protocol.

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
   - upstream PWM algorithm adapter;
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
  - PWM algorithm adapter still strongest.

Current Phase A evidence status, updated 2026-06-03:

| Comparator | Evidence | Conservative diagnosis |
| --- | --- | --- |
| Upstream PWM algorithm adapter | Final/best eval40 and rollout10 gates from fix2 jobs are complete; final/best collapse with fall `1.000`. | Negative adapter-level R0 baseline. |
| Full upstream PWM pipeline bridge | Smoke `9401871` completed 0:0 through `train_dflex.py` and upstream `pwm.algorithms.pwm.PWM`; longer diagnostic `9401906` completed 0:0, wrote best/final and iter50/100/150 checkpoints, and ended with poor internal eval length `35.33`; W&B-disabled final/best real-env eval smoke `9401975` completed 0:0 with final return `-1.7458`, length `38.125`, fall `1.000`, and best return `-1.6728`, length `47.500`, fall `1.000`; duplicate H100 backup `9401980` was canceled. User direction then requested a complete PWM pipeline MJLab effect check, so W&B-on formal final/best eval40 plus rollout10 replacements `9402769`/`9402771`/`9402774`/`9402772`/`9402773`/`9402770` were submitted after canceling W&B-disabled arrays `9402742`-`9402747`; all replacements are pending. | Pipeline mechanically feasible, but current smoke evidence is negative and formal W&B gates are pending. Do not make a full-upstream PWM performance claim until the replacement eval40 and rollout10 artifacts complete. |
| Flow-MBPO H1 endpoint AWR | Final/best eval40 and final/best roll10 summaries exist. Eval final is strong (`60.8721`, length `759.30`, fall `0.450`), but eval best regresses (`46.1720`, fall `0.700`); video best ties matched BC fall `0.400` while final video is worse (`0.500`). | Promising but unverified; split final/best behavior and no strict fall improvement. |
| Flow-MBPO trajectory/chunk H3 | Final/best-real eval40 and roll10 summaries exist. Final eval is above aggregate BC scalar (`48.7296`, fall `0.575`), but best-real eval regresses (`37.5778`, fall `0.800`); videos tie matched BC fall `0.400`. | Promising but unverified; not enough for seed expansion. |
| Flow-MBPO trajectory/chunk H3 low synthetic ratio | Final/best eval40 and roll10 summaries exist. Video return/length are strong and fall ties matched BC, but eval best regresses and no fall improvement is shown. | Promising but unverified. |
| Conservative broad Flow-MBPO AWR sweep | Completed H200/H100/L40S rows are all below BC and fall at `1.000`; A100 rows remain pending. | Negative diagnostic; do not duplicate this setting. |

Current diagnosis: Flow-MBPO has better seed0 diagnostic policies than the PWM
adapter and prior PWM-style runner, but it is still not a verified policy
improvement because the final/best gates are inconsistent and the strict matched
video fall gate is not improved. Treat this as "Flow-PWM promising but
unverified" with continuing evidence for model-exploitation/fall-risk issues.

## Phase A2: Branch On Faithful PWM Results

If the upstream PWM algorithm adapter on MJLab is poor while original DFlex parity holds, prioritize transfer and fall-protocol diagnostics while still allowing Flow, SIGReg, NEWT, and LeWorldModel preparation to proceed in parallel.

Likely causes to test:

```text
MJLab QS / G1 / fall-gated offline protocol mismatch
fall boundary missing from the learned model
dataset support gaps
policy exploiting fake-safe learned-model states
reward, termination, timeout, or fall-signal mismatch
```

Diagnostics worth running:

```text
action saturation and action drift
support/OOD distance
fall-risk windows
done, timeout, termination, and bootstrap masks
real rollout fall reason
BC action distribution vs PWM/Flow action distribution
expert action replay through WM vs logged segments
short actor-update exploitation test at 10, 50, and 100 updates
```

Key interpretation:

```text
imagined return increases while real return/length/fall worsens
  -> model exploitation, not proof that a stronger Flow model alone solves the task
```

If the upstream PWM algorithm adapter clearly beats the previous PWM-style runner, treat the previous implementation as having a fidelity gap or bug. Promote the adapter to a formal MJLab baseline and compare:

```text
Row 0: upstream PWM algorithm/model/update via MJLab-QS adapter
Row 1: current PWM-style runner
Row 2: true upstream-pipeline bridge if one is built
```

This answers whether the problem is PWM transfer to MJLab or a nonfaithful previous implementation.

## Phase B: Controlled State-Based A/B Rows

Run a clean matrix with fixed MJLab QS data, fixed horizon, fixed actor/critic update budget, fixed eval protocol, and fixed seed first.

Rows:

```text
R0: upstream PWM algorithm/model/update via MJLab-QS adapter
R1: Flow WM + original PWM policy/update
R2: original PWM WM + Flow policy architecture
R3: Flow WM + Flow policy architecture
R4: best current Flow-PWM config, exact reproduction
```

Keep fixed unless a row explicitly studies the variable:

```text
horizon
actor and critic sizes
3 critics
TD(lambda)
policy batch size
WM batch size
reward formulation
real eval episode count
video protocol
dataset/window version
seed set
```

Log for each row:

```text
WM loss
one-step and H-step prediction
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

## Phase B2: Control-Relevant World-Model Study

Do not rank world models by MSE alone. PWM's key lesson is that smooth, regularized learned dynamics can be more useful for first-order policy learning than a lower-error but poorly shaped predictor.

Worth testing as controlled variants:

```text
MLP / ReLU WM
Mish WM
SimNorm latent PWM WM
Flow endpoint WM
Flow residual WM
Flow trajectory/chunk WM
Flow trajectory/chunk + SIGReg
Flow trajectory/chunk + uncertainty/fall/support heads
```

Rank candidates by control-relevant evidence:

```text
one-step prediction
H-step prediction
reward calibration
done/fall calibration
expert-in-model behavior
gradient smoothness
imagined return
action drift
action saturation
support/OOD distance
real MJLab return
episode length
fall rate
video quality
```

Useful finding patterns:

```text
lower MSE but worse real policy:
  model is not control-relevant enough.

smooth gradients plus stable real eval:
  model may provide useful policy-improvement signal.

high imagined return plus action drift/OOD/fall:
  model is being exploited.
```

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
M1: upstream PWM algorithm adapter
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

Minimum implementation details:

1. Use BC-warmstarted policies before adding high-variance RL updates.
2. Keep the synthetic:real ratio small-to-medium first.
3. Store generated transitions with source tags:
   - real;
   - Flow endpoint;
   - Flow residual;
   - Flow chunk;
   - uncertainty-gated;
   - support-gated;
   - fall-risk-gated.
4. Log why each synthetic rollout stops:
   - horizon reached;
   - done/termination predicted;
   - uncertainty over threshold;
   - support/OOD over threshold;
   - fall-risk over threshold.
5. Compare AWR/AWAC first, then conservative Q. If AWR/AWAC cannot preserve BC, conservative Q is a later safety intervention, not a substitute for debugging the data/model path.

Interpretation:

```text
M4 > M3:
  residual/chunk trajectory modeling is useful.

M5/M6/M7 reduce fall while M4 does not:
  the useful contribution is pessimistic model use, not Flow alone.

M8 improves return but worsens fall:
  conservative Q is miscalibrated or exploiting synthetic transitions.
```

## Phase D2: Fall, Support, and OOD Boundary Track

This track is high priority because MJLab humanoid failure is likely dominated by unsupported fall-boundary states rather than scalar reward alone.

Tasks:

1. Identify or collect fall-positive real rollouts.
2. Identify near-fall and recovery windows around low torso height, high tilt, unstable contact, or abnormal COM velocity.
3. Label or derive:
   - body height;
   - torso pitch/roll;
   - foot contacts;
   - COM velocity;
   - commanded velocity;
   - fall/termination/timeout;
   - `P(fall within K steps)` targets for several K values.
4. Train/calibrate a fall-risk head and support/OOD score.
5. Validate risk/support scores against real rollout failures and BC trajectories.
6. Use risk/support to:
   - terminate synthetic rollouts early;
   - penalize synthetic rewards;
   - downweight or reject generated transitions;
   - add conservative Q penalties near unsupported states.

Do not assume done-class balancing fixes the problem when QS shards contain few or no positive fall/done labels. If labels are missing, record that as a dataset limitation and use fall-positive collection or support proxies.

## Phase E: Image-Based Tasks, NEWT, and LeWorldModel

Image-based work can start once prior experiments show a reasonable signal or the official setup can be reproduced cheaply in parallel. The goal is not to replace the state-based MJLab work; it is to prepare a second modality track that tests whether Flow/SIGReg and pessimistic rollout ideas transfer beyond state observations.

Useful start signals:

```text
Flow-PWM beats a PWM baseline under matched real eval;
Flow-PWM shows higher imagined return plus clear exploitation diagnostics;
faithful PWM collapses on MJLab while original DFlex parity holds;
fall/support/OOD diagnostics show missing observation or representation signals;
official NEWT or LeWM setup can be reproduced without blocking state-based jobs.
```

Early image work may include repo setup, environment reproduction, import/config smokes, official baseline reproduction, and eval/render smokes. Image-based performance claims still require matched official baselines, final/best checkpoints, real eval, videos, and documented data/compute budgets.

### Phase E1: NEWT Reproduction And Image Benchmark Setup

Goal: build a concrete image-based benchmark path for testing Flow/SIGReg beyond state observations.

Setup tasks:

1. Clone NEWT into an external or vendor path, not mixed into core source until integration policy is explicit.
2. Record:
   - repo URL;
   - commit SHA;
   - environment creation commands;
   - task list;
   - dataset/source requirements;
   - official training commands;
   - official eval/render commands;
   - expected metrics from paper/docs.
3. Run import and config-list smokes.
4. Run the smallest official NEWT train/eval smoke with W&B disabled.
5. Run an official eval/render smoke and save at least one video artifact if supported.
6. Record all failures before modifying code.

Official reproduction rows:

```text
N0: official NEWT shortest-task smoke, W&B disabled
N1: official NEWT baseline one formal seed, W&B enabled
N2: eval-only reload for final and best checkpoints
N3: official render/video job
```

Our adapter rows:

```text
N4: image encoder + current state-based Flow endpoint WM
N5: image encoder + Flow residual/chunk WM
N6: N5 + SIGReg-style latent regularization
N7: N6 + short-horizon pessimistic MBPO
N8: non-Flow latent WM ablation
```

Keep fixed for comparisons:

```text
task
image resolution
frame stack
action repeat
encoder backbone
latent dimension
dataset or environment budget
synthetic horizon
policy update budget
eval episode count
video protocol
seed set
```

Required logs:

```text
pixel reconstruction or embedding prediction loss
latent variance/isotropy proxy
one-step and H-step latent prediction
reward calibration
done/success/fall calibration when available
imagined return
real return
episode length
success or fall rate
final and best checkpoint paths
MP4/W&B video links
```

Claim rule: compare NEWT rows only when the official NEWT baseline and our row use the same task, data budget, eval protocol, and video protocol. If the setup differs, frame the result as feasibility or diagnostic evidence.

### Phase E2: LeWorldModel / LeWM Reproduction And SIGReg Transfer

Goal: make LeWM a real comparison target and a source for SIGReg implementation details, not just a citation.

Setup tasks:

1. Clone LeWM into an external or vendor path.
2. Record:
   - repo URL;
   - commit SHA;
   - environment creation commands;
   - supported tasks;
   - pretrained assets or datasets;
   - official train commands;
   - official eval/render commands;
   - expected metrics from paper/docs.
3. Run import and config smokes.
4. Run the smallest official LeWM smoke with W&B disabled.
5. Run one official formal seed with W&B when the smoke is stable.
6. Run eval-only reload for final and best checkpoints.
7. Run video/render jobs where supported.

SIGReg extraction and local transfer:

```text
L0: read and document the exact LeWM SIGReg objective and tensor shapes
L1: implement minimal SIGReg-style loss for state latents
L2: test finite loss and gradients
L3: test zero-weight no-op behavior
L4: test constant-latent anti-collapse behavior
L5: log SIGReg loss, latent variance, and isotropy proxy
L6: port SIGReg to image latents
```

Comparison rows:

```text
L7: official LeWM baseline
L8: our image Flow WM
L9: our image Flow WM + SIGReg
L10: L9 + pessimistic short-horizon rollout
L11: no-SIGReg ablation
L12: no-Flow ablation
```

Comparison requirements:

```text
same task or documented task mismatch
same observation modality
same data budget where possible
same eval episodes
same video protocol
seed count once beyond smoke
W&B links
final/best checkpoint paths
failure notes
```

Do not present superiority over LeWM unless the official baseline is run under comparable data, compute, and evaluation. If exact parity is not feasible, document the mismatch and treat the result as a methodological comparison.

### Phase E3: Image-Based Decision Rules

```text
NEWT/LeWM official smoke fails:
  document the blocker and keep state-based work moving.

Official baseline runs but our image adapter fails:
  debug modality/encoder/integration before changing the policy algorithm.

Our Flow/SIGReg image row improves latent diagnostics but not real eval:
  treat SIGReg as representation help, not policy-improvement evidence.

Pessimistic image Flow-MBPO improves real eval/video:
  move to multi-seed and ablate Flow, SIGReg, support gate, fall/risk gate, and conservative Q.
```

## Deprioritized Directions

Avoid spending major resources on these until a specific diagnostic asks for them:

```text
large BC sweeps when the BC reference is already valid
large unstructured Flow architecture sweeps
ranking world models by MSE alone
unrestricted long-horizon Flow differentiable policy optimization
changing WM, policy, update, horizon, and eval protocol at the same time
MJLab claims without final/best real eval and MP4/W&B videos
image-based performance claims without official NEWT/LeWM reproduction and matched eval
```

These items can still be used as small exploratory jobs when inputs are ready, but write them as exploratory in W&B notes and docs.

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

Run the Agent Preflight And History Inventory first. After the preflight inventory is committed, submit aggressively when the plan has several useful independent experiments. Do not block broad experiment submission behind Slurm dependencies unless there is a hard data artifact requirement that makes the downstream job impossible to start. If a submitted job later proves invalid because of a config, environment, dataset, checkpoint, or wrapper mistake, cancel it and record the failure reason, affected job IDs, and replacement job IDs.

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

## Compact Codex Goal Prompt

Use this compact prompt when the Codex `/goal` input needs to stay below 4000 characters.

```text
/goal Follow docs/goals/pwm_flow_sigreg_image_research_plan_20260602.md and keep docs/git/W&B notes in English. Preflight first: before new sbatch submissions, inspect git branch/SHA/status/recent commits and search docs/git, docs/goals, scripts, configs, slurm logs, W&B/offline dirs, checkpoints/videos for prior GPU submissions. Query Slurm with squeue/sacct/seff when available, especially jobs 9387942, 9387949, 9387896, 9387895. Write and commit an English inventory table with job id, purpose, status, command/script, git SHA, config, env/dataset/version, seed, GPU/QOS, W&B link/dir, checkpoint/eval/video paths, return/length/fall, failure reason, usability, next action. Then record gate results and, for any MJLab formal checkpoint, submit final/best 40-episode real eval plus final/best 10-episode 1000-step MP4/W&B videos. Before new submissions, list candidate jobs as smoke/diagnostic/eval/formal/exploratory with inputs, W&B mode, artifacts, GPU/QOS, and whether dependency is required. Submit useful jobs with existing inputs; avoid dependencies unless artifacts are missing; prefer H200,H100,A100,L40S; use embers; W&B off for new-code smokes, W&B on for formal runs. If wrong, scancel, record root cause, affected/replacement IDs, and commit docs. Build matched evidence table: faithful PWM, prior runner, Flow WM only, Flow policy only, Flow WM+policy, best reproduction, BC, expert. Treat imagined-only gains as unverified. Run controlled R0-R4 one-variable A/B with fixed protocol; log WM/prediction/calibration/grad/action/OOD/real eval/video metrics. Add LeWM-inspired SIGReg only after documenting objective/shapes and tests: finite grad, zero-weight no-op, constant-latent anti-collapse, latent variance/isotropy. If exploitation/fall/OOD appears, submit pessimistic short-horizon Flow-MBPO H=1/3/5 AWR/AWAC plus support/fall/OOD diagnostics. Start NEWT/LeWM only after reasonable signal or cheap official smoke. Commit every meaningful scheduling/result/failure/config/script/doc update.
```
