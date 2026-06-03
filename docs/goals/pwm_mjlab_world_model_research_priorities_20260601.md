# PWM/MJLab World-Model Research Priorities - 2026-06-01

This note records the parts of the long-term plan that are worth trying once the current PWM/Flow/MJLab experiments show a reasonable signal. It is intentionally selective: the goal is to avoid unfocused sweeps while still allowing useful jobs to run in parallel.

## Priority 1: Faithful PWM As The MJLab Baseline

This is the next high-value milestone after original PWM parity and minimal fidelity probes.

Run:

```text
Faithful PWM on MJLab
= original PWM architecture/update
+ MJLab QS dataset/env adapter
+ original-style latent WM, SimNorm, H=16, 3 critics, TD(lambda), PWM actor/critic settings
```

Questions:

- Does faithful PWM at least approach best BC?
- Does faithful PWM beat the current PWM-style runner?
- Does failure look like gradual degradation or immediate fall collapse?

Claim gate:

- final actor and true-best actor;
- 40-episode real MJLab eval;
- 10-episode, 1000-step MP4/W&B videos before any performance claim;
- return, episode length, and fall rate compared with expert, expert-noisy, medium, random/reference, and best BC.

Known MJLab reference points:

- Expert collector: return `82.6090`, length `1000`, fall `0`.
- Best aggregate BC: return about `45.8491`, length about `594.97`, fall about `0.625`.

## Priority 2: Control-Relevant World-Model Diagnostics

Do not rank world models by MSE alone. PWM's useful insight is that a world model can be more valuable for policy optimization when it is smooth and regularized, even if it is not the lowest-error predictor.

Track:

- one-step and H-step prediction;
- reward calibration;
- fall/done calibration;
- expert action through WM vs logged segment;
- gradient smoothness;
- action drift during policy optimization;
- action saturation;
- support/OOD distance;
- real MJLab return, length, fall rate, and videos.

This diagnostic layer should be attached to faithful PWM and any Flow replacement before expanding architecture sweeps.

## Priority 3: Fall / Support / OOD Boundary Modeling

This is likely more important than another small BC sweep or a large Flow architecture grid. The main MJLab failure mode is expected to be fall risk and model exploitation near unsupported states, not simply low imagined return.

Worth trying:

- collect or identify fall-positive rollouts;
- collect near-fall/recovery windows;
- label body height, torso tilt, contacts, COM velocity, and termination/fall proxies;
- train/calibrate `P(fall within K steps)`;
- score support/OOD distance against real fall episodes;
- terminate synthetic rollouts early on high fall/support risk;
- penalize reward or Q on high-risk/OOD generated transitions.

Avoid assuming that class-balanced done loss will solve this if the QS shards contain almost no positive done/fall labels.

## Priority 4: Pessimistic Short-Horizon Flow-MBPO

Flow is worth trying as a distributional short-horizon trajectory generator, not as an unconstrained long-horizon differentiable simulator.

Promising direction:

```text
Flow residual/trajectory/chunk ensemble
+ short rollouts from real states, H=1/3/5
+ uncertainty/support/fall early termination
+ conservative reward or Q penalty
+ BC-warmstarted AWR/AWAC first, conservative Q second
+ strict real eval/video/fall-rate gate
```

Interpretation:

- If Flow improves imagined return but worsens real rollout/fall, treat it as exploitation.
- If support/fall pessimism is required to reduce fall, the contribution is not "Flow alone"; it is safe use of a learned trajectory model.

## Priority 5: Minimal Flow Replacement Study

Start these rows when the faithful PWM adapter has a usable smoke/formal result or when Flow-PWM already shows a matched-protocol signal that needs causal isolation:

```text
Row 0: original PWM WM + original PWM policy/update
Row 1: Flow WM + original PWM policy/update
Row 2: original WM + Flow policy only if policy architecture is suspected
Row 3: Flow WM + Flow policy last
```

Keep horizon, actor/critic update, eval protocol, and training budget fixed. Row 1 is the key scientific comparison. Row 3 can run as an exploratory job in parallel, but causal interpretation still comes from the one-variable rows.

## Deprioritized For Now

- Large BC sweeps, unless the BC reference itself is found to be invalid.
- Large WM architecture matrices before the faithful PWM baseline is clean.
- Flow as an unrestricted differentiable simulator.
- Image-based / NEWT / LeWorldModel performance claims without official reproduction, matched eval, final/best checkpoints, and videos. Setup, smokes, and baseline reproduction can start when current experiments show a reasonable signal or when the official setup is cheap to reproduce.

## Current Task Submission Policy

Submit useful tasks broadly when inputs already exist. Do not add Slurm dependencies for sequencing aesthetics.

Current priorities can run in parallel:

```text
Phase 1 original parity/eval/Ant sanity
Phase 2 minimal diagnostics
Phase 3 faithful PWM MJLab smoke/formal
Flow one-variable A/B rows
SIGReg smokes and no-op tests
pessimistic short-horizon Flow-MBPO smokes
NEWT and LeWM official setup/reproduction smokes
```

Use dependencies only when a downstream command needs an artifact that does not exist yet. If a job is submitted too early and proves invalid, cancel it, record the reason, and submit a replacement. Final interpretation must still respect evidence quality: smoke output is diagnostic, while real MJLab claims require final/best real eval, videos, return, episode length, fall rate, and baseline comparisons.
