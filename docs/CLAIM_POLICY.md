# MJLab QS Claim Policy

## Evidence Levels

- Level 0: diagnostic only. Examples: world-model MSE, imagined return, BC action loss, latent statistics.
- Level 1: one real MJLab rollout/eval with MP4 or W&B video.
- Level 2: multiple real MJLab evals or seeds with videos and comparison to BC plus collector/reference baselines.
- Level 3: repeated improvement over BC/reference/collector with ablations and videos.

Do not call PWM, Flow, SigReg, or BC warm-start a policy-improvement result below Level 2.

## Required Evidence For Policy Claims

Every claimed policy result must include:

- real MJLab return;
- episode length;
- fall rate, using termination rather than time-limit truncation when available;
- rollout MP4 path;
- W&B video run;
- seed;
- git SHA and branch;
- full command;
- dataset path and version/stage;
- checkpoint path;
- comparison to expert, expert-noisy, medium, random/reference, and BC baselines.

## Interpretation Rules

- World-model MSE and imagined return are diagnostics only.
- True-best imagined-return actors must be evaluated separately from final actors.
- If imagined return improves while real rollout worsens, classify the result as model exploitation.
- A "less bad" collapsed policy is not locomotion improvement.
- Rollout videos are required because scalar returns can hide falls, resets, and unstable gaits.
