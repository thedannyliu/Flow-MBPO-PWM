# MJLab QS Debug Tree

## Collector Baseline Fails

If expert collector return/length falls below the current reference:

1. Check task ID resolution and reject silent fallback.
2. Check checkpoint path, method, and runner load config.
3. Check action clipping, action scale, command input, observation groups, last action, and normalization.
4. Check environment version, reward terms, termination conditions, and episode length.
5. Stop policy-improvement work until the collector baseline is restored.

## Expert-Filtered BC Fails

Current 40-episode expert+noisy uniform BC eval is return `45.7831`, length `589.43`, fall `0.667`; expert-only BC is weaker at return `30.6292`, length `412.38`, fall `0.800`. If a new BC run cannot beat this:

1. Check that `bc_quality_filter=expert,expert_noisy` selects the expected windows.
2. Compare train action MSE, action norm, and rollout action norm.
3. Verify observation split and command normalization match the real MJLab actor observations.
4. Check command-conditioned failures. The initial-condition diagnostic found constant start obs norm, but episode length had moderately negative correlation with yaw command and first-action L2.
5. Prefer data/BC changes that improve yaw and recovery coverage before more smoothness-only runs; expert-only data did not improve robustness.
6. Do not mix medium data uniformly into BC as a default. The expert+noisy+medium uniform run lowered return and increased fall rate. A train-window audit found medium windows are not terminal-adjacent, but have higher action norm than expert windows, so filter or downweight high-action-norm medium windows before reuse.
7. Do not treat action-norm-filtered or loss-weighted medium as sufficient. Filtering medium windows at action norm `0.39` and downweighting medium BC loss to `0.25` both underperformed expert+noisy uniform BC.
8. Avoid medium as a plain BC warmstart target in the current pipeline; use it only with a different objective or after a stronger data-selection diagnostic.
9. Do not launch more PWM sweeps until BC is credible.

## World Model Looks Good But Rollout Fails

If one-step MSE or imagined return improves but real rollout fails:

1. Treat it as model exploitation.
2. Run long-horizon open-loop diagnostics at 5/10/25/50 steps.
3. Check reward and termination calibration.
4. Roll out frozen collector policies inside the world model and verify ranking: expert > medium > random.
5. Add stronger BC/KL/action regularization, shorter horizons, dataset-state starts, and uncertainty penalties.

## Best Actor Is Not Better Than Final

If true-best imagined-return rollout is worse than final:

1. Keep both artifacts and record both results.
2. Do not use imagined return for claims.
3. Prefer real-eval-based early stopping or a BC-preservation gate before expanding seeds.

## SigReg Helps MSE Only

If SigReg improves latent statistics or MSE but not long-horizon behavior or real rollout:

1. Record the result as diagnostic only.
2. Tune regularization strength only after long-horizon diagnostics are in place.
3. Do not move the claim to image tasks until the state-based protocol is trustworthy.
