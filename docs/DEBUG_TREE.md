# MJLab QS Debug Tree

## Collector Baseline Fails

If expert collector return/length falls below the current reference:

1. Check task ID resolution and reject silent fallback.
2. Check checkpoint path, method, and runner load config.
3. Check action clipping, action scale, command input, observation groups, last action, and normalization.
4. Check environment version, reward terms, termination conditions, and episode length.
5. Stop policy-improvement work until the collector baseline is restored.

## Expert-Filtered BC Fails

Current expert-filtered BC rollout is return `19.0827`, length `238.22`, fall `0.333`. If a new BC run cannot beat this:

1. Check that `bc_quality_filter=expert,expert_noisy` selects the expected windows.
2. Compare train action MSE, action norm, and rollout action norm.
3. Verify observation split and command normalization match the real MJLab actor observations.
4. Test longer BC, larger policy capacity, balanced expert/expert_noisy sampling, and action smoothness.
5. Do not launch more PWM sweeps until BC is credible.

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
