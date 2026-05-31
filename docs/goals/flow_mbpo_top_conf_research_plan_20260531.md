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
8. Done for logging: `render_policy_rollout.py --save-support-features` writes
   per-step normalized state, command, action, raw action, reward, and done
   flags.
9. Done for scoring infrastructure: `score_rollout_support_distance.py` scores
   logged real rollout features against the same expert+noisy normalized
   `(state, command, action)` nearest-neighbor support set used by the replay
   penalty. Fake-data validation passed, and Slurm jobs `9355461`/`9355480`
   scored the 50-step no-fall BC smoke. On that smoke, q50 threshold `0.201729`
   gave support-penalty mean `0.3135`; q90 threshold `0.622495` gave penalty
   mean `0.0468` and tail-10 penalty `0.0`. This confirms the scoring path and
   suggests q50 is too aggressive for uncalibrated real rollout states.
10. Next calibration step: render longer matched BC and Flow-MBPO rollouts with
    `--save-support-features`, score episodes containing both falls and
    timeouts, and test whether support distance or tail-window support distance
    separates failed from successful real segments. If it does not, prioritize
    conservative-Q pessimism over another support-penalty formal run.
11. Done for first real-failure calibration: W&B-disabled job `9355621`
    rerendered matched BC seed0 final and Flow trajectory/chunk lowsynth final
    as 10-episode, 1000-step rollouts with support features. Refresh job
    `9355785` regenerated q50/q90 scorer summaries after adding grouped
    tail-window stats. q90 support distance strongly separates terminated from
    timeout episodes in both policies. BC terminated episodes had support
    distance max mean `11.7936` and tail10 mean `6.0608`, versus timeout max
    mean `1.7309` and tail10 mean `0.8289`. Flow terminated episodes had max
    mean `12.8382` and tail10 mean `6.3886`, versus timeout max mean `1.4489`
    and tail10 mean `0.5694`.
12. Next method step: convert the calibrated support signal into a conservative
    support-risk objective. Prefer a risk gate/penalty based on high or late
    support-distance spikes rather than the raw q50 threshold, because q50
    penalized even a no-fall 50-step BC segment heavily. Run W&B-disabled AWR
    smoke first, then formal W&B only if the smoke remains mechanically clean
    and the objective is plausibly targeted at the observed fall signal.
13. Done for first support-risk objective smoke: `run_flow_mbpo_v0_awr_update.py`
    now supports `--support-action-penalty-weight`, which penalizes current
    actor actions whose normalized `(state, command, actor_action)` is outside
    an expert+noisy support set. Fake-data validation passed, then W&B-disabled
    Slurm job `9355897` ran the trajectory/chunk H3 recipe for `20` AWR
    iterations with q90 threshold `0.622495` and support-action weight `1.0`.
    Final support-action loss was `0.003879`, real support distance mean/p90
    was `0.2515`/`0.6042`, and synthetic mean/p90 was `0.1762`/`0.3746`.
    Checkpoints were written.
14. Next method step remains calibration, not formal expansion. The q90
    support-action penalty is mechanically clean but mild on update batches,
    while real falls show much larger late-rollout support spikes. Test a
    stronger support-risk variant or conservative-Q penalty before any formal
    W&B seed.
15. Done for support-action stress diagnostics: active-fraction logging was
    added to the AWR support metrics and validated. q90 weight `10.0` rerun in
    job `9356023` had final real/synthetic active fractions `0.0917`/`0.0`.
    q50 weight `1.0` rerun in the same job had active fractions `0.500`/`0.375`
    and support-action loss `0.08731`, but the final real/synthetic support
    distance summaries remained essentially unchanged from q90. This indicates
    the current AWR update batches do not contain the late high-support-distance
    fall distribution seen in real rollouts.
16. Next method step: do not formalize plain support-action regularization.
    Either add rollout-state/high-risk-state augmentation so the support-risk
    loss sees the failure distribution, or move to a conservative-Q objective
    that penalizes actor actions outside support more directly.
17. Done for high-risk rollout-state diagnostic: `run_flow_mbpo_v0_awr_update.py`
    now accepts scored rollout support tensors through `--support-risk-features`
    and can sample high-distance rollout states during AWR. Job `9356122`
    selected `133` rows with support distance at least `2.0` from the matched
    BC/Flow calibration rollouts. The risk loss saw those states
    (`support_risk_loss=4.5367`, risk active fraction `1.0`), but actor risk
    distance matched the source distance because the high distance is dominated
    by the state/command component.
18. Done for action-ablation diagnosis: scorer job `9356143` recomputed rollout
    support distance with `--action-weight 0.0`. Fall-vs-timeout separation was
    essentially unchanged: BC terminated tail10 `6.0567` versus timeout
    `0.8253`; Flow terminated `6.3846` versus timeout `0.5662`. The support
    signal is therefore primarily state/command OOD, not action OOD.
19. Next method step: do not spend more runs on actor-only support penalties.
    Use support distance as model-rollout pessimism or early termination over
    generated states, or implement conservative-Q on out-of-support generated
    states/actions.
20. Done for state/command replay pessimism smoke: job `9356199` reran support
    replay scoring with `--action-weight 0.0`. State-only q90 was still mild
    on the current H3 synthetic replay: threshold `0.620098`, support penalty
    mean `0.00548`, reward mean `-0.08050`. State-only q50 was strong:
    threshold `0.200275`, support penalty mean/p90/max
    `0.09559`/`0.32431`/`0.77090`, reward mean `-0.53106`. These are almost
    identical to the full-feature support replay numbers, confirming synthetic
    replay support distance is state-dominated.
21. Done for AWR smoke on state-only q50 replay: job `9356236` completed `20`
    W&B-disabled AWR iterations and wrote final/best/best-training checkpoints.
    Final loss was `0.001026`, synthetic reward mean was `-0.3223`, and
    synthetic done fraction was `0.0`.
22. Next method step: state/command replay reward pessimism is mechanically
    usable, but q50 is broad and q90 is mild. Do not formalize either as-is.
    Implement a targeted support-risk termination or conservative-Q objective
    that reacts when model-generated states cross the calibrated real-rollout
    risk boundary.
23. Done for first support-risk replay truncation utility:
    `apply_flow_mbpo_support_truncation.py` marks support-risk crossings in a
    scored synthetic replay and, by default, marks the first crossing and all
    later rows in the same rollout branch as done. It preserves the previous
    done mask and writes a new `synthetic_replay.pt` plus `summary.json`.
    `py_compile` and a fake two-branch replay validated post-risk done
    propagation and reward-penalty arithmetic.
24. Done for replay truncation smokes on the current state-only H3 support
    replays. q90 support-risk truncation used threshold `0.620098`, touched
    `28/256` branches, raised done fraction from `0.13021` to `0.17188`, and
    completed a 20-iteration W&B-disabled AWR smoke in job `9356396`. q50
    truncation used threshold `0.200275`, touched `123/256` branches, and
    raised done fraction to `0.48438`, which is likely too broad for a formal
    setting. Treat q90 truncation as a mechanically clean component, not a
    formal-run trigger.
25. Next method step: move support-risk truncation into rollout generation
    itself or combine it with conservative-Q. Post-hoc q90 truncation only
    affects a small fraction of the already generated H3 replay, while real
    failures show late support spikes after closed-loop drift. The next useful
    experiment should stop or penalize generated branches at the moment they
    cross the calibrated q90 support boundary, then run W&B-disabled AWR smoke
    before any formal W&B seed.
26. Done for moving support-risk termination into replay preparation:
    `prepare_flow_mbpo_v0_synthetic_replay.py` now has an opt-in
    `--support-risk-termination` path. It builds a real-data support set,
    calibrates a probe threshold, computes support distance for synthetic
    transitions, marks rows above threshold as `support_risk_done`, applies an
    optional support-risk reward penalty, and includes the support-risk done
    mask before post-first-done branch truncation.
27. Done for prepare-time support-risk smokes. Fake support data validated the
    new done mask, and a no-support check preserved the prior H3 replay done
    fraction `0.1302083`. Full q90 state/command support-risk preparation ran
    in Slurm job `9356522` with `20000` support rows, `4096` probe rows,
    threshold `0.620098`, action weight `0.0`, support-risk done fraction
    `0.04818`, and final done fraction `0.171875`. A 20-iteration
    W&B-disabled AWR smoke on that replay completed in job `9356566` and wrote
    final/best/best-training checkpoints.
28. Next method step remains support-aware generation or conservative-Q.
    Prepare-time support-risk termination is cleaner than a post-hoc artifact
    transform, but it still operates on an already generated fixed H3 buffer.
    To address the real failure mode, the rollout generator should stop or
    downweight branches as soon as generated states cross the calibrated q90
    support boundary, or the policy update should use conservative-Q over
    generated/out-of-support states. Do not formalize q90 prepare-time
    support-risk termination by itself.
29. Done for first support-aware closed-loop generation slice:
    `run_flow_mbpo_v0_smoke.py` now has opt-in
    `--support-risk-termination`. During model rollout generation it computes
    support distance for the current generated `(state, command, action)`,
    marks above-threshold rows as `support_risk_done`, marks the transition
    done, freezes that branch state for later horizon steps, and logs
    `rollout_active`, `support_risk_distance`, `support_risk_threshold`, and
    `support_risk_done`. Defaults preserve the old closed-loop generator
    behavior.
30. Done for support-aware generation smokes. A fake generator test confirmed
    no-support behavior still advances states normally and support-enabled
    behavior freezes branches after threshold crossing. Full trajectory/chunk
    H3 support-aware generation ran in job `9356635` with q90 threshold
    `0.620098` and action weight `0.0`. It produced support-risk done fraction
    `0.09635`, rollout active fraction `0.94010`, horizon done fractions
    `[0.08203, 0.09766, 0.10938]`, and support-risk distance mean/p90/max
    `0.23963`/`0.60024`/`0.97118`.
31. Done for downstream smoke on the support-aware generated replay. Replay
    preparation yielded final done fraction `0.16406` and conservative reward
    mean `-0.06376`. A 20-iteration W&B-disabled AWR smoke completed in job
    `9356654`, wrote all checkpoints, and ended with loss `0.001026` and
    synthetic reward mean `-0.04690`.
32. Next method step: this is the first support-risk component that affects
    closed-loop generation, but it is still smoke-only evidence. Do not run a
    formal seed from support-aware generation alone. Pair it with a stronger
    conservative update objective, preferably conservative-Q over
    out-of-support generated states/actions, or define a one-seed W&B formal
    plan with final/true-best eval and matched videos only if there is a
    specific reason it should improve fall rate versus matched BC.
