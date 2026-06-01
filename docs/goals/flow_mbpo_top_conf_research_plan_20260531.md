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
33. Done for first conservative-Q critic integration:
    `run_flow_mbpo_v0_awr_update.py` now has an opt-in deterministic Q critic
    path controlled by `--conservative-q-weight` and `--critic-actor-weight`.
    The critic trains on mixed real/synthetic one-step transitions with Bellman
    loss plus a CQL-style `softplus(Q(actor_action) - Q(data_action))` penalty.
    When requested, the actor receives a small `-Q(actor)` loss from the frozen
    critic for that actor step. The script saves `final_q_critic.pt` when the
    critic is enabled.
34. Done for conservative-Q smokes. CPU fake data validated finite critic
    metrics. W&B-disabled Slurm job `9356778` ran `20` iterations on the
    support-aware generated H3 replay with `--conservative-q-weight 1.0` and
    `--critic-actor-weight 0.01`. It completed and wrote actor checkpoints.
    Final critic metrics were critic loss `0.76992`, Bellman loss `0.07677`,
    CQL loss `0.69315`, CQL gap mean `1.38e-5`, `Q(data)` mean `0.09484`, and
    `Q(actor)` mean `0.09485`. Job `9356793` confirmed `final_q_critic.pt` is
    written and loadable.
35. Next method step: tune conservative-Q only under W&B-disabled smoke until
    it shows a meaningful conservative gap or stable actor effect. The first
    20-iteration smoke proves mechanics, not usefulness; the near-zero CQL gap
    suggests the critic has not yet learned a useful distinction between data
    and actor actions. Do not launch a formal seed from this configuration.
36. Done for random-action conservative-Q tuning infrastructure:
    `run_flow_mbpo_v0_awr_update.py` now supports `--critic-random-actions` and
    `--critic-cql-temperature`. With random actions enabled, the CQL term samples
    uniform actions in `[-1, 1]`, evaluates actor plus random actions, and uses a
    temperature-scaled logsumexp OOD value against `Q(data)`. The default
    `--critic-random-actions 0` preserves the first actor-only CQL smoke.
37. Done for the first random-action conservative-Q smoke. CPU fake data with
    `critic_random_actions=4` produced finite metrics. W&B-disabled Slurm job
    `9356862` ran `20` iterations on the support-aware generated H3 replay with
    `--conservative-q-weight 1.0`, `--critic-random-actions 10`,
    `--critic-cql-temperature 1.0`, and `--critic-actor-weight 0.01`. It wrote
    actor checkpoints and `final_q_critic.pt`. Final critic metrics were loss
    `2.41859`, Bellman loss `0.07703`, CQL loss `2.34156`, CQL gap mean
    `2.34156`, `Q(data)` mean `0.10054`, `Q(actor)` mean `0.10057`,
    `Q(random)` mean `0.03787`, and `Q(random)` max `0.19931`. This fixes the
    near-zero CQL-gap diagnostic from actor-only CQL, but it remains smoke-only
    evidence. Next tune training duration, CQL weight, and random-action mix
    before any formal W&B seed.
38. Done for longer random-action conservative-Q smoke. W&B-disabled Slurm job
    `9357006` ran the same support-aware generated H3 replay and random-action
    CQL settings for `100` iterations. It completed on `embers`, wrote actor
    checkpoints and a loadable `final_q_critic.pt`. Final metrics were critic
    loss `1.9391`, Bellman loss `0.08688`, CQL gap `1.85219`, `Q(data)` mean
    `0.25246`, `Q(actor)` mean `0.26809`, `Q(random)` mean `-0.42555`, and
    `Q(random)` max `0.73695`. Across logged iterations, the CQL gap decreased
    from `2.3921` to `1.8522` but did not collapse.
39. Next method step: random-action CQL now has a persistent conservative
    training signal, but the rising `Q(random)` max and increasing actor/data Q
    values mean it is still only a critic diagnostic. Tune CQL
    weight/temperature/random-action count, and consider lowering
    `critic_actor_weight`, before running any formal W&B seed.
40. Done for actor-weight ablation. W&B-disabled Slurm job `9357054` repeated
    the `100`-iteration random-action CQL smoke with `--critic-actor-weight 0.0`
    and otherwise identical settings to job `9357006`. It completed on
    `embers`, wrote actor checkpoints, and saved a loadable `final_q_critic.pt`.
    Final metrics were critic loss `1.9366`, Bellman loss `0.08701`, CQL gap
    `1.84959`, `Q(data)` mean `0.25059`, `Q(actor)` mean `0.25223`,
    `Q(random)` mean `-0.42797`, and `Q(random)` max `0.73796`.
41. Next method step: the actor-weight ablation shows the high random-action
    max is almost unchanged when actor-Q pressure is removed. Use
    `critic_actor_weight=0.0` as the safer critic-tuning default, and tune CQL
    weight/temperature/random-action sampling before reintroducing actor-Q loss
    or launching a formal W&B seed.
42. Done for CQL-weight stress smoke. W&B-disabled Slurm job `9357126` repeated
    the critic-only 100-iteration random-action CQL smoke with
    `--conservative-q-weight 5.0` instead of `1.0`. It completed on `embers` and
    wrote actor checkpoints plus `final_q_critic.pt`. Final metrics were critic
    loss `8.7681`, Bellman loss `0.12316`, CQL gap `1.72899`, `Q(data)` mean
    `0.39997`, `Q(actor)` mean `0.40181`, `Q(random)` mean `-0.45006`, and
    `Q(random)` max `1.02192`.
43. Next method step: increasing CQL weight from `1.0` to `5.0` slightly lowered
    the mean CQL gap and average random-action value, but it worsened Q scale,
    Bellman loss, and the high-valued random-action tail. Do not formalize this
    setting. Keep `critic_actor_weight=0.0`, return CQL weight to `1.0`, and
    tune temperature or sampled-action coverage next.
44. Done for low-temperature CQL smoke. W&B-disabled Slurm job `9357174` repeated
    the critic-only random-action smoke with `--critic-cql-temperature 0.1`,
    CQL weight `1.0`, random actions `10`, and actor critic weight `0.0`. It
    completed on `embers` and wrote all expected checkpoints. Final metrics were
    critic loss `0.14584`, Bellman loss `0.08007`, CQL gap `0.06577`,
    `Q(data)` mean `0.18269`, `Q(actor)` mean `0.18361`, `Q(random)` mean
    `-0.19735`, and `Q(random)` max `0.44153`.
45. Next method step: temperature `0.1` controls the high random-action tail and
    Q scale better than temperature `1.0`, but it likely makes the conservative
    gap too weak. Do not formalize it. Test an intermediate temperature such as
    `0.5` before reintroducing actor-Q pressure or launching a formal W&B seed.
46. Done for intermediate-temperature CQL smoke. W&B-disabled Slurm job
    `9357227` repeated the critic-only random-action smoke with
    `--critic-cql-temperature 0.5`, CQL weight `1.0`, random actions `10`, and
    actor critic weight `0.0`. It completed on `embers` and wrote all expected
    checkpoints. Final metrics were critic loss `0.82317`, Bellman loss
    `0.08494`, CQL gap `0.73823`, `Q(data)` mean `0.23597`, `Q(actor)` mean
    `0.23751`, `Q(random)` mean `-0.39914`, and `Q(random)` max `0.66968`.
47. Next method step: temperature `0.5` is the best compromise among the three
    short temperature smokes: more conservative than `0.1`, with lower tail than
    `1.0`. It is still smoke-only evidence. Continue with either a longer
    W&B-disabled temp-`0.5` run or a small real-eval plumbing check before any
    formal W&B seed.
48. Done for temp-`0.5` real-eval plumbing check. W&B-disabled Slurm job
    `9357292` reran the temp-`0.5`, CQL weight `1.0`, random-action `10`,
    actor-weight `0.0` setting with `real_eval_every=100` and
    `real_eval_episodes=8`. It completed on `embers`, wrote final,
    best-real-eval, best-training, real-eval snapshot, and critic checkpoints.
    The best checkpoint is correctly marked `checkpoint_kind=best_real_eval` and
    `is_true_best_snapshot=True`.
49. Result: the real-eval plumbing works, but the candidate is not promising.
    Iter-100 8-episode eval was return `18.7727`, length `283.50`, fall
    `1.000`, timeout `0.000`, far below matched BC. Do not formalize this
    conservative-Q setting. Either keep conservative-Q as a diagnostic or change
    the update objective before spending W&B/video budget.
50. Done for a W&B-disabled roll10 MP4 diagnostic on that same temp-`0.5`
    conservative-Q actor. Slurm job `9370468` rendered the final checkpoint for
    `10` episodes at `max_steps=1000` and wrote
    `scripts/outputs/mjlab_qs/flow_mbpo_v1_rollouts/flow_trajectory_chunk_5k_h3_gen_support_riskterm_q90_cq1_rand10_temp0p5_actor0_iter100_eval8_s0/final_roll10/rollout.mp4`.
    The result was return `54.2864`, length `689.80`, fall `0.400`, with six
    timeouts and four short falls. The final, best-real, and iter-100 snapshot
    actor weights are identical, so this final render also covers the true-best
    actor for this one-eval smoke.
51. Interpretation: the roll10 diagnostic is much better than the 8-episode
    eval and essentially ties matched BC seed0 final roll10 (`54.1283`,
    `688.40`, fall `0.400`), but it does not lower fall rate and the scalar
    eval remains poor. This is variance/selection evidence, not policy
    improvement. Do not formalize this conservative-Q branch without changing
    the update objective or requiring a stronger 40-episode eval plus matched
    roll10 gate.
52. Done for gate-aware true-best selection infrastructure. The AWR update now
    has opt-in `--real-eval-selection-metric return_length_fall`, with score
    `return_mean + length_weight * episode_length_mean - fall_penalty *
    fall_rate_mean`; the default remains return-only to preserve old manifests.
    This makes best-real snapshot selection match the formal return/length/fall
    gate better than return-only selection.
53. Validation passed via `py_compile`, CLI help, a direct formula check, and
    W&B-disabled Slurm job `9370586` on `embers`. That integration smoke used
    one update iteration and two eval episodes only to verify plumbing; it wrote
    `best_policy_extraction.pt` as a true best-real snapshot and recorded
    `real_eval/selection_score` plus `real_eval/selection_metric` in both the
    best checkpoint and real-eval snapshot. This is infrastructure evidence, not
    policy evidence.
54. Done for real-eval early-stop infrastructure. The AWR update now has opt-in
    `--real-eval-stop-score-below`, `--real-eval-early-stop-patience`, and
    `--real-eval-min-delta`. After each real-eval snapshot it computes the same
    selection score used for best-real checkpoint selection and can stop the
    loop when the actor is below a configured score threshold or stops
    improving. Defaults leave existing runs unchanged.
55. Validation passed via `py_compile`, CLI help, and W&B-disabled Slurm job
    `9370641` on `embers`. That smoke intentionally set an artificial stop
    threshold `9999` with `update_iters=5` and `real_eval_every=1`, so it
    stopped after iter `1` and wrote `early_stop_iter=1` plus the stop reason in
    `summary.json`. This validates the control path only; it is not policy
    evidence.
