# Flow-MBPO v1 Pessimistic Design

Date: 2026-05-31

## Objective

Implement the smallest practical Pessimistic Flow-MBPO path:

```text
Flow/MLP ensemble WM
-> short synthetic rollouts from real states
-> uncertainty/fall-penalized replay
-> BC-warmstarted AWR/AWAC update
-> real MJLab eval and rollout videos
```

The v1 goal is to lower real video fall rate, not only to improve imagined
return, one-step MSE, or short scalar diagnostics.

## Model Interface

Step models should expose:

```python
predict_step(obs, command, action) -> {
    "next_obs": Tensor,
    "reward": Tensor,
    "done_prob": Tensor | None,
    "fall_prob": Tensor | None,
    "uncertainty": Tensor,
    "aux": dict,
}
```

Chunk models should expose:

```python
predict_chunk(obs, command, action_seq, horizon) -> {
    "obs_seq": Tensor,
    "reward_seq": Tensor,
    "done_prob_seq": Tensor | None,
    "fall_prob_seq": Tensor | None,
    "uncertainty_seq": Tensor,
    "aux": dict,
}
```

## Synthetic Replay Schema

Keep real and synthetic transitions separate. Synthetic rows should include:

```text
obs
command
action
reward_model
reward_conservative
next_obs
done_model
fall_prob
uncertainty
source_model
horizon_step
start_dataset_index
start_quality_label
policy_checkpoint
wm_checkpoint
```

## Conservative Reward

Start with:

```text
reward_conservative =
    reward_model
    - lambda_uncertainty * normalize(uncertainty)
    - lambda_fall * fall_prob
```

Smoke grid:

```text
lambda_uncertainty in {0.0, 0.5, 1.0}
lambda_fall in {0.0, 1.0, 5.0}
horizon in {1, 3, 5}
synthetic_real_ratio in {0.0625, 0.125, 0.25}
```

Do not expand this grid until one setting clears the real eval and video gate.

## Early Termination

Terminate synthetic rollouts on the first of:

```text
fall_prob > fall_threshold
done_prob > done_threshold
uncertainty > uncertainty_quantile_threshold
horizon reached
```

Initial defaults:

```text
fall_threshold = 0.5
done_threshold = 0.5
uncertainty_quantile_threshold = heldout q90 or q95
```

If the done/fall head predicts near-zero for all rollout starts, do not treat it
as a safety signal. First add class-balanced done/fall loss, near-fall
oversampling, or auxiliary height/contact targets.

Current 2026-05-31 label audit found that the existing QS raw shards have zero
positive `done`, `termination`, and `truncation` labels across random_smooth,
medium, expert, and expert_noisy. On this dataset, class balancing or a larger
done-loss weight alone cannot calibrate fall prediction because there are no
positive labels. Use fall-positive collection, near-fall/recovery data, a
state/support/OOD risk proxy, or conservative-Q pessimism before trusting
model-done termination.

## Policy Update

Start from the strongest expert+noisy uniform BC checkpoint.

Actor objective:

```text
weighted real-action BC
+ weighted synthetic/model-action regression
+ BC anchor
+ KL or action-deviation penalty to the BC/previous actor
```

The AWR update must not drift far from BC until real rollout is stable. If
explicit uncertainty/fall penalties do not reduce video fall rate, add a critic
and COMBO-style conservative Q regularization on model-generated or
out-of-support actions.

Implementation status: `run_flow_mbpo_v0_awr_update.py` supports
`--action-deviation-weight`. When enabled, it keeps a frozen copy of the
warm-start actor and penalizes current-policy action MSE to that reference on
real and active synthetic states. This is a deterministic-actor KL-like
safeguard, not a full distributional KL.

## Smoke Checklist

W&B disabled:

1. Load dataset, normalization, best BC, and one WM ensemble.
2. Generate a 256-start synthetic buffer at `H=1`.
3. Save reward, conservative reward, uncertainty, done/fall, action norm,
   next-state delta, and quality-label diagnostics.
4. Verify replay rows include `fall_prob`, `uncertainty`, and source metadata.
5. Run 10-20 AWR/AWAC update iterations.
6. Verify final and best checkpoints load.
7. Make no policy-improvement claim.

Current smoke status: the trajectory/chunk H3 AWR action-deviation smoke passed
mechanically in Slurm job `9354631` using commit `17a2545`. The artifact is
`scripts/outputs/mjlab_qs/flow_mbpo_v1_awr_smoke/flow_trajectory_chunk_5k_h3_r240_s16_anchor1_actdev10_iter20_s0/`.
It ran `20` AWR iterations with real batch `240`, synthetic batch `16`, BC
anchor `1.0`, and `--action-deviation-weight 10.0`; it saved final, best, and
best-training-loss checkpoints. This only validates the update path and does
not provide real-rollout evidence.

## Formal Checklist

W&B enabled, `embers` QOS:

1. Run one seed only.
2. Log git SHA, command, dataset/version, seed, checkpoint paths, and notes.
3. Save final actor and true-best actor.
4. Run 40-episode real MJLab eval for both.
5. Render 10-episode 1000-step MP4/W&B videos for both.
6. Compare against matched BC seed0 and aggregate BC.
7. Expand only if return/length preserve or exceed BC and fall rate is lower
   than BC in both scalar eval and video evidence.

Current formal status: the trajectory/chunk H3 `r240/s16` action-deviation
variant ran as one formal seed in Slurm jobs `9354764` and `9354806` with W&B
run IDs `dy7hzh0r`, `fy3zyqka`, `39tx14vg`, `30xx57uc`, and `zxjfbwc7`.
Final 40-episode eval was return `43.9079`, length `577.45`, fall `0.725`;
best-real 40-episode eval was return `41.4285`, length `544.575`, fall
`0.750`. Final/best-real roll10 videos had fall `0.400`, tying matched BC
rather than improving it. Do not expand this variant.

Current support/OOD status: `add_flow_mbpo_support_penalty.py` can augment a
synthetic replay with normalized real-data nearest-neighbor distance over
`(state, command, action)` and subtract a calibrated support penalty from
`reward_conservative`. The first W&B-disabled smoke used `20000` expert+noisy
support rows, `4096` disjoint probe rows, q90 threshold `0.622495`, and
`lambda_support=5.0`. The current H3 replay had support-distance p90 `0.5258`,
so the penalty was mild; support penalty mean was `0.00544` and p90 was `0.0`.
This path is mechanically clean but still needs calibration against real
failures before a formal run.

The stronger q50 support-threshold smoke used the same support/probe split with
threshold `0.201729` and `lambda_support=5.0`. It raised support penalty mean to
`0.09565`, p90 to `0.32410`, and reduced conservative reward mean from
`-0.053114` to `-0.531373`; a 20-iteration AWR smoke still completed cleanly.
Existing rollout-step logs only contain reward/action norm/done fields, not
full state/action vectors, so support distance cannot yet be calibrated against
real fall events from saved videos.

Rollout support-feature logging is now available through
`render_policy_rollout.py --save-support-features`. It saves
`rollout_support_features.pt` with normalized state, normalized command,
applied/raw actions, reward, and done/terminated/truncated flags for each
rendered step. A W&B-disabled BC seed0 smoke wrote `50` rows with shapes
`state=(50,96)`, `command=(50,3)`, `action=(50,29)`, and
`raw_action=(50,29)`. Use this to calibrate support distance against real
failures before treating support penalty as a claim-worthy safety signal.

`score_rollout_support_distance.py` now performs that scoring step for saved
real rollout features. It reuses the support-set construction and normalized
feature definition from `add_flow_mbpo_support_penalty.py`, then writes per-step
CSV, per-episode CSV, scored tensors, and a JSON summary. It passed
`py_compile`, a fake-data terminated/truncated episode check, and two Slurm
smokes on the 50-step no-fall BC rollout support artifact. With the same
`20000` support rows and `4096` probe rows, q50 threshold `0.201729` produced
rollout support-penalty mean `0.3135`, while q90 threshold `0.622495` produced
mean `0.0468` and tail-10 penalty `0.0`. Treat this as calibration
infrastructure only: the first scored rollout has no fall or timeout, so the
next required evidence is full matched BC/Flow rollouts with support features
covering both failed and successful episodes.

The first matched support-feature calibration is now available. Job `9355621`
rendered BC seed0 final and Flow trajectory/chunk lowsynth final for 10
episodes at `max_steps=1000` with W&B disabled and support features enabled;
job `9355785` refreshed q50/q90 scores after adding grouped tail-window stats.
BC rerendered at return `48.2874`, length `635.20`, fall `0.500`; Flow
rerendered at return `54.5913`, length `694.00`, fall `0.400`. Under the q90
real-probe threshold `0.622495`, terminated episodes show large late support
spikes while timeout episodes remain much closer to support. For BC, terminated
episodes had support-distance max mean `11.7936` and tail10 mean `6.0608`,
versus timeout max mean `1.7309` and tail10 mean `0.8289`. For Flow, terminated
episodes had max mean `12.8382` and tail10 mean `6.3886`, versus timeout max
mean `1.4489` and tail10 mean `0.5694`.

Design implication: support distance is now a plausible real-fall risk proxy,
but raw q50 support gating is too aggressive. The next replay/policy update
should use a q90-style support-risk penalty or late-spike gate calibrated from
real rollout features, then clear W&B-disabled AWR smoke before any formal W&B
seed.

`run_flow_mbpo_v0_awr_update.py` now has an opt-in support-action penalty for
that first policy-side risk test. When `--support-action-penalty-weight > 0`,
it builds an expert+noisy support set over normalized
`(state, command, actor_action)`, calibrates the threshold from a disjoint real
probe set, and penalizes current actor actions outside support on real and
active synthetic batches. The default weight remains `0.0`.

The first W&B-disabled support-action smoke ran in Slurm job `9355897` with the
trajectory/chunk H3 replay, `20` AWR iterations, q90 threshold `0.622495`, and
support-action weight `1.0`. It completed mechanically and wrote final, best,
and best-training-loss checkpoints. Final support-action loss was `0.003879`;
real support distance mean/p90 was `0.2515`/`0.6042`; synthetic support
distance mean/p90 was `0.1762`/`0.3746`. This confirms the objective can run,
but it is likely too mild because update-batch distances mostly remain below
the q90 threshold. Use this as infrastructure, not a formal-run trigger.

Stress diagnostics confirm that conclusion. Active-fraction and max-distance
logging was added to the AWR support metrics. With q90 threshold `0.622495` and
weight `10.0`, the final real active fraction was only `0.0917` and synthetic
active fraction was `0.0`. With q50 threshold `0.201729` and weight `1.0`, the
final real/synthetic active fractions rose to `0.500`/`0.375` and support loss
rose to `0.08731`, but final support-distance summaries were still essentially
unchanged in a 20-iteration smoke. The current AWR batches therefore do not
expose the high-risk late rollout states where falls occur. Prefer
rollout-state/high-risk-state augmentation or a conservative-Q penalty before a
formal support-risk run.

High-risk rollout-state augmentation has now been tested as a diagnostic. AWR
can load scored rollout support tensors via `--support-risk-features`, select
rows with high recorded support distance, and apply the support penalty to the
current actor on those states. Job `9356122` selected `133` high-risk rows
(`support_distance >= 2.0`) from the matched BC/Flow calibration rollouts. The
risk loss was active (`support_risk_loss=4.5367`, active fraction `1.0`), but
the actor risk support distance stayed equal to the source distance. A follow-up
scorer run with `--action-weight 0.0` showed the same fall-vs-timeout
separation as the full feature distance. The support-fall signal is therefore
mostly state/command OOD. Actor-only support penalties cannot move an already
out-of-support state back into support.

Design implication: support distance should move into model-rollout pessimism,
early termination, or a conservative-Q objective over generated/rollout states.
Do not run a formal seed from the actor-only support-penalty path.

State/command-only replay reward pessimism is also mechanically available. Job
`9356199` reran `add_flow_mbpo_support_penalty.py` on the trajectory/chunk H3
synthetic replay with `--action-weight 0.0`. The q90 state support penalty was
mild, with threshold `0.620098`, penalty mean `0.00548`, and reward mean
`-0.08050`. The q50 state support penalty was much stronger, with threshold
`0.200275`, penalty mean/p90/max `0.09559`/`0.32431`/`0.77090`, and reward mean
`-0.53106`. These values are nearly identical to the full-feature support
replay, again showing that support risk is state-dominated. A 20-iteration
W&B-disabled AWR smoke on the q50 state replay completed in job `9356236` and
wrote all checkpoints. Treat this as a usable replay-pessimism path, but not as
a formal setting: q90 is too mild on current synthetic replay, while q50 is too
broad relative to stable real rollout segments.

Support-risk replay truncation is now available as the first early-termination
component for scored synthetic replays. `apply_flow_mbpo_support_truncation.py`
takes a replay with `support_distance` and `support_threshold`, marks rows that
cross the support threshold, and can mark all later rows in the same
`start_index`/`horizon_step` branch as done. It preserves the previous done mask
for auditability and writes a new `synthetic_replay.pt` plus `summary.json`.

On the current state/command-only H3 replay, q90 truncation is targeted but
mild: threshold `0.620098`, crossing fraction `0.04818`, post-risk truncation
fraction `0.09635`, risk branch fraction `0.10938`, and done fraction rising
from `0.13021` to `0.17188`. A 20-iteration W&B-disabled AWR smoke on this
q90-truncated replay completed in Slurm job `9356396`, wrote final, best, and
best-training checkpoints, and ended with loss `0.001026` and replay synthetic
done fraction `0.171875`.

q50 truncation is much broader: threshold `0.200275`, crossing fraction
`0.45573`, post-risk truncation fraction `0.45964`, risk branch fraction
`0.48047`, and done fraction `0.48438`. This is useful as a stress diagnostic
but too broad to formalize. Design implication: q90 support-risk truncation is
mechanically clean, but post-hoc truncation of an existing H3 replay is not yet
a claim-worthy method. The next implementation should terminate or penalize
branches during model rollout generation when generated states cross the
calibrated q90 support boundary, or move to conservative-Q over out-of-support
generated states/actions.

The same support-risk signal is now integrated into the normal replay
preparation entry point. `prepare_flow_mbpo_v0_synthetic_replay.py
--support-risk-termination` builds the real-data support/probe split, computes
synthetic support distance, marks rows above threshold as `support_risk_done`,
optionally subtracts a support-risk reward penalty, and includes support-risk
done before post-first-done branch truncation. Defaults keep the old behavior
off.

Validation covered three levels: `py_compile` and CLI help; a fake support
dataset where support-risk done was known exactly; and a no-support replay check
that preserved the prior H3 done fraction `0.1302083`. The full q90
state/command support-risk preparation ran in job `9356522` with `20000`
support rows, `4096` probe rows, threshold `0.620098`, action weight `0.0`,
support-risk done fraction `0.04818`, post-first-done fraction `0.11198`, and
final done fraction `0.171875`. A 20-iteration W&B-disabled AWR smoke on this
prepared replay completed in job `9356566`, wrote all three checkpoints, and
ended with loss `0.001026` and synthetic reward mean `-0.04121`.

Design implication: support-risk termination is now in the replay-preparation
method path, but still not in closed-loop world-model rollout generation. It is
therefore infrastructure, not a formal-run trigger. The next high-value variant
should make the synthetic rollout generator support-aware, so branches stop or
are downweighted at the first calibrated q90 support crossing instead of only
being repaired after the fixed buffer exists.

Support-risk termination is now also available during closed-loop generation.
`run_flow_mbpo_v0_smoke.py --support-risk-termination` builds the same
real-data support/probe split, computes support distance for each generated
`(state, command, action)` before advancing the model, marks threshold-crossing
rows as `support_risk_done`, and freezes that branch state for later horizon
steps. The buffer records `rollout_active`, `support_risk_distance`,
`support_risk_threshold`, and `support_risk_done`. With the flag disabled, a
fake-generator regression check confirmed that state advancement and done
behavior remain unchanged.

The first full support-aware generation smoke ran in Slurm job `9356635` on the
trajectory/chunk 5k H3 Flow ensemble with `256` starts, horizon `3`, q90
support threshold `0.620098`, and action weight `0.0`. It produced
support-risk done fraction `0.09635`, rollout active fraction `0.94010`,
horizon done fractions `[0.08203, 0.09766, 0.10938]`, support-risk distance
mean/p90/max `0.23963`/`0.60024`/`0.97118`, and lower next-state delta mean
`0.12522` because stopped branches no longer advance. Preparing that generated
buffer with the usual uncertainty q90 settings yielded final replay done
fraction `0.16406` and conservative reward mean `-0.06376`.

A 20-iteration W&B-disabled AWR smoke on the support-aware generated replay ran
in job `9356654`, wrote final, best, and best-training checkpoints, and ended
with loss `0.001026`, synthetic reward mean `-0.04690`, and replay synthetic
done fraction `0.1640625`. This is method-aligned infrastructure, not a policy
claim. The next design step is to combine support-aware generated replay with a
stronger conservative update objective, or run a carefully justified one-seed
formal only if it has a credible path to lower fall rate than matched BC.

A first conservative-Q update path is now implemented. In
`run_flow_mbpo_v0_awr_update.py`, `--conservative-q-weight > 0` trains a
deterministic Q critic on mixed real/synthetic one-step transitions with Bellman
loss plus a CQL-style `softplus(Q(actor_action) - Q(data_action))` penalty.
`--critic-actor-weight > 0` can add a small actor loss that maximizes the
current conservative critic while the existing AWR/BC-anchor losses remain in
place. The path is opt-in and saves `final_q_critic.pt` with both critic and
target critic state when enabled.

Validation so far is mechanical. `py_compile`, CLI help, and a CPU fake-data
critic smoke passed. W&B-disabled job `9356778` ran `20` iterations on the
support-aware generated H3 replay with conservative-Q weight `1.0` and actor
critic weight `0.01`; it wrote actor checkpoints and logged critic loss
`0.76992`, Bellman loss `0.07677`, CQL loss `0.69315`, CQL gap mean
`1.38e-5`, `Q(data)` mean `0.09484`, and `Q(actor)` mean `0.09485`. Job
`9356793` confirmed the critic checkpoint is written and loadable. The near-zero
CQL gap means this is not yet a useful conservative separation; tune critic
training length, CQL weight, sampled actions, or generated-state mix before any
formal run.

Random-action CQL is now the first tuning step for that critic. With
`--critic-random-actions K`, the critic samples `K` uniform actions per state,
evaluates actor plus random actions, and uses a temperature-scaled logsumexp OOD
value against `Q(data)`. Defaults keep the earlier actor-only CQL path. A CPU
fake-data smoke with `critic_random_actions=4` passed, and W&B-disabled job
`9356862` ran `20` iterations on the support-aware generated H3 replay with
`K=10`, conservative-Q weight `1.0`, CQL temperature `1.0`, and actor critic
weight `0.01`. Final metrics were critic loss `2.41859`, Bellman loss
`0.07703`, CQL loss/gap `2.34156`, `Q(data)` mean `0.10054`, `Q(actor)` mean
`0.10057`, `Q(random)` mean `0.03787`, and `Q(random)` max `0.19931`. This
produces a meaningful conservative training gap, unlike the actor-only smoke,
but remains W&B-disabled mechanical evidence. Do not formalize until a longer or
better-tuned smoke shows stable critic/actor behavior.

The first longer random-action CQL smoke ran in job `9357006` for `100`
iterations with the same support-aware generated H3 replay and `K=10`. The CQL
gap decreased from `2.3921` at iter 1 to `1.8522` at iter 100, while
`Q(random)` mean dropped from `0.0290` to `-0.4255`; this is the intended
average random-action pessimism. The warning sign is that `Q(random)` max rose
from `0.1326` to `0.7369`, and `Q(actor)` rose with `Q(data)`. Treat this as a
useful critic-training diagnostic, not a formal candidate. Before W&B formal
eval, tune CQL weight, temperature, random-action count, and actor critic weight
so the critic is conservative without creating high-valued random-action tails.

The first actor-weight ablation ran in job `9357054` with
`--critic-actor-weight 0.0` and otherwise identical settings to the 100-iteration
random-action CQL smoke. Final CQL gap was `1.8496`, `Q(random)` mean was
`-0.4280`, and `Q(random)` max was `0.7380`, nearly matching the actor-weight
`0.01` run. This means the high random-action tail is mainly a critic/CQL
sampling issue, not just actor optimization pressure. Keep actor critic weight
off while tuning CQL weight, temperature, and sampled-action coverage.

The first CQL-weight stress test ran in job `9357126` with
`--conservative-q-weight 5.0`, `--critic-actor-weight 0.0`, and otherwise the
same random-action setup. It did not fix the tail: final `Q(random)` max rose to
`1.0219` versus `0.7380` for weight `1.0`, while `Q(data)` and `Q(actor)` also
rose and Bellman loss worsened. Higher CQL weight is therefore not a clean safety
knob here. Use weight `1.0` for the next smokes and tune temperature or random
action coverage instead.

The first temperature test ran in job `9357174` with CQL temperature `0.1`,
weight `1.0`, random actions `10`, and actor critic weight `0.0`. It reduced the
tail and Q scale: final `Q(random)` max was `0.4415` versus `0.7380` at
temperature `1.0`, and `Q(data)`/`Q(actor)` stayed near `0.18` instead of
`0.25`. The cost is that CQL gap fell to `0.0658`, which may be too weak for
policy improvement. The next tuning point should be an intermediate temperature,
for example `0.5`, before any formal W&B run.

The intermediate temperature smoke ran in job `9357227` with temperature `0.5`.
It landed between the previous endpoints: final CQL gap `0.7382`,
`Q(random)` mean `-0.3991`, and `Q(random)` max `0.6697`. This is the cleanest
short-smoke compromise so far: stronger than temperature `0.1` and less tail
heavy than temperature `1.0`. It still does not provide real policy evidence.
Use this setting only for the next W&B-disabled continuation or a small
real-eval plumbing check, not for a policy-improvement claim.

That real-eval plumbing check ran in job `9357292` with W&B disabled. The code
path works: final, best-real-eval, best-training, real-eval snapshot, and critic
checkpoints were written, and `best_policy_extraction.pt` is marked as a true
best-real-eval snapshot. The actual 8-episode eval was poor: return `18.7727`,
length `283.50`, fall `1.000`. This rejects the current temp-`0.5`
conservative-Q setting as a formal candidate. Conservative-Q can remain a
diagnostic, but the policy-update objective needs another change before W&B
formal eval/video is worth running.

A W&B-disabled roll10 MP4 diagnostic of the same temp-`0.5` actor ran in job
`9370468`. It rendered `10` episodes at `max_steps=1000` and produced return
`54.2864`, length `689.80`, fall `0.400`. The final, best-real, and iter-100
snapshot actor weights are identical, so this render covers the true-best actor
for that smoke. This result is close to matched BC seed0 final roll10
(`54.1283`, length `688.40`, fall `0.400`) but does not reduce fall rate, and it
conflicts with the poor 8-episode scalar eval. Treat this as a selection/eval
variance warning: a candidate that merely ties BC on one roll10 while failing
short real eval is not ready for formal W&B evaluation.

True-best selection now has an opt-in gate-aware score. In
`run_flow_mbpo_v0_awr_update.py`, `--real-eval-selection-metric` defaults to
`return` for backward compatibility, and can be set to `return_length_fall` to
rank real-eval snapshots by
`return_mean + length_weight * episode_length_mean - fall_penalty *
fall_rate_mean`. The selected best checkpoint records `real_eval/selection_score`
and `real_eval/selection_metric`. This does not replace the final formal gate,
but it prevents return-only checkpoint selection from silently diverging from
the return/length/fall claim criterion. A one-iteration, two-episode
W&B-disabled smoke in job `9370586` verified that the metric is logged into the
snapshot and best-real checkpoint.

The update loop also has opt-in real-eval early stopping. After a real-eval
snapshot, `--real-eval-stop-score-below` can stop clearly rejected candidates by
selection-score threshold, and `--real-eval-early-stop-patience` with
`--real-eval-min-delta` can stop runs that stop improving. Defaults keep this
disabled. A W&B-disabled smoke in job `9370641` forced the threshold path and
confirmed the loop stopped at iter `1`, still writing final, best-real,
best-training, snapshot checkpoints, and summary fields `early_stop_iter` and
`early_stop_reason`. This is meant to conserve formal-run budget, not to relax
the final 40-episode eval plus roll10 video gate.

Real-eval snapshots can now log an explicit baseline gate. Supplying
`--real-eval-baseline-return`, `--real-eval-baseline-length`, and
`--real-eval-baseline-fall` records gaps to baseline, pass bits for each metric,
and `real_eval/baseline_gate_pass`. The rule matches the claim policy: return
and episode length must be at least baseline, and fall rate must be strictly
lower. A W&B-disabled smoke in job `9370667` verified these fields in stdout,
`summary.json`, and the saved real-eval snapshot. This makes W&B and checkpoint
metadata self-checking against the BC gate, but it still does not replace the
required final 40-episode eval and roll10 MP4 comparison.

Rollout rendering now mirrors that baseline gate. Supplying
`--baseline-return`, `--baseline-length`, and `--baseline-fall` to
`render_policy_rollout.py` records baseline values, gaps, pass bits, and
`baseline_gate_pass` in rollout `summary.json` and W&B logs. The pass rule is
the same as eval: return and length must meet or exceed baseline, fall must be
strictly lower. A two-episode W&B-disabled MP4 smoke in job `9370771` verified
the fields and wrote `rollout.mp4`; it is renderer logging evidence only, not a
policy result.

Manifest-driven rollout jobs can pass the same gate metadata. In
`run_policy_rollout_row.py`, optional fields `rollout_baseline_return`,
`rollout_baseline_length`, and `rollout_baseline_fall` are forwarded to the
renderer; shorter `baseline_*` aliases are accepted too. This keeps array
rollout jobs and hand-rendered diagnostics on the same evidence protocol.

Standalone policy eval now follows the same baseline gate protocol.
`eval_policy_checkpoint.py` accepts `--baseline-return`, `--baseline-length`,
and `--baseline-fall`, then writes gaps, pass bits, and `baseline_gate_pass` to
`summary.json`, stdout, and W&B numeric logs. `run_policy_eval_row.py` forwards
`eval_baseline_return`, `eval_baseline_length`, and `eval_baseline_fall` from
manifests, with shorter `baseline_*` aliases. A two-episode W&B-disabled smoke
in job `9370850` verified the path; it is logging evidence only and does not
replace the required 40-episode eval.

Candidate snapshot eval/render planning also carries baseline gate metadata.
`build_flow_mbpo_candidate_eval_plan.py` writes eval baseline columns using the
aggregate BC scalar baseline (`45.8491`, length `594.97`, fall `0.625`) and
rollout baseline columns using the matched seed0 BC final roll10 baseline
(`54.1283`, length `688.40`, fall `0.400`) by default. The generated commands
pass those values to standalone eval/render via `--baseline-*`, so candidate
plans and array manifests now follow the same gate-reporting protocol. A `/tmp`
plan check on the existing snapshot AWR run confirmed the baseline columns and
command flags for all `8` candidates. This is protocol hardening only, not new
policy evidence.

Candidate snapshot eval/render can now use the standard Slurm array runners.
The plan builder optionally emits direct-checkpoint eval and rollout manifests
with `policy_checkpoint`, candidate-specific output directories, W&B metadata,
and baseline gate fields. `run_policy_eval_row.py` and
`run_policy_rollout_row.py` treat rows with `policy_checkpoint` as exact
checkpoint jobs, while preserving the old stage-based final/best behavior for
existing manifests. This lets snapshot candidates run through
`submit_array.sh --kind policy_eval` and `--kind policy_rollout`, keeping the
default `embers` QOS guard and W&B row-runner path instead of relying only on a
hand-run shell command list.

Formal eval/render provenance is now explicit. Both standalone eval and rollout
accept `--notes`, row runners forward manifest `notes`, and candidate plan
manifests include notes. Rollout summaries now mirror eval summaries by
recording dataset, metadata, normalization, seed, task id, WM method, policy
type, git SHA/branch, command, checkpoint, notes, and baseline gate fields.
Candidate W&B names are derived from the actual episode/step settings, so a
nonstandard candidate smoke is not mislabeled as `eval40` or `rollout1000_ep10`.

Candidate ranking now prefers recorded gate metadata when available.
`rank_flow_mbpo_candidate_evidence.py` uses `baseline_gate_pass` from eval or
rollout summaries if the summary says a baseline gate was configured, and it
labels the source as `summary`. Older results without gate metadata still use
the computed comparison against supplied baseline summaries and are labeled
`computed`. This keeps reports aligned with the eval/render summaries that will
be used for formal evidence claims.

Formal candidate array submission has an opt-in preflight check.
`submit_array.sh --require-formal-metadata` applies to `policy_eval` and
`policy_rollout` manifests and rejects missing W&B metadata, missing notes,
disabled W&B, missing baseline gates, and incomplete direct-checkpoint output
paths before calling `sbatch`. This keeps formal candidate eval/render jobs on
the same evidence protocol while leaving smoke and legacy submissions
unchanged unless the flag is requested.

AWR training notes are now part of checkpoint provenance. The Flow-MBPO AWR
update script accepts `--notes`, records notes in its summary/W&B config, and
writes them into checkpoint `args`. Downstream eval/render therefore can recover
the training-run notes from final, best, best-training, real-eval snapshot, or
critic checkpoints.

Synthetic replay notes are now part of replay provenance too. The synthetic
rollout smoke generator accepts `--notes`, records them in `summary.json`, and
writes a `synthetic_buffer_metadata.json` sidecar with git/command/data/model
paths, support-risk settings, and tensor shapes. The replay-preparation script
also accepts `--notes`, writes them to `summary.json`, and emits
`synthetic_replay_metadata.json` with the preparation settings plus any input
buffer notes. The `.pt` artifacts remain tensor-schema compatible for existing
AWR/support consumers; provenance is kept in JSON sidecars.

Replay preparation can now log to W&B as a formal provenance step. With
`--enable-wandb`, `prepare_flow_mbpo_v0_synthetic_replay.py` initializes a
`flow_mbpo_v0_synthetic_replay` run, records the full summary as config, and
logs scalar replay diagnostics such as transition count, conservative reward,
uncertainty, and done/support-risk fractions. This keeps future formal replay
preparation on the same W&B/notes path as AWR training, eval, and rollout
rendering.

Synthetic generation and replay-preparation artifacts now also store the W&B
run id and URL locally. When W&B initialization succeeds, both scripts rewrite
their `summary.json` and metadata sidecar JSON with `wandb_run_id` and
`wandb_run_url`, so a saved buffer/replay can be traced back to W&B without
searching Slurm logs.

Synthetic generation and replay preparation can now run through the standard
MJLab QS Slurm array wrapper. `submit_array.sh` supports
`--kind flow_mbpo_smoke` and `--kind flow_mbpo_replay`, using row runners that
forward manifest metadata to the underlying synthetic-buffer and replay
scripts. With `--require-formal-metadata`, these rows must enable W&B and
provide W&B project/group/name, notes, direct output dirs, and required inputs
before `sbatch` is called.
