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
