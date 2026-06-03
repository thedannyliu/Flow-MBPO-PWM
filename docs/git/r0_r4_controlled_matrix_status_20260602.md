# R0-R4 Controlled Matrix Status

Date: 2026-06-02

Purpose: turn the active plan's R0-R4 matrix into a concrete artifact map before
submitting more GPU work. This is a preparation record, not a performance claim.

## Fixed Protocol Target

The controlled matrix should use one fixed MJLab QS protocol:

```text
task: Mjlab-Velocity-Flat-Unitree-G1
dataset: scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.pt
metadata: scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.json
normalization: scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16_normalization.json
BC comparator: return 45.8491, length 594.97, fall 0.625
video comparator: seed0 BC rollout return 54.1283, length 688.40, fall 0.400
eval protocol: final and true-best actors, 40 real episodes, 1000 max steps
video protocol: final and true-best actors, 10 episodes, 1000 max steps, MP4/W&B
claim boundary: imagined return, training loss, 8-episode smoke eval, or a single
completed checkpoint without matched video is diagnostic only
```

## Row Status

| Row | Intended one-variable comparison | Current evidence | Missing for controlled claim | Next action |
| --- | --- | --- | --- | --- |
| R0 faithful original PWM WM + original PWM policy/update | Baseline faithful PWM transfer to MJLab | Formal adapter checkpoints from `9387895`; fix2 eval `9395746` and rollout `9396189` completed. Matched evidence says final/best collapse with eval fall `1.000` and video fall `1.000`. | Nothing for the negative R0 gate; this row is complete as a collapsed MJLab baseline. | Use as fixed R0 baseline. Do not resubmit unless changing runtime or dataset. |
| R1 Flow WM + original PWM policy/update | Flow WM only | Old 2x2 runner has a `flow_endpoint` WM with `mlp` policy row, but that row is not the faithful original PWM update under the fixed R0 protocol. | Need a row that swaps only the WM while preserving the faithful original PWM policy/update, dataset, seed, eval, and video protocol. | Do not claim R1 from old 2x2 rows. Build or identify a faithful-policy runner that accepts Flow WM. |
| R2 original PWM WM + Flow policy architecture | Flow policy/update only | Old 2x2 runner has `mlp_ref` WM with `flow` policy, but it used the prior PWM-style runner rather than the faithful R0 update and does not satisfy final/best eval/video gates. | Need one row that keeps original PWM WM and changes only the policy architecture/update to Flow. | Treat old row as diagnostic only; design a fixed-protocol row before submission. |
| R3 Flow WM + Flow policy architecture | Combined Flow replacement | Old 2x2 Flow WM + Flow policy row exists and broad Flow-MBPO AWR diagnostics exist. Both remain below BC or collapse; broad AWR best diagnostic return is `25.9699`, length `360.0`, fall `1.000`. | Need matched final/best 40-episode eval and 10-episode videos under one fixed seed/protocol if using this row for a causal matrix. | Do not expand the conservative AWR setting; use it as exploitation/fall evidence. |
| R4 best current Flow-PWM config, exact reproduction | Exploratory reproduction of strongest current Flow-MBPO candidate | Best documented Flow-MBPO rows are H1 endpoint AWR and trajectory/chunk H3 variants with stronger seed0 eval/video metrics than old 2x2 rows, but not a one-variable causal row. | Need explicit selection of one R4 candidate and then final/best eval/video if missing. | Select R4 from existing ranked evidence only after recording which gates are already present and which are missing. |

## Current Candidate Interpretation

```text
R0 is a completed negative baseline.
R1 and R2 are not currently satisfied by existing artifacts.
R3 has old diagnostic evidence and broad AWR negative evidence, but no clean
one-variable claim.
R4 should be treated as exploratory even if it beats R0, because it changes more
than one variable.
```

The completed broad conservative AWR sweep strengthens the exploitation/fall
diagnosis: all completed broad/shard rows fall at rate `1.000` in 8-episode real
evals and remain below the BC comparator. This supports the active plan's pivot
toward stronger pessimism, support/OOD, fall-risk, and short-horizon model use
instead of duplicating the same AWR settings.

## Candidate Jobs Before Next Submission

| Candidate | Type | Inputs exist? | W&B mode | Expected artifacts | GPU / QOS | Dependency required? | Submit decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `r1_flow_wm_faithful_pwm_update` | formal design / not ready | Not yet; faithful original PWM update must accept Flow WM under the fixed MJLab bridge. | W&B on when formal. | Final/best checkpoints, 40-episode eval, 10-episode videos, WM/prediction/calibration/grad/action/OOD metrics. | H200/H100/A100/L40S / `embers`. | Yes if runner or WM artifact is missing. | Do not submit until runner inputs are explicit. |
| `r2_original_wm_flow_policy_update` | formal design / not ready | Not yet; need original PWM WM plus Flow policy/update with all other protocol choices fixed. | W&B on when formal. | Same fixed-protocol artifacts. | H200/H100/A100/L40S / `embers`. | Yes if runner or checkpoint path is missing. | Do not submit until row is implementable. |
| `r4_select_existing_best_flow_mbpo` | eval / exploratory | Partly; existing Flow-MBPO candidate eval/video artifacts exist, but candidate selection needs a fresh gate table. | W&B on for any missing formal eval/video. | Ranking table plus missing final/best eval/video if selected candidate lacks them. | H200/H100/A100/L40S / `embers`. | No if selected checkpoint exists; yes only if missing checkpoint. | Prepare selection record first; no duplicate conservative AWR submission. |
| `pessimistic_short_horizon_flow_mbpo_next` | diagnostic / exploratory | Existing H=1/3/5 replay and support artifacts partly exist; the broad AWR result motivates stronger pessimism/fall gating rather than exact duplicate rows. | W&B off for new-code smokes. | Support/OOD/fall-stop diagnostics, real eval every 10 updates, checkpoint summaries. | H200/H100/A100/L40S / `embers`. | No for rows using existing replays; yes if fall-risk labels/head are missing. | Candidate for future submission after a concrete row list is written. |

No new sbatch submission is made from this record. The queue already contains
pending LeWM pathfix arrays and A100 NEWT/Flow jobs, and no started row has
exposed a new failure to repair.
