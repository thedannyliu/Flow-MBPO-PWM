# R0-R4 Controlled Matrix Status

Date: 2026-06-02

Purpose: turn the active plan's R0-R4 matrix into a concrete artifact map before
submitting more GPU work. This is a preparation record, not a performance claim.

## Fidelity Boundary

The current R0 evidence is adapter-level evidence, not a byte-identical upstream
PWM pipeline run. The MJLab job uses
`scripts/experiments/mjlab_qs/run_original_pwm_adapter.py`, which imports the
upstream `baselines/PWM/src/pwm.algorithms.pwm.PWM` class and uses the upstream
PWM actor, critic, SimNorm world model, `compute_wm_loss`, `update`, TD(lambda),
return RMS, and LR schedule. It does not call
`baselines/PWM/scripts/train_dflex.py`, does not instantiate a DFlex env through
the upstream Hydra config, and does not call the upstream `agent.train()` loop.

Therefore R0 should be read as:

```text
upstream PWM algorithm/model/update adapted to MJLab-QS windows and MJLab eval
```

It should not be read as:

```text
full original PWM train_dflex/train_multitask pipeline reproduced on MJLab
```

Since that boundary was written, a separate full upstream bridge has been
smoke-tested: job `9401871` ran `baselines/PWM/scripts/train_dflex.py` with
upstream `pwm.algorithms.pwm.PWM` and a wrapper-generated MJLab env config, then
completed 0:0 and wrote init/best/final policies. This proves feasibility of a
full upstream orchestration path on MJLab, but it is not a fixed-protocol
performance row because it lacks 40-episode eval and 10-episode videos.
Long diagnostic job `9401906` has since completed 0:0 through the same bridge,
wrote best/final and iter50/100/150 checkpoints, and ended with poor internal
upstream eval length 35.33. W&B-disabled final/best real-env eval smoke arrays
`9401975` on H200 completed 0:0 and showed collapse for both checkpoints; the
duplicate H100 backup `9401980` was canceled.

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
| R0 upstream PWM algorithm adapter + original PWM policy/update | Baseline PWM-algorithm transfer to MJLab through the documented QS-window adapter | Formal adapter checkpoints from `9387895`; fix2 eval `9395746` and rollout `9396189` completed. Matched evidence says final/best collapse with eval fall `1.000` and video fall `1.000`. | Nothing for the adapter-level negative gate; this row does not prove failure of a full upstream `train_dflex.py` pipeline on MJLab. | Use as fixed adapter-level R0 baseline. Do not resubmit unless changing runtime, dataset, or building a true upstream-pipeline MJLab bridge. |
| R0b full upstream PWM pipeline bridge | Feasibility check for upstream `train_dflex.py` / `PWM.train()` on MJLab | Smoke job `9401871` completed 0:0, initialized MJLab G1 with obs_dim 210 and act_dim 29, ran upstream actor/critic/WM updates, and wrote init/best/final policies. Long diagnostic `9401906` completed 0:0, wrote best/final and iter50/100/150 checkpoints, and reported internal eval mean episode loss `0.44`, discounted loss `0.38`, length `35.33`. Real-env eval smoke `9401975` completed 0:0: final return `-1.7458`, length `38.125`, fall `1.000`; best return `-1.6728`, length `47.500`, fall `1.000`. W&B-on formal arrays `9402769`/`9402771`/`9402772` failed only at W&B init/upload after local eval or video work; thread-backed replacements completed across H200/H100/L40S. H200 formal eval40: final `-1.7261 / 39.525 / 1.000`, best `-1.5293 / 50.650 / 1.000`. H200 rollout10: final `-1.8182 / 37.900 / 1.000`, best `-1.3551 / 52.500 / 1.000`, MP4/W&B uploaded. H100/L40S backups match the same collapse. | Nothing for the full-upstream negative gate; final/best eval40 and rollout10 MP4/W&B artifacts are complete and consistently fail the BC gate. | Treat as a completed negative full-upstream-pipeline R0b gate. Do not spend more budget on this exact upstream PWM checkpoint unless changing the MJLab bridge, seed, training budget, or protocol. |
| R1 Flow WM + original PWM policy/update | Flow WM only | Old 2x2 runner has a `flow_endpoint` WM with `mlp` policy row, but that row is not the faithful original PWM update under the fixed R0 protocol. | Need a row that swaps only the WM while preserving the faithful original PWM policy/update, dataset, seed, eval, and video protocol. | Do not claim R1 from old 2x2 rows. Build or identify a faithful-policy runner that accepts Flow WM. |
| R2 original PWM WM + Flow policy architecture | Flow policy/update only | Old 2x2 runner has `mlp_ref` WM with `flow` policy, but it used the prior PWM-style runner rather than the faithful R0 update and does not satisfy final/best eval/video gates. | Need one row that keeps original PWM WM and changes only the policy architecture/update to Flow. | Treat old row as diagnostic only; design a fixed-protocol row before submission. |
| R3 Flow WM + Flow policy architecture | Combined Flow replacement | Old 2x2 Flow WM + Flow policy row exists and broad Flow-MBPO AWR/AWAC diagnostics exist. Both remain below BC or collapse; broad AWR best diagnostic return is `25.9699`, length `360.0`, fall `1.000`. Completed AWAC diagnostics across endpoint, trajectory, residual, and MLP-reference replay families also remain below BC and fall at rate `1.000`. | Need matched final/best 40-episode eval and 10-episode videos under one fixed seed/protocol if using this row for a causal matrix. | Do not expand the current conservative AWR/AWAC settings; use them as exploitation/fall evidence and change mechanism before further R3/R4 submissions. |
| R4 best current Flow-PWM config, exact reproduction | Exploratory reproduction of strongest current Flow-MBPO candidate | Best documented Flow-MBPO rows are H1 endpoint AWR and trajectory/chunk H3 variants with stronger seed0 eval/video metrics than old 2x2 rows, but not a one-variable causal row. | Need explicit selection of one R4 candidate and then final/best eval/video if missing. | Select R4 from existing ranked evidence only after recording which gates are already present and which are missing. |

## Current Candidate Interpretation

```text
R0 is a completed negative adapter-level baseline.
R1 and R2 are not currently satisfied by existing artifacts.
R3 has old diagnostic evidence and broad AWR negative evidence, but no clean
one-variable claim.
R4 should be treated as exploratory even if it beats R0, because it changes more
than one variable.
```

## R4 Existing-Candidate Selection

This selection uses already completed artifacts only. It does not trigger a new
submission because all three strongest existing R4 candidates already have
final/best 40-episode eval summaries and final/best 10-episode rollout videos.

Reference gates:

```text
aggregate BC eval comparator: return 45.8491, length 594.97, fall 0.625
matched seed0 BC video comparator: return 54.1283, length 688.40, fall 0.400
strict improvement requires return >= baseline, length >= baseline, and fall
strictly lower than baseline. Tying fall is not enough.
```

| Candidate | Selection basis | Eval40 final | Eval40 best | Roll10 final | Roll10 best | Gate interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| `flow_endpoint_seed0_h1_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0` | Best scalar eval final among existing Flow-MBPO rows | return `60.8721`, length `759.30`, fall `0.450` | return `46.1720`, length `600.60`, fall `0.700` | return `47.4617`, length `625.60`, fall `0.500` | return `55.5533`, length `707.60`, fall `0.400` | Select as R4 scalar-eval reproduction if one existing row must be named. It is not a verified improvement because best eval regresses and video fall only ties matched BC for best while final video is worse. |
| `flow_trajectory_chunk_5k_seed0_h3_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0` | Best documented trajectory/chunk balance before low-synth | return `48.7296`, length `637.22`, fall `0.575` | return `37.5778`, length `496.05`, fall `0.800` | return `54.4904`, length `694.00`, fall `0.400` | return `55.3382`, length `706.30`, fall `0.400` | Promising but not selected over H1 on scalar eval. Both videos tie matched BC fall and best eval regresses. |
| `flow_trajectory_chunk_5k_seed0_h3_unc0p5_q0p90_cons_r240_s16_anchor1_iter500_s0` | Strongest video return/length among existing Flow-MBPO rows | return `47.5960`, length `612.00`, fall `0.600` | return `39.8802`, length `527.17`, fall `0.725` | return `55.4222`, length `707.20`, fall `0.400` | return `55.5495`, length `708.00`, fall `0.400` | Best video scalar row, but eval best regresses and video fall only ties matched BC. Do not expand as success. |

R4 selection decision:

```text
selected_r4_for_scalar_reproduction:
  flow_endpoint_seed0_h1_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0

reason:
  its final checkpoint has the strongest existing 40-episode scalar eval:
  return 60.8721, length 759.30, fall 0.450.

claim_boundary:
  this is an exploratory best-current reproduction row, not a causal R0-R3
  matrix row and not a verified policy improvement. The final/best gates are
  inconsistent and no video fall improvement over matched seed0 BC is shown.

next_action:
  do not submit duplicate eval/video for this R4 candidate. If expanding R4,
  first change the objective toward fall-risk/OOD reduction or short-horizon
  pessimistic Flow-MBPO, then require the same final/best eval/video gates.
```

The completed broad conservative AWR sweep strengthens the exploitation/fall
diagnosis: all completed broad/shard rows, including the final A100 shards
`9400442`, `9400528_1`, and `9400528_2`, fall at rate `1.000` in 8-episode real
evals and remain below the BC comparator. The repaired support-truncation
diagnostic `9402171_[0-1]` is also negative: q0.90 reports return `23.1614`,
length `337.375`, fall `1.000`, and q0.50 reports return `19.4189`, length
`291.250`, fall `1.000`; both early-stop at iter 20 and fail the BC gate. This
supports the active plan's pivot toward a genuinely different fall-risk/OOD or
short-horizon objective instead of duplicating the same AWR/support settings.

The completed AWAC diagnostics also close the current critic-derived advantage
setting as negative. Fixed-SHA endpoint seed1 rows on H200/H100/L40S
`9402337`/`9402338`/`9402339` report returns `8.5052`-`11.3574`, lengths
`154.000`-`192.375`, and fall `1.000`. Replay-family rows on H200/H100/L40S
`9402361`/`9402364`/`9402363` across trajectory, residual, and MLP-reference
replays report returns `18.6628`-`19.9263`, lengths `272.000`-`298.125`, and
fall `1.000`. H100/L40S endpoint backups `9402277`/`9402278` are likewise
negative. Do not promote this AWAC setting to formal eval/video.

Existing v1 support/pessimism artifacts were re-exported after the full
upstream PWM negative smoke. The high-return rows are 1-update / 2-episode
gate-logging checks from the BC checkpoint and should not be treated as
Flow-MBPO improvements. The substantive v1 rows remain negative: the 100-iter
CQL random-action eval8 row reports return `18.7727`, length `283.5`, fall
`1.000`, and the 500-iter action-deviation row reports best real return
`19.577`, length `295.0`, fall `1.000`. The later support-truncation manifest
has now been tried and is also negative, so the next useful R3/R4 work should
change the mechanism rather than retrying support truncation plus conservative
AWR.

## Candidate Jobs Before Next Submission

| Candidate | Type | Inputs exist? | W&B mode | Expected artifacts | GPU / QOS | Dependency required? | Submit decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `r1_flow_wm_faithful_pwm_update` | formal design / not ready | Not yet; faithful original PWM update must accept Flow WM under the fixed MJLab bridge. | W&B on when formal. | Final/best checkpoints, 40-episode eval, 10-episode videos, WM/prediction/calibration/grad/action/OOD metrics. | H200/H100/A100/L40S / `embers`. | Yes if runner or WM artifact is missing. | Do not submit until runner inputs are explicit. |
| `r2_original_wm_flow_policy_update` | formal design / not ready | Not yet; need original PWM WM plus Flow policy/update with all other protocol choices fixed. | W&B on when formal. | Same fixed-protocol artifacts. | H200/H100/A100/L40S / `embers`. | Yes if runner or checkpoint path is missing. | Do not submit until row is implementable. |
| `r4_select_existing_best_flow_mbpo` | eval / exploratory | Partly; existing Flow-MBPO candidate eval/video artifacts exist, but candidate selection needs a fresh gate table. | W&B on for any missing formal eval/video. | Ranking table plus missing final/best eval/video if selected candidate lacks them. | H200/H100/A100/L40S / `embers`. | No if selected checkpoint exists; yes only if missing checkpoint. | Prepare selection record first; no duplicate conservative AWR submission. |
| `pessimistic_short_horizon_flow_mbpo_next` | diagnostic / exploratory | Existing H=1/3/5 replay and support artifacts partly exist; the broad AWR/AWAC results motivate stronger pessimism/fall gating rather than exact duplicate rows. | W&B off for new-code smokes. | Support/OOD/fall-stop diagnostics, real eval every 10 updates, checkpoint summaries. | H200/H100/A100/L40S / `embers`. | No for rows using existing replays; yes if fall-risk labels/head are missing. | Candidate for future submission only after a mechanism change beyond the current conservative AWR/AWAC settings. |

No new sbatch submission is made from this record. R4 existing candidates have
their eval/video gates, the queue is currently empty, and the completed AWR plus
support-truncation diagnostics close the current duplicate-support branch as
negative.

## Full-Upstream PWM MJLab Formal Diagnostics

User direction changed the R0b next action from "do not submit formal gates" to
"run a complete PWM pipeline on MJLab and inspect the effect." Commit `6d6949a`
adds an upstream render runner that loads full `train_dflex.py` checkpoints via
the saved Hydra config and evaluates/renders through upstream `PWM.train()`
artifacts, not policy-extraction proxy checkpoints.

Submission time: 2026-06-03, `embers`, W&B disabled, existing long diagnostic
checkpoint from `9401906`:

```text
hydra_run_dir:
  baselines/PWM/scripts/outputs/2026-06-02/21-35-04
policy_dir:
  baselines/PWM/scripts/outputs/2026-06-02/21-35-04/logs/upstream_pwm_mjlab_full_longdiag_h200_seed0_20260602
checkpoints:
  final_policy.pt
  best_policy.pt
```

Submitted arrays:

| Job ID | GPU | Mode | Rows | Output root | Purpose |
| --- | --- | --- | --- | --- | --- |
| `9402743_[0-1%2]` | H200 | eval40 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_formal_eval40_h200_20260603/` | 40-episode real MJLab eval for full upstream PWM checkpoint |
| `9402742_[0-1%2]` | H200 | rollout10 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_rollout10_h200_20260603/` | 10-episode MP4 rollout for full upstream PWM checkpoint |
| `9402746_[0-1%2]` | H100 | eval40 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_formal_eval40_h100_20260603/` | Backup/parallel formal eval with distinct output root |
| `9402747_[0-1%2]` | H100 | rollout10 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_rollout10_h100_20260603/` | Backup/parallel rollout with distinct output root |
| `9402744_[0-1%2]` | L40S | eval40 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_formal_eval40_l40s_20260603/` | Backup/parallel formal eval with distinct output root |
| `9402745_[0-1%2]` | L40S | rollout10 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_rollout10_l40s_20260603/` | Backup/parallel rollout with distinct output root |

Initial queue state: all pending. H200 showed `Resources`; H100/L40S showed
`None`; A100 already had pending AWAC arrays, so no additional A100 upstream
duplicate was submitted in this batch.

Replacement record:

The first full-upstream formal arrays were canceled before start because the
submit wrapper still exported `WANDB_MODE=disabled`. That made the jobs useful
as diagnostics but incomplete for the formal gate, which requires W&B-backed
eval notes and MP4/W&B rollout videos. Affected arrays:

```text
9402743_[0-1%2] H200 eval40, canceled before start
9402742_[0-1%2] H200 rollout10, canceled before start
9402746_[0-1%2] H100 eval40, canceled before start
9402747_[0-1%2] H100 rollout10, canceled before start
9402744_[0-1%2] L40S eval40, canceled before start
9402745_[0-1%2] L40S rollout10, canceled before start
```

Commit `0d64c84` adds W&B logging to the full-upstream eval/render runners and
changes the submit wrapper to use W&B-on output roots. Replacement arrays:

| Job ID | GPU | Mode | Rows | Output root | W&B |
| --- | --- | --- | --- | --- | --- |
| `9402769_[0-1%2]` | H200 | eval40 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_formal_eval40_wandb_h200_20260603/` | project `flow-mbpo-mjlab-full-upstream-pwm`, group `upstream_pwm_mjlab_full_pipeline_20260603_eval` |
| `9402771_[0-1%2]` | H200 | rollout10 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_rollout10_wandb_h200_20260603/` | project `flow-mbpo-mjlab-full-upstream-pwm`, group `upstream_pwm_mjlab_full_pipeline_20260603_rollout` |
| `9402774_[0-1%2]` | H100 | eval40 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_formal_eval40_wandb_h100_20260603/` | same W&B project/group convention |
| `9402772_[0-1%2]` | H100 | rollout10 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_rollout10_wandb_h100_20260603/` | same W&B project/group convention |
| `9402773_[0-1%2]` | L40S | eval40 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_formal_eval40_wandb_l40s_20260603/` | same W&B project/group convention |
| `9402770_[0-1%2]` | L40S | rollout10 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_rollout10_wandb_l40s_20260603/` | same W&B project/group convention |

The replacement `sbatch` calls returned Slurm socket timeout messages, but
`squeue`/`sacct` confirmed that all six replacement arrays were accepted and are
pending. Treat the canceled W&B-disabled arrays as superseded and unusable for
formal claims.

### Full-Upstream Formal W&B Threadfix

The first W&B-on replacement set exposed a formal-gate infrastructure bug rather
than an MJLab/PWM failure. Jobs `9402769_0`, `9402769_1`, `9402771_0`,
`9402771_1`, and `9402772_0` reached the real-env eval or MP4-render path, but
failed at `wandb.init` with `ModuleNotFoundError:
wandb.sdk.internal.internal` from the old W&B 0.12.21 backend subprocess in the
mixed locked-PWM/MJLab wrapper. H200 rollout jobs still wrote local MP4 files,
but the jobs are unusable for formal claims because W&B upload failed.

Affected queued arrays were canceled to avoid repeating the known root cause:
`9402770_[0-1%2]`, `9402772_[1%2]`, `9402773_[0-1%2]`, and
`9402774_[0-1%2]`. Commit `7c959cb` sets
`WANDB_START_METHOD=thread` and passes `settings={"start_method": "thread"}` to
both formal W&B runners. A locked-wrapper offline W&B thread smoke completed
successfully before resubmission.

Replacement arrays submitted after `7c959cb`:

| Job ID | GPU | Mode | Rows | Output root | Notes |
| --- | --- | --- | --- | --- | --- |
| `9402882_[0-1%2]` | H200 | eval40 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_formal_eval40_wandb_threadfix_h200_20260603/` | pending Resources at first check |
| `9402885_[0-1%2]` | H200 | rollout10 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_rollout10_wandb_threadfix_h200_20260603/` | pending Resources at first check |
| `9402884_[0-1%2]` | H100 | eval40 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_formal_eval40_wandb_threadfix_h100_20260603/` | pending Priority at first check |
| `9402883_[0-1%2]` | H100 | rollout10 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_rollout10_wandb_threadfix_h100_20260603/` | pending Priority at first check |
| `9402887_[0-1%2]` | L40S | eval40 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_formal_eval40_wandb_threadfix_l40s_20260603/` | submitted with `CPUS_PER_TASK=4` for L40S CPU:GPU policy |
| `9402888_[0-1%2]` | L40S | rollout10 | final/best | `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_rollout10_wandb_threadfix_l40s_20260603/` | submitted with `CPUS_PER_TASK=4` for L40S CPU:GPU policy |

Threadfix result update: all six replacement arrays completed and synced W&B.
The full-upstream `train_dflex.py` / `PWM.train()` checkpoint is a complete
negative MJLab gate. H200 eval40 W&B URLs are
`https://wandb.ai/danny010324/flow-mbpo-mjlab-full-upstream-pwm/runs/qqloou0l`
for final and
`https://wandb.ai/danny010324/flow-mbpo-mjlab-full-upstream-pwm/runs/3y6rzkg9`
for best. H200 rollout10 W&B URLs are
`https://wandb.ai/danny010324/flow-mbpo-mjlab-full-upstream-pwm/runs/78bhu0l3`
for final and
`https://wandb.ai/danny010324/flow-mbpo-mjlab-full-upstream-pwm/runs/0qpe9tlq`
for best, with MP4 files under
`scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_rollout10_wandb_threadfix_h200_20260603/{final,best}/rollout.mp4`.
