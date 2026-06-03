# PWM, Flow-PWM, Flow-MBPO, BC, and Expert Evidence Inventory

Date: 2026-06-02

Purpose: collect the existing matched evidence rows requested by
`docs/goals/pwm_flow_sigreg_image_research_plan_20260602.md`. The faithful
original-PWM final/best eval and rollout jobs have now produced a complete
negative MJLab evidence package. This inventory is evidence-only. It does not
make a policy-improvement claim.

## PWM Fidelity Boundary

The row historically called "faithful original PWM adapter" is not a full
upstream `baselines/PWM/scripts/train_dflex.py` or `train_multitask.py`
reproduction on MJLab. It is an adapter around the upstream PWM implementation:
the runner imports `baselines/PWM/src/pwm.algorithms.pwm.PWM` and uses the
upstream actor, critic, SimNorm world model, `compute_wm_loss`, `update`,
TD(lambda), return RMS, and LR schedule, while the MJLab-QS window sampling,
pretrain loop, policy-update loop orchestration, and real MJLab eval bridge live
in `scripts/experiments/mjlab_qs/run_original_pwm_adapter.py`.

The negative row therefore supports this claim:

```text
the upstream PWM algorithm/model/update, when adapted to MJLab-QS windows and
MJLab eval, collapses under the fixed MJLab protocol.
```

It does not support this stronger claim:

```text
the byte-identical upstream PWM train_dflex/train_multitask pipeline fails on
MJLab.
```

Follow-up bridge status:

```text
9401871 upstream_pwm_mjlab_full_smoke_h200_20260602 completed 0:0 and proves
that the full upstream `baselines/PWM/scripts/train_dflex.py` orchestration can
run on MJLab through the locked-PWM/MJLab bridge. It used upstream
`pwm.algorithms.pwm.PWM`, upstream actor/critic/world-model targets, and a
wrapper-generated MJLab env config. This is feasibility evidence only: it ran a
short W&B-disabled smoke, wrote init/best/final policies, and did not perform
the fixed 40-episode eval plus 10-episode video gate.

9401906 upstream_pwm_mjlab_full_longdiag_h200_20260602 completed 0:0 and
extends the bridge evidence to a 200-epoch diagnostic. It wrote init, best,
final, and iter50/100/150 checkpoints, but the internal upstream eval remained
poor: mean episode loss 0.44, mean discounted loss 0.38, and mean episode
length 35.33. W&B-disabled real-env final/best eval smoke arrays `9401975`
on H200 completed 0:0 and validated checkpoint loading/evaluation, but both
checkpoints collapsed: final return -1.7458, length 38.125, fall 1.000; best
return -1.6728, length 47.500, fall 1.000. The duplicate H100 backup `9401980`
was canceled. No fixed-protocol eval/video claim exists yet.
```

## Scheduler Status During This Inventory

```text
9388552_[0-1] MJLab faithful original PWM final/best eval40: FAILED, exit 1:0; infrastructure only, `pwm` import path missing.
9388553_[0-1] MJLab faithful original PWM final/best rollout10 video: FAILED, exit 1:0; infrastructure only, `pwm` import path missing.
9388605 Ant locked DFlex final/best true eval repair: FAILED, exit 1:0; infrastructure only, DFlex rebuild lost system header CPATH.
9388606 Hopper locked DFlex WM-vs-real probe repair: FAILED, exit 1:0; infrastructure only, DFlex rebuild lost system header CPATH.
```

No Slurm logs existed for those jobs at the inventory check.

## Shared MJLab Dataset

The prior PWM/Flow rows below use the same Velocity Flat Unitree G1 QS window
dataset:

```text
dataset: scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.pt
metadata: scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.json
normalization: scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16_normalization.json
```

Reference table sources:

```text
docs/EXPERIMENT_LEDGER.md
scripts/outputs/mjlab_qs/reports/rollout_comparison_20260528.csv
scripts/outputs/mjlab_qs/status/rerun_g1_pwm_flow_policy2x2_aggregate_latest.csv
scripts/outputs/mjlab_qs/status/rerun_g1_pwm_flow_wm_aggregate_latest.csv
scripts/outputs/mjlab_qs/status/rerun_g1_pwm_flow_wm_sigreg_aggregate_latest.csv
```

## Matched Evidence Rows

| Row | Source stage | WM | Policy | Seed scope | Checkpoint/evidence | Return | Length | Fall | Status |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Full upstream PWM pipeline bridge | `9401871 upstream_pwm_mjlab_full_smoke_h200_20260602`; long diagnostic `9401906 upstream_pwm_mjlab_full_longdiag_h200_20260602`; real-env eval smoke `9401975` completed; duplicate H100 backup `9401980` canceled | Upstream PWM SimNorm world model through `train_dflex.py` | Upstream PWM actor/update through `PWM.train()` | seed 0 smoke/diagnostic | Smoke artifacts under `baselines/PWM/scripts/outputs/2026-06-02/21-19-04/logs/upstream_pwm_mjlab_full_smoke_h200_seed0_20260602/{init_policy.pt,best_policy.pt,final_policy.pt,final_policy.buffer/}`. Longdiag artifacts under `baselines/PWM/scripts/outputs/2026-06-02/21-35-04/logs/upstream_pwm_mjlab_full_longdiag_h200_seed0_20260602/{init_policy.pt,best_policy.pt,final_policy.pt,PWM_iter50_rew-2.pt,PWM_iter100_rew-2.pt,PWM_iter150_rew-2.pt,final_policy.buffer/}`. Eval smoke summaries under `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_real_eval_smoke_20260602/{final,best}/summary.json`. Slurm logs under `logs/slurm/mjlab_qs/upstream_pwm_full_pipeline/`. | eval smoke final `-1.7458`; eval smoke best `-1.6728`; smoke/longdiag internal losses were also poor | eval smoke final `38.125`; eval smoke best `47.500`; longdiag internal eval length `35.33` | eval smoke final/best `1.000` | Mechanically feasible full upstream pipeline, but negative real-env smoke evidence. This is not a formal 40-episode gate, but both checkpoints fail the BC baseline badly; do not spend formal eval/video budget on this diagnostic checkpoint unless the goal changes. |
| Upstream PWM algorithm adapter | `original_pwm_adapter_phase3_formal_20260601` plus fix2 eval/video jobs `9395746` and `9396189` | Upstream PWM model/update via MJLab-QS adapter | Original PWM extraction/update via adapter orchestration | seed 0 final/best | Formal checkpoints plus fall-aware eval/video package. Eval summaries: `scripts/outputs/mjlab_qs/policy_evals/original_pwm_adapter_phase3_eval40_fix2_20260602/.../{final,best}/summary.json`; rollout videos and summaries: `scripts/outputs/mjlab_qs/policy_rollouts/original_pwm_adapter_phase3_rollout10_fix2_20260602/.../{final,best}/`. | eval final `-0.8010`; eval best `-0.7778`; video final `-0.5265`; video best `-0.5218` | eval `44.45`; video final `46.80`; video best `46.40` | eval `1.000`; video `1.000` | Complete negative adapter-level evidence gate. The upstream PWM algorithm/model/update collapses through the MJLab-QS adapter and fails the BC baseline gate. |
| Prior PWM-style runner | `rerun_g1_pwm_flow_policy2x2_20260527` | `mlp_ref` | `mlp` | seeds 0-2 | `scripts/outputs/mjlab_qs/policy_rollouts/rerun_g1_pwm_flow_policy2x2_20260527/.../mlp_ref/mlp/offline/policy50k/` | `-4.5700` | `66.33` | `1.000` | Real rollout diagnostic only. Fails badly. |
| Flow policy only | `rerun_g1_pwm_flow_policy2x2_20260527` | `mlp_ref` | `flow` | seeds 0-2, aggregate from 3 final rollouts; status CSV says 2 of 3 completed in policy-eval aggregate | `scripts/outputs/mjlab_qs/policy_rollouts/rerun_g1_pwm_flow_policy2x2_20260527/.../mlp_ref/flow/offline/policy50k/` | `-3.4818` | `66.78` | `1.000` | Slightly less bad than MLP policy in this diagnostic, still failed. |
| Flow WM only | `rerun_g1_pwm_flow_policy2x2_20260527` | `flow_endpoint` | `mlp` | seeds 0-2 | `scripts/outputs/mjlab_qs/policy_rollouts/rerun_g1_pwm_flow_policy2x2_20260527/.../flow_endpoint/mlp/offline/policy50k/` | `-4.7720` | `60.33` | `1.000` | Failed. No evidence that Flow WM alone fixed policy extraction. |
| Flow WM + Flow policy | `rerun_g1_pwm_flow_policy2x2_20260527` | `flow_endpoint` | `flow` | seeds 0-2 | `scripts/outputs/mjlab_qs/policy_rollouts/rerun_g1_pwm_flow_policy2x2_20260527/.../flow_endpoint/flow/offline/policy50k/` | `-3.5414` | `85.33` | `0.889` | Best old 2x2 rollout row by fall, but still a collapse. |
| Flow-MBPO H1 endpoint AWR | `flow_mbpo_v0_awr_cons_r224_s32_anchor1_iter500_s0` | Flow endpoint synthetic replay, H1 | AWR policy | seed 0 final/best | Eval: `scripts/outputs/mjlab_qs/flow_mbpo_v0_eval/flow_endpoint_seed0_h1_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0/{final,best}/summary.json`; rollouts: `scripts/outputs/mjlab_qs/flow_mbpo_v0_rollouts/flow_endpoint_seed0_h1_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0/{final,best}/summary.json`. | eval final `60.8721`; eval best `46.1720`; video final `47.4617`; video best `55.5533` | eval final `759.30`; eval best `600.60`; video final `625.60`; video best `707.60` | eval final `0.450`; eval best `0.700`; video final `0.500`; video best `0.400` | Complete final/best eval and rollout gate is available. It is promising but unverified as an improvement: the best checkpoint regresses in eval40 fall, final video underperforms matched BC video, and best video only ties matched BC fall. |
| Flow-MBPO trajectory/chunk H3 | `flow_trajectory_chunk_5k_seed0_h3_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0` | Flow trajectory/chunk 5k synthetic replay, H3 | AWR policy | seed 0 final/best-real | Eval: `scripts/outputs/mjlab_qs/flow_mbpo_v0_eval/trajchunk_h3_awr_{final,best_real}_eval40_20260530/summary.json`; rollouts: `scripts/outputs/mjlab_qs/flow_mbpo_v0_rollouts/flow_trajectory_chunk_5k_seed0_h3_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0/{final_roll10,best_roll10}/summary.json`. | eval final `48.7296`; eval best-real `37.5778`; video final `54.4904`; video best-real `55.3382` | eval final `637.22`; eval best-real `496.05`; video final `694.00`; video best-real `706.30` | eval final `0.575`; eval best-real `0.800`; video final/best-real `0.400` | Complete final/best-real scalar and video gate is available. Final eval beats aggregate BC scalar return/fall, but best-real eval collapses and both videos only tie matched BC fall, so this remains promising but unverified. |
| Flow-MBPO trajectory/chunk H3 low synthetic ratio | `flow_trajectory_chunk_5k_seed0_h3_unc0p5_q0p90_cons_r240_s16_anchor1_iter500_s0` | Flow trajectory/chunk 5k synthetic replay, H3 | AWR policy | seed 0 final/best | Eval: `scripts/outputs/mjlab_qs/flow_mbpo_v0_eval/trajchunk_h3_awr_lowsynth_r240_s16_eval40_20260531/{final,best}/summary.json`; rollouts: `scripts/outputs/mjlab_qs/flow_mbpo_v0_rollouts/flow_trajectory_chunk_5k_seed0_h3_unc0p5_q0p90_cons_r240_s16_anchor1_iter500_s0/{final_roll10,best_roll10}/summary.json`. | eval final `47.5960`; eval best `39.8802`; video final `55.4222`; video best `55.5495` | eval final `612.00`; eval best `527.17`; video final `707.20`; video best `708.00` | eval final `0.600`; eval best `0.725`; video final/best `0.400` | Complete final/best scalar and video gate is available. Preserves video return/length gains and ties matched BC video fall, but eval best regresses and no fall improvement is shown. |
| Conservative broad Flow-MBPO AWR sweep | `flow_mbpo_broad_embers_awr_20260602` plus H200/H100/L40S/A100 shards `9400410`, `9400436`, `9400435`, `9400525`, `9400442`, `9400528` | Flow endpoint / trajectory / residual synthetic replays, H1/H3/H5 | Conservative AWR with support/action penalty and CQL-style critic | seeds 0-1 diagnostic rows | Summaries under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_20260602/*/summary.json` and `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_shards_20260602/{h200,h100,l40s,a100}/*/summary.json`. | best 8-episode row `25.9699`; A100 shard returns `16.5337`, `16.6024`, `10.1437` | best length `367.00`; best score row length `360.00`; A100 shard lengths `257.50`, `256.625`, `170.625` | `1.000` for every completed row | Negative diagnostic. All rows remain below BC `45.8491` / `594.97` / `0.625`; do not expand this exact setting or treat it as a formal improvement candidate. |
| Support-truncation Flow-MBPO AWR diagnostic | H200 repaired-wrapper `9402171_0` q0.90 and `9402171_1` q0.50; failed/canceled infrastructure attempts `9402080`, `9402136`, `9402128`, `9402170` recorded separately | Support-truncated Flow trajectory/chunk H3 replays | Conservative AWR with mixed CQL, support action penalty, real-eval early stop | seed 0 diagnostic rows | Summaries and checkpoints under `scripts/outputs/mjlab_qs/flow_mbpo_support_trunc_awr_diag_20260602/state_support_q90_trunc_cql_mixed_evalstop_s0/` and `.../state_support_q50_trunc_cql_mixed_evalstop_s0/`; real-eval snapshots at iter 20. | q0.90 `23.1614`; q0.50 `19.4189` | q0.90 `337.375`; q0.50 `291.250` | q0.90/q0.50 `1.000` | Negative diagnostic. Support truncation did not fix fall/OOD behavior; both rows early-stopped at iter 20 and failed the BC gate. |
| Best aggregate BC reference | `rerun_g1_bc_eval40_long1000_uniform_vs_smooth_20260528` | none | `mlp` BC | seeds 0-2 aggregate | 40-episode eval under `scripts/outputs/mjlab_qs/policy_evals/rerun_g1_bc_eval40_long1000_uniform_vs_smooth_20260528/`. | `45.8491` | `594.97` | `0.625` | Formal BC reference in the active plan. |
| Matched seed0 BC video reference | `rerun_g1_bc_matched_roll10_20260530` | none | `mlp` BC | seed 0 | `scripts/outputs/mjlab_qs/policy_rollouts/rerun_g1_bc_matched_roll10_20260530/velocity_flat_unitree_g1/mlp_ref/mlp/offline/bc50k_expert_uniform_policy0k/seed_0/final/`. | `54.1283` | `688.40` | `0.400` | Video gate comparator for seed0 Flow-MBPO candidates. |
| Expert collector | `rerun_g1_collector_reference_rollouts_20260528` | native collector | native collector | seed 1 iter15000 | `scripts/outputs/mjlab_qs/native_collector_rollouts/rerun_g1_collector_reference_rollouts_20260528/velocity_flat_unitree_g1/expert_seed1_iter15000/seed_1/rollout.mp4`. | `82.6090` | `1000.00` | `0.000` | Target reference. |
| Expert-noisy collector | `rerun_g1_collector_reference_rollouts_20260528` | native collector | native collector | seed 1 iter15000 noisy | `scripts/outputs/mjlab_qs/native_collector_rollouts/rerun_g1_collector_reference_rollouts_20260528/velocity_flat_unitree_g1/expert_noisy_seed1_iter15000/seed_1/rollout.mp4`. | `80.3525` | `1000.00` | `0.000` | Stable noisy expert reference. |
| Medium collector | `rerun_g1_collector_reference_rollouts_20260528` | native collector | native collector | seed 2 iter15000 | `scripts/outputs/mjlab_qs/native_collector_rollouts/rerun_g1_collector_reference_rollouts_20260528/velocity_flat_unitree_g1/medium_seed2_iter15000/seed_2/rollout.mp4`. | `49.1935` | `653.33` | `0.667` | Medium/reference data point. |
| Random/reference | `rerun_g1_collector_reference_rollouts_20260528` | none | random smooth | seed 0 | `scripts/outputs/mjlab_qs/native_collector_rollouts/rerun_g1_collector_reference_rollouts_20260528/velocity_flat_unitree_g1/random_smooth/seed_0/rollout.mp4`. | `0.4857` | `75.33` | `1.000` | Lower-bound reference. |

## World-Model-Only Diagnostics

World-model prediction diagnostics on the same QS window dataset do not by
themselves support policy-improvement claims:

```text
scripts/outputs/mjlab_qs/status/rerun_g1_pwm_flow_wm_aggregate_latest.csv
mlp_ref:        test_rollout_dyn_mse_H16_mean 0.0251417, test_reward_mse_mean 0.0211917
flow_endpoint:  test_rollout_dyn_mse_H16_mean 0.0278401, test_reward_mse_mean 0.0220010

scripts/outputs/mjlab_qs/status/rerun_g1_pwm_flow_wm_sigreg_aggregate_latest.csv
flow_endpoint_sigreg0.05: test_rollout_dyn_mse_H16_mean 0.0244410, test_reward_mse_mean 0.0213082
```

SIGReg has a diagnostic world-model metric row, but it is not enough evidence
for policy improvement and does not satisfy the active plan's requirement to
document the objective, tensor shapes, and tests before adding new SIGReg work.

## Current Interpretation

The existing evidence supports this conservative diagnosis:

```text
upstream PWM algorithm adapter on MJLab: complete fix2 eval/video gate; final and best both collapse with fall 1.000
full upstream PWM pipeline bridge: mechanically feasible on MJLab through smoke, long diagnostic, and checkpoint real-env eval smoke, but final/best smoke collapses with fall 1.000
previous PWM-style runner: failed
Flow WM only: failed in the old 2x2 policy extraction matrix
Flow policy only: failed in the old 2x2 policy extraction matrix
Flow WM + Flow policy: failed in the old 2x2 policy extraction matrix
Flow-MBPO synthetic replay: final/best gates exist for the strongest seed0 rows, but strict matched video/fall gate is not cleared; best checkpoints often regress in eval40
Conservative broad Flow-MBPO AWR sweep and support-truncation diagnostics:
negative; every completed row falls in all 8 real-eval episodes and remains
below BC
best BC: still a hard comparator, especially matched seed0 video fall 0.400
expert and expert-noisy collectors: far above all learned policy-improvement rows
```

Next action: do not treat PWM adapter imagined gains as verified. The fix2
MJLab package now has fall-aware eval and video evidence and is a clear negative
adapter-level baseline. The new full-upstream-pipeline bridge preserves
`train_dflex.py` and `PWM.train()` orchestration and has completed a longer
diagnostic (`9401906`) plus final/best real-env eval smoke (`9401975`), but both
checkpoints collapse with fall 1.000. Do not make a positive MJLab performance
claim or submit formal video gates for this diagnostic checkpoint. The Ant DFlex
supplemental diagnostic is complete under the
locked original environment plus explicit GCC 11 `CPATH`; Hopper fix3 reached
the DFlex kernel/eval path but failed in the probe script because locked PWM
does not accept `PWM.load(..., with_buffer=False)`.
