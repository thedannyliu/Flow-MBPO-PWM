# PWM, Flow-PWM, Flow-MBPO, BC, and Expert Evidence Inventory

Date: 2026-06-02

Purpose: collect the existing matched evidence rows requested by
`docs/goals/pwm_flow_sigreg_image_research_plan_20260602.md` while the required
faithful original-PWM final/best eval and rollout jobs are still pending. This
inventory is evidence-only. It does not make a policy-improvement claim.

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
| Faithful original PWM adapter | `original_pwm_adapter_phase3_formal_20260601` plus pending jobs `9388552` and `9388553` | Original PWM adapter | Original PWM extraction | seed 0 | Formal checkpoints exist: `final_policy_extraction.pt`, `best_policy_extraction.pt`; eval/video with fall metrics pending. Formal adapter summary: `scripts/outputs/mjlab_qs/original_pwm_adapter/original_pwm_adapter_phase3_formal_20260601/velocity_flat_unitree_g1/normobs_normrew/seed_0/summary.json`. | `-0.8010` | `44.45` | missing | Incomplete evidence gate. Current formal summary is collapse-like but not the required final/best fall/video package. |
| Prior PWM-style runner | `rerun_g1_pwm_flow_policy2x2_20260527` | `mlp_ref` | `mlp` | seeds 0-2 | `scripts/outputs/mjlab_qs/policy_rollouts/rerun_g1_pwm_flow_policy2x2_20260527/.../mlp_ref/mlp/offline/policy50k/` | `-4.5700` | `66.33` | `1.000` | Real rollout diagnostic only. Fails badly. |
| Flow policy only | `rerun_g1_pwm_flow_policy2x2_20260527` | `mlp_ref` | `flow` | seeds 0-2, aggregate from 3 final rollouts; status CSV says 2 of 3 completed in policy-eval aggregate | `scripts/outputs/mjlab_qs/policy_rollouts/rerun_g1_pwm_flow_policy2x2_20260527/.../mlp_ref/flow/offline/policy50k/` | `-3.4818` | `66.78` | `1.000` | Slightly less bad than MLP policy in this diagnostic, still failed. |
| Flow WM only | `rerun_g1_pwm_flow_policy2x2_20260527` | `flow_endpoint` | `mlp` | seeds 0-2 | `scripts/outputs/mjlab_qs/policy_rollouts/rerun_g1_pwm_flow_policy2x2_20260527/.../flow_endpoint/mlp/offline/policy50k/` | `-4.7720` | `60.33` | `1.000` | Failed. No evidence that Flow WM alone fixed policy extraction. |
| Flow WM + Flow policy | `rerun_g1_pwm_flow_policy2x2_20260527` | `flow_endpoint` | `flow` | seeds 0-2 | `scripts/outputs/mjlab_qs/policy_rollouts/rerun_g1_pwm_flow_policy2x2_20260527/.../flow_endpoint/flow/offline/policy50k/` | `-3.5414` | `85.33` | `0.889` | Best old 2x2 rollout row by fall, but still a collapse. |
| Flow-MBPO H1 endpoint AWR | `flow_mbpo_v0_awr_cons_r224_s32_anchor1_iter500_s0` | Flow endpoint synthetic replay, H1 | AWR policy | seed 0 | Eval: `scripts/outputs/mjlab_qs/flow_mbpo_v0_eval/flow_endpoint_seed0_h1_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0/`; rollouts: `scripts/outputs/mjlab_qs/flow_mbpo_v0_rollouts/flow_endpoint_seed0_h1_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0/`. | eval final `60.8721`; video best `55.5533` | eval final `759.30`; video best `707.60` | eval final `0.450`; video best `0.400` | Stronger than old PWM/Flow 2x2. Split final/best evidence and no claim because matched video/fall gate was not cleared. |
| Flow-MBPO trajectory/chunk H3 | `flow_trajectory_chunk_5k_seed0_h3_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0` | Flow trajectory/chunk 5k synthetic replay, H3 | AWR policy | seed 0 | Eval: `scripts/outputs/mjlab_qs/flow_mbpo_v0_eval/trajchunk_h3_awr_final_eval40_20260530/`; rollouts: `scripts/outputs/mjlab_qs/flow_mbpo_v0_rollouts/flow_trajectory_chunk_5k_seed0_h3_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0/`. | eval final `48.7296`; video final `54.4904` | eval final `637.22`; video final `694.00` | eval final `0.575`; video final `0.400` | Best documented trajectory/chunk scalar-plus-video candidate by return/length balance, but strict fall gate ties matched BC instead of improving it. |
| Flow-MBPO trajectory/chunk H3 low synthetic ratio | `flow_trajectory_chunk_5k_seed0_h3_unc0p5_q0p90_cons_r240_s16_anchor1_iter500_s0` | Flow trajectory/chunk 5k synthetic replay, H3 | AWR policy | seed 0 | Eval: `scripts/outputs/mjlab_qs/flow_mbpo_v0_eval/trajchunk_h3_awr_lowsynth_r240_s16_eval40_20260531/`; rollouts: `scripts/outputs/mjlab_qs/flow_mbpo_v0_rollouts/flow_trajectory_chunk_5k_seed0_h3_unc0p5_q0p90_cons_r240_s16_anchor1_iter500_s0/`. | eval final `47.5960`; video final `55.4222` | eval final `612.00`; video final `707.20` | eval final `0.600`; video final `0.400` | Preserves video return/length gains, still ties matched BC fall and does not justify seed expansion. |
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
faithful original PWM on MJLab: incomplete gate; formal summary is collapse-like but required fall/video jobs are pending
previous PWM-style runner: failed
Flow WM only: failed in the old 2x2 policy extraction matrix
Flow policy only: failed in the old 2x2 policy extraction matrix
Flow WM + Flow policy: failed in the old 2x2 policy extraction matrix
Flow-MBPO synthetic replay: promising seed0 diagnostics, but strict matched video/fall gate not cleared
best BC: still a hard comparator, especially matched seed0 video fall 0.400
expert and expert-noisy collectors: far above all learned policy-improvement rows
```

Next action: replace the failed infrastructure jobs. The faithful original PWM
fall-aware eval/video package remains incomplete until the `fix1` eval and
rollout replacements produce `summary.json`, `eval_episodes.csv`, and
`rollout.mp4`. The Ant and Hopper DFlex supplemental diagnostics remain
incomplete until `fix3` runs with the locked original environment plus explicit
GCC 11 `CPATH`.
