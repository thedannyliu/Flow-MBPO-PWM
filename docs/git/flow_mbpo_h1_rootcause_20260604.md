# Flow-MBPO H1 Root-Cause Runs - 2026-06-04

## Priority

The immediate question is why the historical Flow-MBPO H1 candidate reached `return_mean=60.8721` in formal `eval40`, while later follow-up H1 AWR runs mostly fell below the BC baseline.

The root-cause plan separates four failure modes:

1. Historical artifact/config trace for the `60.87` result.
2. Whether AWR updates damage the BC policy over iterations.
3. Whether damage appears in real-only AWR or only after synthetic H1 replay is mixed in.
4. Whether the H1 synthetic replay contains reward/support/done artifacts that can pull the policy into bad real-env behavior.

## Historical Strongest H1 Artifact

Historical AWR output:

`scripts/outputs/mjlab_qs/flow_mbpo_v0_awr/flow_endpoint_seed0_h1_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0`

Historical formal eval outputs:

- final: `scripts/outputs/mjlab_qs/flow_mbpo_v0_eval/flow_endpoint_seed0_h1_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0/final/summary.json`
- best: `scripts/outputs/mjlab_qs/flow_mbpo_v0_eval/flow_endpoint_seed0_h1_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0/best/summary.json`

Key paths from the historical AWR summary:

- dataset: `scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.pt`
- metadata: `scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.json`
- normalization: `scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16_normalization.json`
- initial BC checkpoint: `scripts/outputs/mjlab_qs/policy_extraction/rerun_g1_bc_expert_uniform_mlp50k_20260528/velocity_flat_unitree_g1/mlp_ref/mlp/offline/bc50k_expert_uniform_policy0k/seed_0/final_policy_extraction.pt`
- synthetic replay: `scripts/outputs/mjlab_qs/flow_mbpo_v0_replay/flow_endpoint_ensemble_seed0_h1_unc0p5_q0p90/synthetic_replay.pt`

Important: the strongest historical run used the exact replay path above, not the later `_truncate_check` replay path.

Historical AWR settings:

- seed: `0`
- update_iters: `500`
- real_batch_size: `224`
- synthetic_batch_size: `32`
- actor_lr: `1e-5`
- advantage_source: `reward`
- adv_temperature: `1.0`
- weight_clip: `20.0`
- bc_anchor_weight: `1.0`
- real_eval_every: `250`
- real_eval_episodes: `8`
- real_eval_num_envs: `16`
- AWR summary git SHA: `b194199de14a8a666a6abf5f54d37ea200cfd969`

Historical direct eval results:

| checkpoint | eval git SHA | return | length | fall |
| --- | --- | ---: | ---: | ---: |
| final | `32c7857e3a0443a274c32c3952d7e950800cdc16` | `60.8721` | `759.30` | `0.45` |
| best_real_eval | `32c7857e3a0443a274c32c3952d7e950800cdc16` | `46.1720` | `600.60` | `0.70` |

The `60.87` number came from `final_policy_extraction.pt`, not `best_policy_extraction.pt`. The historical in-training AWR real eval was weak: `best_real_return=17.0962`. This is the main suspicious gap to re-check with fresh direct eval.

## New Manifests

Checkpoint eval:

- manifest: `scripts/experiments/mjlab_qs/manifests/flow_mbpo_h1_rootcause_checkpoint_eval_h100_20260604.csv`
- rows: `41`
- scope: BC seed 0-2, historical strongest final/best, and all completed fix1 H1 iter250/iter500/final/best checkpoints.
- output root: `scripts/outputs/mjlab_qs/flow_mbpo_h1_rootcause_checkpoint_eval_20260604/h100`

Exact-replay real-only/synthetic-ratio AWR:

- manifest: `scripts/experiments/mjlab_qs/manifests/flow_mbpo_h1_exact_replay_realonly_ratio_h200_20260604.csv`
- rows: `12`
- scope: seeds 0-2 x real/synthetic ratios `256/0`, `248/8`, `224/32`, `192/64`.
- output root: `scripts/outputs/mjlab_qs/flow_mbpo_h1_exact_replay_realonly_ratio_20260604/h200`
- replay: historical exact H1 replay, not `_truncate_check`.
- selection metric: `return`, matching the current default and the strongest-era summary behavior.

AWR damage diagnostics:

- manifest: `scripts/experiments/mjlab_qs/manifests/flow_mbpo_h1_awr_diagnostics_h100_20260604.csv`
- rows: `10`
- scope: historical strongest final/best plus fix1 H1 iter250/iter500/final/best per completed run.
- output root: `scripts/outputs/mjlab_qs/flow_mbpo_h1_awr_diagnostics_20260604/h100`
- metrics: action deviation from BC on real/synthetic rows, BC MSE on logged expert windows, synthetic reward/AWR weights/done slices.

Replay quality diagnostics:

- manifest: `scripts/experiments/mjlab_qs/manifests/flow_mbpo_h1_replay_quality_l40s_20260604.csv`
- rows: `2`
- scope: exact historical H1 replay and `_truncate_check` replay.
- output root: `scripts/outputs/mjlab_qs/flow_mbpo_h1_replay_quality_20260604/l40s`
- metrics: nearest-real support distance, synthetic reward vs nearest-real reward, nearest-real done/termination rates, high-reward synthetic slices, OOD slices.

## Validation Before Submission

Commands run:

```bash
python -m py_compile \
  scripts/experiments/mjlab_qs/build_flow_mbpo_h1_rootcause_manifests_20260604.py \
  scripts/experiments/mjlab_qs/run_flow_mbpo_awr_diagnostic_row.py \
  scripts/experiments/mjlab_qs/run_flow_mbpo_replay_quality_row.py \
  scripts/experiments/mjlab_qs/analyze_flow_mbpo_replay_quality.py \
  scripts/experiments/mjlab_qs/run_flow_mbpo_v0_awr_update.py
bash -n scripts/experiments/mjlab_qs/submit_array.sh
python scripts/experiments/mjlab_qs/build_flow_mbpo_h1_rootcause_manifests_20260604.py
```

All required input paths in the generated manifests were checked before submission.

## Submission Status

Executable code and manifests were committed in `a60d1ba99a55b3576e140ea04510b211f23743f6`; later commits in this file are documentation-only submission records. Jobs used QOS `embers`, not `inferno`.

SHA note:

- Array jobs submitted through `submit_array.sh` export `FLOW_MBPO_SUBMIT_GIT_SHA=a60d1ba99a55b3576e140ea04510b211f23743f6`.
- Sequential fallback jobs call `git rev-parse HEAD` at runtime; if they start after this documentation update, their recorded HEAD may be a later doc-only commit, with the same executable code/manifests.

| track | job ID | mode | GPU | status at submission |
| --- | --- | --- | --- | --- |
| AWR damage/action-drift diagnostics | `9419844` | array `0-9%4` | H100 | pending |
| exact-replay real-only/synthetic-ratio AWR | `9419845` | array `0-11%4` | H200 | pending |
| replay quality diagnostics | `9419848` | sequential rows `0,1` | L40S | pending |
| checkpoint direct eval | `9419850` | sequential rows `0..40` | H100 | pending |

Submission notes:

- The initial checkpoint-eval array submission was rejected by `QOSMaxSubmitJobPerUserLimit`, so it was resubmitted as one sequential H100 job.
- The initial replay-quality L40S array submission used 8 CPUs and was rejected by the L40S CPU:GPU ratio limit, so it was resubmitted as one sequential L40S job with 4 CPUs.
- A 12h sequential checkpoint-eval submission was rejected by the embers walltime limit; the accepted checkpoint-eval job uses 4h. If it times out, rerun the same manifest after completed rows are skipped by existing output checks.
