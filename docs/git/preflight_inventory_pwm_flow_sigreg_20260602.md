# PWM / Flow / SIGReg Preflight Inventory

Date: 2026-06-02

Purpose: satisfy the required preflight before any new GPU submissions for `docs/goals/pwm_flow_sigreg_image_research_plan_20260602.md`.

## Repository State

Preflight commands run from `/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM`.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD | `1ddcf2d28bd92a684b6ef09d33675e2c9198577c` |
| Dirty status before inventory | `M docs/goals/pwm_flow_sigreg_image_research_plan_20260602.md`; `docs/git/` did not exist before this inventory file |
| Recent relevant commits | `1ddcf2d Clarify Flow SIGReg image research plan`; `b5000f4 Clarify broad GPU submission policy`; `04befbe Add Flow SIGReg image research plan`; `88a4ca5 Record no-dependency PWM job resubmissions`; `9af4598 Record PWM progress and resubmitted MJLab adapter jobs`; `27129ef Record locked Hopper parity diagnosis`; `846cad4 Record sandboxed multi-GPU Hopper parity jobs`; `95527e7 Fix locked Hopper submission SHA`; `1f495b1 Record locked Hopper formal submission`; `6fbbaf6 Document locked original PWM environment`; `aac7dc8 Document PWM environment fidelity failure`; `ce4c5ad Record PWM real-env parity diagnostic` |
| Slurm commands | `squeue -u $USER`; `sacct -j 9387942,9387949,9387896,9387895 --format=... -P`; `seff` attempted but unavailable (`command not found`) |
| Artifact searches | Searched `docs/goals`, `scripts`, `configs`, `logs`, `outputs`, `eval_results`, `scripts/outputs`, and W&B/offline directories for the named job IDs, sbatch records, checkpoint/eval/video paths, and result summaries. |

## Gate Job Inventory

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9387896_0` | MJLab faithful original PWM adapter smoke | `COMPLETED`, exit `0:0`, elapsed `00:00:18` | `scripts/experiments/mjlab_qs/submit_array.sh --kind original_pwm_adapter --manifest scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_smoke_20260601.csv`; row runner `scripts/experiments/mjlab_qs/run_original_pwm_adapter_row.py` | Not logged locally; submitted in the no-dependency sequence recorded at `88a4ca5` | `original_pwm_adapter_phase3_smoke_20260601.csv`; `pretrain_iters=2`, `policy_iters=2`, `horizon=16`, `latent_dim=512`, `rew_rms=false`, `ret_rms=true`, W&B disabled | MJLab QS H16 window dataset `scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.pt`; metadata and normalization JSON beside it; project `pwm` env | `0` | H100 / `embers` | W&B disabled | `scripts/outputs/mjlab_qs/original_pwm_adapter/original_pwm_adapter_phase3_smoke_20260601/velocity_flat_unitree_g1/normobs_normrew/seed_0/{pretrained_original_pwm_adapter.pt,final_policy_extraction.pt,best_policy_extraction.pt}` | `.../seed_0/summary.json`; `.../seed_0/eval_summary.json`; no video | Real eval skipped by `--skip-real-eval`; best imagined proxy `-3.7219765`; fall not measured | None | Yes as an adapter/runtime smoke only | Use as proof that the adapter starts and writes checkpoints; not performance evidence |
| `9387895_0` | MJLab faithful original PWM adapter formal | `COMPLETED`, exit `0:0`, elapsed `00:49:21` | `scripts/experiments/mjlab_qs/submit_array.sh --kind original_pwm_adapter --manifest scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_formal_h200_seed0_20260601.csv`; row runner `scripts/experiments/mjlab_qs/run_original_pwm_adapter_row.py` | `88a4ca5b30a224f0df72ca4994b2ae19a480bf2a` from W&B metadata | `original_pwm_adapter_phase3_formal_h200_seed0_20260601.csv`; `pretrain_iters=50000`, `policy_iters=15000`, `horizon=16`, `wm_batch_size=256`, `policy_batch_size=64`, `num_critics=3`, `critic_iterations=8`, `critic_batches=4`, `rew_rms=false`, `ret_rms=true`, W&B enabled | Same MJLab QS H16 core dataset; metadata reports `num_windows=351051`, `num_episodes=1562`, quality IDs random_smooth/medium/expert/expert_noisy; project `pwm` env, Python `3.10.19`, CUDA `13.2` on H200 | `0` | H200 / `embers` | `https://wandb.ai/danny010324/flow-mbpo-mjlab-original-pwm-adapter/runs/17tlyzo2`; local W&B dir `/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/scripts/outputs/mjlab_qs/wandb/wandb/run-20260602_022355-17tlyzo2` | `scripts/outputs/mjlab_qs/original_pwm_adapter/original_pwm_adapter_phase3_formal_20260601/velocity_flat_unitree_g1/normobs_normrew/seed_0/{pretrained_original_pwm_adapter.pt,final_policy_extraction.pt,best_policy_extraction.pt}` | `.../seed_0/summary.json`; `.../seed_0/eval_summary.json`; no video yet | 40-episode eval return `-0.800985`, length `44.45`; fall not logged by this adapter eval; best imagined proxy `12.491907` at iter `14984` | None; behavior indicates real-env collapse despite improved imagined proxy | Usable as negative faithful-PWM MJLab evidence; not enough for final claim because fall and video are missing | Submit final and best checkpoint 40-episode real eval with fall metrics plus final/best 10-episode 1000-step MP4/W&B videos |
| `9387942` | Ant final/best true DFlex real-env eval for original PWM parity | `FAILED`, exit `1:0`, elapsed `00:00:26` | Wrapped Slurm eval using `scripts/eval/eval_online_single_task.py` against Ant final/best actors from job `9384344` | Current wrapper provenance not logged in output; resubmission was recorded after `88a4ca5` | Original DFlex Ant config from PWM parity path; outputs intended under `eval_results/pwm_phase1_ant_locked_h200_realenv_{final,best}_20260602` | Locked original env `/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4`; DFlex sandbox copied from locked env site-packages | `0` actor seed; eval seed not confirmed from log | H200 / `embers` | W&B not used | Inputs: Ant `final_policy.pt` and `best_policy.pt` from `baselines/PWM/scripts/outputs/2026-06-01/23-06-24/logs/phase1_ant_formal_locked_h100_s0_20260601/`; no new eval checkpoint | Slurm logs `logs/pwm_original_parity/locked_env_20260601/pwm_ant_locked_realenv_eval_h200_fix_9387942.{out,err}`; no eval outputs written | Not available | DFlex kernel rebuild failed because the locked env compiler wrappers could not execute `cc1plus` (`x86_64-conda-linux-gnu-c++: fatal error: cannot execute 'cc1plus'`) | No, infrastructure failure only | Repair locked-env compiler or reuse a node/cache path with valid DFlex kernels; resubmit Ant final/best true eval after documenting the fix |
| `9387949` | Hopper final/best WM-vs-real probe for original PWM Phase 2 | `FAILED`, exit `1:0`, elapsed `00:00:26` | Wrapped Slurm diagnostic using `scripts/diagnostics/pwm_dflex_checkpoint_probe.py` against Hopper final/best actors from job `9383814` | Current wrapper provenance not logged in output; resubmission was recorded after `88a4ca5` | Original DFlex Hopper probe config; outputs intended under `eval_results/pwm_phase2_hopper_locked_probe_20260602/` | Locked original env `/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4`; DFlex sandbox copied from locked env site-packages | `0` actor seed; eval seed not confirmed from log | H100 / `embers` | W&B not used | Inputs: Hopper `final_policy.pt` and `best_policy.pt` from the locked Hopper formal run; no new probe outputs | Slurm logs `logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_wmprobe_h100_fix_9387949.{out,err}`; no JSON probe outputs written | Not available | Same DFlex kernel rebuild failure: missing executable `cc1plus` in locked compiler path, then Hydra could not instantiate `dflex.envs.HopperEnv` | No, infrastructure failure only | Apply the same compiler/kernel-cache fix as Ant; resubmit Hopper WM-vs-real probe |

## Gate Interpretation

The MJLab formal faithful adapter completed and produced final/best checkpoints, but real eval is far below BC and expert references: return `-0.800985` and length `44.45` over 40 episodes. Because fall rate and videos are missing, this is only a conservative collapse signal, not a complete MJLab evidence package.

The imagined-return proxy increased from negative values to `12.491907` while real return remained near zero with very short episodes. Treat this as likely model exploitation or transfer/protocol mismatch until final/best eval with fall metrics, videos, and diagnostics confirm the failure mode.

The Ant true eval and Hopper WM-vs-real probe are blocked by a compiler/kernel infrastructure issue in the locked original DFlex path. They do not contradict the earlier Hopper locked-env parity result and should not be interpreted as algorithm failures.

## Immediate Required Follow-Up Before Claims

1. Submit final and best checkpoint 40-episode real MJLab eval for the formal adapter checkpoints.
2. Submit final and best checkpoint 10-episode, 1000-step MP4/W&B rollout videos for the same checkpoints.
3. Record return, episode length, fall rate, W&B links, checkpoint paths, and video paths in `docs/goals/pwm_fidelity_mjlab_flow_migration_20260601.md`.
4. Repair the locked DFlex compiler path or switch to a documented valid DFlex kernel build path before resubmitting jobs `9387942` and `9387949`.

## Continuation Inventory: Broad Embers GPU Wave

Preflight time: 2026-06-02 evening, America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD | `515b243` |
| Dirty status before this continuation inventory | clean |
| Recent relevant commits | `515b243 Record H200 Flow MBPO AWR results`; `bd2773a Record official image backup submissions`; `d473642 Record embers shard backfill`; `0a679b3 Document Flow SIGReg preflight policy`; `6f82e7d Record continuation Slurm poll`; `12fd08e Record broad shard submission results`; `a2a7aac Add broad Flow MBPO shard submissions`; `25a7438 Record broad embers GPU submissions` |
| Slurm commands | `squeue -u $USER`; `sacct -j 9387942,9387949,9387896,9387895,9399798,9399799,9400409,9400411,9400412,9400435,9400442,9400525,9400528,9400532,9400537,9400543,9400544,9400545 --format=... -P`; `seff` still unavailable on PATH |
| Artifact searches | Searched `docs/git`, `docs/goals`, `scripts`, `logs/slurm`, `logs/pwm_original_parity`, `scripts/outputs/mjlab_qs`, `eval_results`, and local W&B directories for current job IDs, sbatch records, checkpoints, summaries, videos, and W&B/offline dirs. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9400410_[0-13]` | Broad Flow-MBPO AWR diagnostics on the full H200 manifest | All 14 elements `COMPLETED`, exit `0:0` | `scripts/experiments/mjlab_qs/submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_20260602.csv --gpu-type H200 --partition gpu-h200 --qos embers --max-concurrent 8` | submitted after `3a3e161`; result recorded at `515b243` | `flow_mbpo_broad_embers_awr_20260602.csv`; endpoint H=1/3/5, trajectory H3 fall/support-risk, residual H=3/5; real eval every 10 updates; W&B disabled | Project `pwm` env; MJLab QS H16 core dataset, BC seed0 policy, existing synthetic replay buffers and support calibration artifacts | seeds `0` and `1` by manifest row | H200 / `embers` | W&B disabled | Per-row `final_policy_extraction.pt`, `best_policy_extraction.pt`, `best_training_loss_policy_extraction.pt`, `final_q_critic.pt`, and real-eval snapshots under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_20260602/*/` | Per-row `summary.json`; no formal MP4 videos | Best row by selection and return: `9400410_8`, iter 10, return `25.969`, length `360.0`, fall `1.0`; baseline return `45.8491`, length `594.97`, fall `0.625` | None; valid negative result | Usable as diagnostic evidence that this conservative AWR sweep did not fix MJLab falls | Do not expand the exact setting blindly; compare pending A100/L40S/H100 shards, then pivot to stronger pessimism/OOD gating or shorter-horizon rollouts if confirmed |
| `9400436_[0-3]` | H200 shard duplicate of selected Flow-MBPO AWR rows with independent output roots | All 4 elements `COMPLETED`, exit `0:0` | `scripts/experiments/mjlab_qs/submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_h200_20260602.csv --gpu-type H200 --partition gpu-h200 --qos embers --max-concurrent 4` | submitted after `a2a7aac`; result recorded at `515b243` | `flow_mbpo_broad_embers_awr_h200_20260602.csv`; endpoint H=1/3/5 and trajectory H3 support-risk rows | Same project env and MJLab QS H16 data; independent output root `flow_mbpo_broad_embers_awr_shards_20260602/h200/` | seed `0` rows | H200 / `embers` | W&B disabled | Per-row policy and critic checkpoints under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_shards_20260602/h200/*/` | Per-row `summary.json`; no formal MP4 videos | Best sampled return about `22.945`, length `333.75`, fall `1.0`; no row beat BC baseline | None; valid negative result | Usable as H200 shard confirmation | Wait for lower-priority shards before making final branch decision |
| `9400525_[0-2]` | L40S shard backfill for Flow-MBPO AWR rows not covered by earlier successful submissions | `PENDING`, reason `Priority` | `submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_l40s_20260602.csv --gpu-type L40S --partition gpu-l40s --qos embers --cpus 4` | `d473642` submission record; current status checked at `515b243` | `flow_mbpo_broad_embers_awr_l40s_20260602.csv`; trajectory fall-penalized and residual rows | Project `pwm` env; same MJLab QS H16 data and synthetic replay inputs | seed `1` rows | L40S / `embers` | W&B disabled | Expected under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_shards_20260602/l40s/*/` | None yet | Not available | None yet | Pending only | Inspect logs/results once started; replace only if runtime failure appears |
| `9400528_[1-2]` | A100 remaining Flow-MBPO AWR shard rows after row 0 was submitted as `9400442` | `PENDING`, reason `Priority` | Manual `sbatch --array=1-2%2` running `scripts/experiments/mjlab_qs/run_flow_mbpo_awr_row.py --manifest flow_mbpo_broad_embers_awr_a100_20260602.csv` | `d473642` submission record; current status checked at `515b243` | `flow_mbpo_broad_embers_awr_a100_20260602.csv`; endpoint H5 and trajectory support-risk rows | Project `pwm` env; same MJLab QS H16 data and synthetic replay inputs | seed `1` rows | A100 / `embers` | W&B disabled | Expected under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_shards_20260602/a100/*/` | None yet | Not available | None yet | Pending only | Inspect logs/results once started; avoid duplicating row 0 while `9400442` remains pending |
| `9400442` | A100 single-row Flow-MBPO AWR shard backfill | `PENDING`, reason `Priority` | Manual single-row `sbatch` against `flow_mbpo_broad_embers_awr_a100_20260602.csv`, row 0 | `12fd08e` / `d473642` records | A100 shard manifest row 0 | Project `pwm` env; same MJLab QS H16 data and replay inputs | seed `1` row | A100 / `embers` | W&B disabled | Expected under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_shards_20260602/a100/endpoint_h3_trunc_cql_mixed_s1/` | None yet | Not available | None yet | Pending only | Inspect after start; do not resubmit the same output root |
| `9400435_[0-3]` | H100 Flow-MBPO AWR shard backfill | `PENDING`, reason `Priority` | `submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_h100_20260602.csv --gpu-type H100 --partition gpu-h100 --qos embers --max-concurrent 4` | submitted after `a2a7aac`; current status checked at `515b243` | `flow_mbpo_broad_embers_awr_h100_20260602.csv` | Project `pwm` env; same MJLab QS H16 data and replay inputs | seed `0` rows | H100 / `embers` | W&B disabled | Expected under `flow_mbpo_broad_embers_awr_shards_20260602/h100/*/` | None yet | Not available | None yet | Pending only | Inspect after start; use as cross-partition confirmation |
| `9400409_[0-15]` | NEWT official broad smoke on A100 | `PENDING`, reason `Priority` | `scripts/experiments/image_official/submit_newt_lewm_broad_gpu_20260602.sh` NEWT array payload | `25a7438` / `bd2773a` records | Official `newt/tdmpc2/train.py`; tasks walker-walk, walker-run, cheetah-run, hopper-hop, reacher-easy, pendulum-swingup, cartpole-swingup, cup-catch; seeds 0 and 1; `steps=500`, `model_size=B`, W&B disabled | Official NEWT repo `/storage/project/r-agarg35-0/eliu354/external_repos/newt`; official env `/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602`; marker `.newt_official_setup_ok_20260602` | seeds `0`, `1` | A100 / `embers` | W&B disabled | No checkpoint expected (`save_agent=false`) | Logs expected under `logs/slurm/image_official/newt_official_broad_smoke_9400409_*.out`; no logs yet | Not available | None yet | Pending only | Wait; if A100 stays pending, H200 backup rows provide earlier signal |
| `9399799` | NEWT official walker smoke on A100 after SWIG setup repair | `PENDING`, reason `Priority` | `scripts/experiments/image_official/submit_newt_official_swig_followups_20260602.sh` walker payload | `c7c33b` / later records | Official NEWT `train.py`, walker-walk smoke, W&B disabled | Same official NEWT repo/env as broad smoke | seed not separately recorded in this inventory | A100 / `embers` | W&B disabled | No checkpoint expected | Log expected under `logs/slurm/image_official/newt_official_walker_swig_fix1_9399799.*`; no logs yet | Not available | None yet | Pending only | Wait; cancel only if a later completed broad row proves the payload is invalid |
| `9400537_[0-3]` | NEWT official H200 backup rows 0-3 | `PENDING`, reason `Resources` | Manual H200 backup `sbatch --array=0-3%4` using official NEWT `train.py` | `bd2773a` submission record | Same NEWT official broad payload with unique `exp_name=official_broad_h200_chunk0_*` | Same official NEWT repo/env; data dir `/storage/project/r-agarg35-0/eliu354/external_data/newt_demos` | rows 0-3 | H200 / `embers` | W&B disabled | No checkpoint expected | Logs expected under `logs/slurm/image_official/newt_official_broad_h200_chunk0_9400537_*.out`; no logs yet | Not available | None yet | Pending only | Wait for H200 resources; remaining rows 7-15 are unsent due submit quota |
| `9400543_[4]`, `9400544_[5]`, `9400545_[6]` | NEWT official H200 backup rows 4-6 | `PENDING`, reason `Resources` | Manual single-row H200 `sbatch --array=<row>` using official NEWT `train.py` | `bd2773a` submission record | Same NEWT official payload with unique `exp_name=official_broad_h200_row*_*` | Same official NEWT repo/env | rows 4, 5, 6 | H200 / `embers` | W&B disabled | No checkpoint expected | Logs expected under `logs/slurm/image_official/newt_official_h200_row{4,5,6}_*.out`; no logs yet | Not available | `row7` submission failed with `QOSMaxSubmitJobPerUserLimit` | Pending only | Retry rows 7-15 only after submit slots free |
| `9400411_[0-5]` | LeWM official PushT eval on H100 | `PENDING`, reason `Priority` | `scripts/experiments/image_official/submit_newt_lewm_broad_gpu_20260602.sh` LeWM eval payload | `25a7438` / `bd2773a` records | Official `eval.py --config-name=pusht`; policies `pusht/lewm` and `random`; horizons 2 and 5; `eval.num_eval=4`, `eval.eval_budget=30` | Official LeWM repo `/storage/project/r-agarg35-0/eliu354/external_repos/le-wm`; env `/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602`; assets under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm` | seeds 0, 1, 2 by row | H100 / `embers` | W&B disabled | Uses converted checkpoint `external_data/lewm_stablewm/pusht/lewm_object.ckpt`; no training checkpoint expected | Logs expected under `logs/slurm/image_official/lewm_official_pusht_eval_9400411_*.out`; no logs yet | Not available | None yet | Pending only | Wait; H200 backup `9400532` may finish first |
| `9400412_[0-1]` | LeWM official PushT train smoke on H100 | `PENDING`, reason `Priority` | `scripts/experiments/image_official/submit_newt_lewm_broad_gpu_20260602.sh` LeWM train payload | `25a7438` / `bd2773a` records | Official `train.py data=pusht`; max 1 epoch, 2 train batches, 1 val batch; W&B disabled | Same official LeWM repo/env/assets | seeds 0 and 1 | H100 / `embers` | W&B disabled | Expected official training output under LeWM output dirs if the smoke starts | Logs expected under `logs/slurm/image_official/lewm_official_pusht_train_smoke_9400412_*.out`; no logs yet | Not available | None yet | Pending only | Wait; inspect quickly once started because official Hydra args may need adjustment |
| `9400532_[0-5]` | LeWM official PushT eval H200 backup | `PENDING`, reason `Resources` | Manual H200 backup `sbatch --array=0-5%6` using official LeWM `eval.py` | `bd2773a` submission record | Same LeWM eval payload as `9400411`, with output filenames suffixed `_h200` | Same official LeWM repo/env/assets | seeds 0, 1, 2 by row | H200 / `embers` | W&B disabled | Uses `external_data/lewm_stablewm/pusht/lewm_object.ckpt` | Logs expected under `logs/slurm/image_official/lewm_official_pusht_eval_h200_9400532_*.out`; no logs yet | Not available | None yet | Pending only | Wait for H200 resources; use as first official LeWM signal if it starts before H100 |

Continuation inventory after LeWM repair and L40S backfill:

```text
poll_time: 2026-06-02 19:05-19:08 America/New_York.
git_head_before_record: 86df949.
git_status_before_record: uncommitted official-image scripts/docs from the prior submission pass were present; they were preserved and extended rather than reverted.
seff: unavailable on PATH; status used `squeue` and `sacct`.
9400532_0..5: LeWM H200 eval backup FAILED 1:0 before evaluation because official LeWM imported old HuggingFace `datasets` code that expected `pyarrow.PyExtensionType`.
9400411 and 9400412: canceled before start to avoid the same known LeWM environment failure on H100.
LeWM repair: official env now uses `datasets 3.6.0`, `pyarrow 24.0.0`, and a batch-visible `PyExtensionType` compatibility shim; local import/config smoke passed for the actual `stable_worldmodel.policy.AutoCostModel` and `eval.py --config-name=pusht --cfg job` entrypoint.
9400525_0..2: L40S Flow-MBPO AWR shard COMPLETED 0:0; final returns were 11.3848, 10.2740, and 12.9974 with fall_rate_mean 1.0, confirming the conservative AWR setting remains a valid negative result.
9400714: NEWT official broad L40S accepted on embers, array 0-15%4; rows 0 and 1 completed successfully with valid train/eval output.
9400715: LeWM official PushT eval L40S accepted on embers, array 0-5%3, pending at record time.
9400716: LeWM official PushT train L40S accepted on embers, array 0-1%2, pending at record time.
replacement_candidate: `scripts/experiments/image_official/submit_newt_h200_remaining_lewm_h200_fix_20260602.sh` for remaining NEWT H200 rows and repaired LeWM H200 eval/train rows.
next_action: commit this repair/submission record, then submit repaired H200 jobs if the embers submit quota accepts them.
```

Post-commit H200 repair submission inventory:

```text
commit_before_submission: ce439b0.
large_array_result: committed H200 repair wrapper failed with `QOSMaxSubmitJobPerUserLimit`; no rows were accepted by that array attempt.
single_row_submit_probe: one-row H200 and L40S `sbatch --test-only` probes succeeded, so replacement jobs were submitted as single-row arrays.
9400771_[0] through 9400776_[5]: repaired LeWM official PushT H200 eval rows accepted on embers; pending Resources at first check; unique logs under `logs/slurm/image_official/lewm_official_pusht_eval_h200_fix_single_%A_%a.*`.
9400778_[7]: remaining NEWT H200 row 7 accepted on embers; pending Resources at first check; unique logs under `logs/slurm/image_official/newt_official_h200_remaining_single_%A_%a.*`.
blocked_by_quota: NEWT H200 row 8 failed with `QOSMaxSubmitJobPerUserLimit`; rows 8-15 and LeWM H200 train-smoke singles were not submitted in this pass.
concurrent_status: 9400714_0..6 completed 0:0 on L40S with valid NEWT official output; 9400715 and 9400716 remained pending Priority; 9400435_0 H100 Flow-MBPO AWR was running.
next_action: monitor 9400771..9400776 as the repaired LeWM Slurm proof; retry NEWT rows 8-15 and LeWM train singles when submit slots free.
```
| `9399798` | Hopper locked WM-vs-real probe fix4 | `PENDING`, reason `Priority` | Repaired locked DFlex Slurm wrapper for Hopper final/best WM-vs-real probe | `c7c33b` / later records | Hopper locked original PWM probe with DFlex sandbox and compiler include repair | Locked env `/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4`; Hopper final/best actors from locked parity run | seed 0 actor | H100 / `embers` | W&B disabled | Inputs from locked Hopper formal run; expected `eval_results/pwm_phase2_hopper_locked_probe_20260602/*fix4*.json` | No output yet | Not available | None yet | Pending only | Wait; if it fails, record root cause and resubmit only after compiler/path diagnosis |

Current continuation interpretation: the queue is saturated with useful embers GPU work across A100, H100, H200, and L40S. The only completed new broad wave so far is H200 Flow-MBPO AWR, which is a valid negative result: all runs fall at rate `1.0` and do not beat BC. No current pending image/MJLab backup job has produced logs yet, so there is no runtime failure to cancel or repair at this poll.

## Continuation Inventory: NEWT/LeWM and Hopper Probe Update

Preflight time: 2026-06-02 19:10-19:13 America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD | `ce439b026cc06760db6e90912768b606bd3628b9` |
| Dirty status before this continuation inventory | clean |
| Recent relevant commits | `ce439b0 Record LeWM repair and L40S official submissions`; `86df949 Record embers submit quota probe`; `a81bc59 Update broad embers preflight inventory`; `515b243 Record H200 Flow MBPO AWR results`; `bd2773a Record official image backup submissions`; `d473642 Record embers shard backfill`; `0a679b3 Document Flow SIGReg preflight policy`; `6f82e7d Record continuation Slurm poll` |
| Slurm commands | `squeue -u $USER`; `sacct -X -j 9387942,9387949,9387896,9387895,9399798,9400409,9400435,9400442,9400528,9400714,9400715,9400716 --format=...`; `seff` unavailable on PATH |
| Artifact searches | Searched `docs/git`, `docs/goals`, `scripts`, `logs/slurm`, `logs/pwm_original_parity`, `scripts/outputs`, `eval_results`, `outputs`, and W&B/offline directories for current job IDs, checkpoints, summaries, videos, and W&B/offline dirs. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9399798` | Hopper locked WM-vs-real probe fix4 | `COMPLETED`, exit `0:0`, elapsed `00:01:19` | `scripts/experiments/mjlab_qs/submit_hopper_wmprobe_fix4_20260602.sh` running `scripts/diagnostics/pwm_dflex_checkpoint_probe.py` for final and best locked Hopper actors | wrapper repaired before `c7c33b`; result recorded after `ce439b0` | Original locked Hopper DFlex probe, 64 envs, 128 steps, policy actor, W&B disabled | Locked env `/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4`; job-local DFlex sandbox `/tmp/dflex_hopper_probe_sandbox_9399798_fix4`; Hopper actors from locked parity run | actor seed 0; probe seed 0 | H100 / `embers` | W&B disabled | `baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601/{final_policy.pt,best_policy.pt}` | `eval_results/pwm_phase2_hopper_locked_probe_20260602/{final_actor_wm_vs_real_fix4.json,best_actor_wm_vs_real_fix4.json}`; Slurm logs `logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_wmprobe_h100_fix4_9399798.{out,err}` | Final actor: real_reward_mean `3.9009`, normalized WM-real corr `0.999928`, normalized MAE `0.004424`, termination/truncation `0.0`; best actor: real_reward_mean `3.9188`, normalized WM-real corr `0.999961`, normalized MAE `0.003968`, termination/truncation `0.0` | None | Yes as locked Hopper WM-vs-real diagnostic evidence | Record in the active fidelity doc; keep Ant true-env repair separate |
| `9400714_0-6` | NEWT official broad L40S rows 0-6 | Rows 0-6 `COMPLETED`, exit `0:0`; rows 7-15 pending/running by queue state | `scripts/experiments/image_official/submit_newt_lewm_l40s_backups_20260602.sh` NEWT array payload | `ce439b0` | Official `newt/tdmpc2/train.py`, `model_size=B`, `steps=500`, W&B/video/checkpoint disabled | Official NEWT repo/env; marker `/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602/.newt_official_setup_ok_20260602`; data dir `/storage/project/r-agarg35-0/eliu354/external_data/newt_demos` | seed 0 rows for walker-walk, walker-run, cheetah-run, hopper-hop, reacher-easy, pendulum-swingup, cartpole-swingup | L40S / `embers` | W&B disabled | No checkpoint expected (`save_agent=false`) | Logs under `logs/slurm/image_official/newt_official_broad_l40s_9400714_*.out` | Eval/train rewards by row: 0 `42.247/51.202`; 1 `42.179/23.809`; 2 `5.871/6.516`; 3 `0.0/0.0`; 4 `34.000/0.0`; 5 `0.0/0.0`; 6 `183.107/8.803` | None | Yes as official NEWT smoke coverage, not performance evidence | Continue monitoring remaining L40S rows; keep H200 rows 7-15 as a useful backup candidate |
| `9400715_[0-5]` | LeWM official PushT eval L40S with repaired env/shim | `PENDING`, reason `Priority` | `scripts/experiments/image_official/submit_newt_lewm_l40s_backups_20260602.sh` LeWM eval payload | `ce439b0` | Official `eval.py --config-name=pusht`; policies `pusht/lewm` and `random`; horizons 2/5; `eval.num_eval=4`, `eval.eval_budget=30`; W&B disabled | Official LeWM env `/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602`; `datasets 3.6.0`, `pyarrow 24.0.0`; compatibility shim `scripts/experiments/image_official/compat/sitecustomize.py`; PushT checkpoint and h5 dataset under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm` | seeds 0, 1, 2 by row | L40S / `embers` | W&B disabled | Uses `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/pusht/lewm_object.ckpt` | Logs expected under `logs/slurm/image_official/lewm_official_pusht_eval_l40s_9400715_*.{out,err}` | Not available | None yet | Pending only | Inspect immediately when the first row starts because it exercises the LeWM pyarrow repair under Slurm |
| `9400716_[0-1]` | LeWM official PushT train smoke L40S with repaired env/shim | `PENDING`, reason `Priority` | `scripts/experiments/image_official/submit_newt_lewm_l40s_backups_20260602.sh` LeWM train payload | `ce439b0` | Official `train.py data=pusht`, `trainer.max_epochs=1`, 2 train batches, 1 val batch, W&B disabled | Same repaired official LeWM env/assets as `9400715` | seeds 0 and 1 | L40S / `embers` | W&B disabled | Expected official train smoke output under LeWM output dirs | Logs expected under `logs/slurm/image_official/lewm_official_pusht_train_l40s_9400716_*.{out,err}` | Not available | None yet | Pending only | Inspect after eval rows start or finish; replace only if the repaired env still fails |

Submit probe after this inventory:

```text
command: sbatch --test-only --job-name=h200_submit_quota_probe_20260602_cont --account=gts-agarg35 --partition=gpu-h200 --qos=embers --gres=gpu:h200:1 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:05:00 --wrap=true
result: accepted by scheduler test; example predicted start `2026-06-05T16:06:06` on gpu-h200.
candidate_to_submit_after_commit: `scripts/experiments/image_official/submit_newt_h200_remaining_lewm_h200_fix_20260602.sh`, covering remaining NEWT H200 rows 7-15 plus repaired LeWM H200 eval/train rows.
```

Additional post-commit submission inventory:

```text
commit_before_submission: a0a0751.
accepted_h200_newt_rows: 9400797_[8], 9400798_[9], 9400799_[10], and 9400800_[11], all gpu-h200 / embers, pending Resources at first check.
blocked_h200_newt_rows: row 12 failed with `QOSMaxSubmitJobPerUserLimit`; rows 12-15 remain unsent.
already_pending_h200_repair_rows: 9400771_[0]..9400776_[5] repaired LeWM eval rows and 9400778_[7] NEWT row 7 remain pending Resources.
new_l40s_results: 9400714_7, 9400714_8, and 9400714_9 completed 0:0 with valid official NEWT output; eval/train rewards were `0.0/0.0`, `18.338/37.122`, and `18.157/28.181`.
lewm_gpu_repair_status: no repaired LeWM GPU row has started yet; 9400715/9400716 remain pending Priority and 9400771..9400776 remain pending Resources.
next_action: monitor LeWM repaired rows immediately after they start; retry NEWT H200 rows 12-15 and LeWM H200 train singles only after submit quota frees again.
```

Post-903642d submission inventory:

```text
accepted_h200_newt_rows: 9400814_[12], 9400815_[13], 9400816_[14], and 9400817_[15], all gpu-h200 / embers, pending Resources at first check.
newt_h200_remaining_status: rows 7-15 are now all accepted; do not submit duplicate NEWT H200 remaining rows.
lewm_eval_h200_status: 9400771_0 started running on gpu-h200 / embers; initial stdout had only the Slurm prolog and no immediate pyarrow failure signature.
still_missing_h200: LeWM train-smoke singles for seeds 0 and 1.
next_action: monitor 9400771_0 for the repaired LeWM eval proof; if no new root cause appears and submit quota permits, submit only the LeWM H200 train-smoke singles.
```

LeWM HDF5 plugin repair inventory:

```text
failed_job: 9400771_0, H200 / embers, FAILED 1:0 after 00:01:29.
failure_reason: `stable_worldmodel.data.HDF5Dataset` missing because optional `hdf5plugin` was absent from the read-only official LeWM env.
affected_jobs_canceled: 9400772_[1]..9400776_[5], 9400715, 9400716.
artifact_check: `pusht_expert_train.h5` can read scalar/state/action columns without plugins, but `pixels[0]` fails without HDF5 plugin filters and succeeds after importing repo-local `hdf5plugin`.
repair_files: `.gitignore`; `scripts/experiments/image_official/compat/sitecustomize.py`; `scripts/experiments/image_official/install_lewm_compat_vendor_20260602.sh`.
local_vendor: `scripts/experiments/image_official/compat/vendor/`, intentionally untracked because it is a large binary dependency cache.
validation: `HDF5Dataset` visible on `stable_worldmodel.data`; `pixels[0]` shape `(224, 224, 3)` read successfully under the LeWM env with `PYTHONPATH` pointing at the compat root.
next_action: commit the repair, then resubmit LeWM eval/train replacement rows if embers submit quota permits.
```

## Continuation Inventory: Current Preflight Refresh

Preflight time: 2026-06-02 after commit `f923be5`, America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD | `f923be59b563498e3e9dc65ddcb2d06a6044201a` |
| Dirty status before this inventory edit | clean |
| Recent relevant commits | `f923be5 Record LeWM HDF5 compatibility repair`; `903642d Record Hopper probe and H200 NEWT singles`; `a0a0751 Record H200 image repair submissions`; `ce439b0 Record LeWM repair and L40S official submissions`; `86df949 Record embers submit quota probe`; `a81bc59 Update broad embers preflight inventory`; `515b243 Record H200 Flow MBPO AWR results`; `bd2773a Record official image backup submissions`; `d473642 Record embers shard backfill`; `0a679b3 Document Flow SIGReg preflight policy`; `6f82e7d Record continuation Slurm poll`; `12fd08e Record broad shard submission results` |
| Slurm commands | `squeue -u $USER -o ...` failed with `slurm_load_jobs error: Unable to contact slurm controller (connect failure)`; `sacct -j ... --format=... -P` failed with SlurmDB connection errors to `sched-phoenix-slurmdb:6819`; `seff` unavailable on PATH |
| Artifact searches | Re-searched `docs/git`, `docs/goals`, `scripts`, `configs`, `logs/slurm`, `logs/pwm_original_parity`, `scripts/outputs`, `eval_results`, `outputs`, checkpoints, videos, and local W&B directories for gate job IDs, later replacements, summaries, videos, and failure roots. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9387895_0` | Faithful original PWM MJLab formal checkpoint source | `COMPLETED`, exit `0:0` from earlier Slurm evidence | `scripts/experiments/mjlab_qs/submit_array.sh --kind original_pwm_adapter --manifest scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_formal_h200_seed0_20260601.csv` | `88a4ca5b30a224f0df72ca4994b2ae19a480bf2a` in W&B metadata | `original_pwm_adapter_phase3_formal_h200_seed0_20260601.csv` | Project `pwm` env; MJLab QS H16 G1 dataset and normalization files under `scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/` | `0` | H200 / `embers` | `https://wandb.ai/danny010324/flow-mbpo-mjlab-original-pwm-adapter/runs/17tlyzo2` | `scripts/outputs/mjlab_qs/original_pwm_adapter/original_pwm_adapter_phase3_formal_20260601/velocity_flat_unitree_g1/normobs_normrew/seed_0/{final_policy_extraction.pt,best_policy_extraction.pt}` | Formal `summary.json` and `eval_summary.json`; complete fix2 eval/video artifacts are recorded in rows `9395746` and `9396189` below | Formal real eval return about `-0.801`, length `44.45`; formal adapter did not log fall | None; policy collapses in real MJLab | Usable as checkpoint source and negative imagined-vs-real evidence | Keep final/best fix2 eval/video package as the claim gate; do not make imagined-return claims |
| `9395746_[0-1]` | Faithful original PWM final/best 40-episode MJLab eval with fall metrics | `COMPLETED`, exit `0:0` from artifacts and prior Slurm record | `submit_array.sh --kind policy_eval --manifest scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_eval40_final_best_fix2_20260602.csv` | Artifact summaries record `3b1a42fa4457e9152db16e649aa7b695351b89b8` | Fix2 eval manifest; `checkpoint_format=original_pwm_adapter`; 40 episodes, 16 envs, max 1000 steps | Project `pwm` env with `PYTHONPATH=src:baselines/PWM/src`; same MJLab QS H16 dataset/metadata/normalization | `0` | H200 / `embers` | W&B project `flow-mbpo-mjlab-original-pwm-adapter-eval40` | Final/best extracted checkpoints from `9387895_0` | `scripts/outputs/mjlab_qs/policy_evals/original_pwm_adapter_phase3_eval40_fix2_20260602/velocity_flat_unitree_g1/normobs_normrew/seed_0/{final,best}/summary.json` and `eval_episodes.csv` | Final return `-0.8009908`, length `44.45`, fall `1.0`; best return `-0.7777858`, length `44.45`, fall `1.0` | None; algorithm/runtime produced a valid collapse result | Yes, complete negative faithful-PWM MJLab eval gate | Use as baseline collapse evidence; compare only against real eval/video rows |
| `9396189_[0-1]` | Faithful original PWM final/best 10-episode 1000-step MJLab MP4 videos | `COMPLETED`, exit `0:0` from artifacts and prior Slurm record | `submit_array.sh --kind policy_rollout --manifest scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_rollout10_final_best_fix2_20260602.csv` | Artifact summaries record `5bef39a3860afeedd727d7d40c6b63f6b36f1917` | Fix2 rollout manifest; `checkpoint_format=original_pwm_adapter`; 10 episodes, max 1000 steps | Project `pwm` env; same MJLab QS H16 dataset/metadata/normalization | `0` | H200 / `embers` | W&B project `flow-mbpo-mjlab-original-pwm-adapter-rollout1000` | Final/best extracted checkpoints from `9387895_0` | `scripts/outputs/mjlab_qs/policy_rollouts/original_pwm_adapter_phase3_rollout10_fix2_20260602/velocity_flat_unitree_g1/normobs_normrew/seed_0/{final,best}/rollout.mp4` and `summary.json` | Final video return `-0.5264854`, length `46.8`, fall `1.0`; best video return `-0.5217915`, length `46.4`, fall `1.0` | None; valid video collapse evidence | Yes, complete negative faithful-PWM MJLab video gate | Keep as the faithful adapter visual evidence package |
| `9394869` | Ant final/best true DFlex eval repair | `COMPLETED`, exit `0:0`, elapsed `00:02:05` from logs/artifacts | `scripts/experiments/mjlab_qs/submit_original_dflex_gate_fix3_20260602.sh` | Submitted after `df4847b`; artifacts are authoritative | Locked Ant DFlex eval, final and best actors, 40 episodes | Locked original PWM env `/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4`; job-local DFlex sandbox and GCC 11 `CPATH` repair | actor seed `0` | H200 / `embers` | Final W&B run `https://wandb.ai/danny010324/flow-mbpo-pwm-fidelity/runs/wfyrsutm`; best W&B not re-read in this pass | Ant final/best actors from `baselines/PWM/scripts/outputs/2026-06-01/23-06-24/logs/phase1_ant_formal_locked_h100_s0_20260601/` | `eval_results/pwm_phase1_ant_locked_h200_realenv_{final,best}_20260602_fix3/eval_summary.json` | Final return `7570.4829`, length `1000.0`; best return `7591.1702`, length `1000.0`; fall not applicable in DFlex summary | None | Yes as supplemental original-PWM DFlex parity evidence | Preserve; it does not answer MJLab transfer |
| `9399798` | Hopper locked WM-vs-real probe fix4 | `COMPLETED`, exit `0:0`, elapsed `00:01:19` from logs/artifacts | `scripts/experiments/mjlab_qs/submit_hopper_wmprobe_fix4_20260602.sh` | Wrapper repaired before `c7c33b`; result recorded in later commits | Locked Hopper probe, 64 envs, 128 steps, final and best actors | Locked original PWM env `/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4`; job-local DFlex sandbox | actor seed `0`, probe seed `0` | H100 / `embers` | W&B disabled | Hopper final/best actors from `baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601/` | `eval_results/pwm_phase2_hopper_locked_probe_20260602/{final_actor_wm_vs_real_fix4.json,best_actor_wm_vs_real_fix4.json}` | Final real reward `3.9009`, WM-real normalized corr `0.999928`, MAE `0.004424`, term/trunc `0.0`; best real reward `3.9188`, corr `0.999961`, MAE `0.003968`, term/trunc `0.0` | None | Yes as locked Hopper diagnostic evidence | Preserve; no obvious locked-Hopper reward-model mismatch |
| `9400410_[0-13]` | Broad Flow-MBPO H200 AWR diagnostics | `COMPLETED`, exit `0:0` from artifacts/prior record | `submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_20260602.csv --gpu-type H200 --qos embers` | Submitted after `3a3e161`; results recorded at `515b243` | 14-row endpoint/trajectory/residual conservative AWR manifest | Project `pwm` env; MJLab QS H16 dataset, BC policy, synthetic replay and support-risk inputs | seeds `0` and `1` | H200 / `embers` | W&B disabled | Per-row policy and critic checkpoints under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_20260602/*/` | Per-row `summary.json` | Best recorded row `9400410_8`: return `25.969`, length `360.0`, fall `1.0`, below BC return `45.8491`, length `594.97`, fall `0.625` | None | Usable negative diagnostic evidence | Pivot rather than expanding this exact setting blindly |
| `9400525_[0-2]` | L40S Flow-MBPO AWR shard | `COMPLETED`, exit `0:0` from artifacts/prior record | `submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_l40s_20260602.csv --gpu-type L40S --qos embers --cpus 4` | Submitted after `d473642` | L40S shard manifest | Project `pwm` env; same MJLab QS and replay inputs | seed `1` shard rows | L40S / `embers` | W&B disabled | Per-row outputs under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_shards_20260602/l40s/` | Per-row `summary.json` | Returns `11.3848`, `10.2740`, `12.9974`; fall `1.0` | None | Usable negative confirmation | Same as H200: do not expand this exact conservative AWR setting |
| `9400771_0` | LeWM official PushT H200 eval repair row 0 | `FAILED`, exit `1:0`, elapsed `00:01:29` from log | `scripts/experiments/image_official/submit_newt_h200_remaining_lewm_h200_fix_20260602.sh` split into single-row H200 submission | Submitted after `ce439b0` / `a0a0751`; repaired at `f923be5` | Official LeWM `eval.py --config-name=pusht`, policy `pusht/lewm`, horizon 2, eval budget 30, W&B disabled | Official LeWM env `/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602`; PushT checkpoint/data under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm`; pyarrow/datasets shim active | seed `0` | H200 / `embers` | W&B disabled | Uses `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/pusht/lewm_object.ckpt` | Logs `logs/slurm/image_official/lewm_official_pusht_eval_h200_fix_single_9400771_0.{out,err}`; no eval result | Not available | New root cause after pyarrow repair: `stable_worldmodel.data.HDF5Dataset` missing because optional `hdf5plugin` was absent; pixel reads need plugin filters | No, infrastructure only | Repair committed at `f923be5`; resubmit LeWM eval/train rows only after Slurm is reachable and quota permits |
| `9400772_1` | LeWM official PushT H200 eval repair row 1 | `CANCELED` after row 0 failure | Same single-row H200 LeWM eval payload | Submitted after `ce439b0` / `a0a0751` | Same as `9400771_0` | Same as `9400771_0` | seed `0`, horizon 5 row | H200 / `embers` | W&B disabled | Same converted checkpoint | Logs `logs/slurm/image_official/lewm_official_pusht_eval_h200_fix_single_9400772_1.{out,err}` | None | Canceled to avoid repeating known HDF5 plugin failure | No | Keep canceled; replace only with post-`f923be5` HDF5 plugin repair |
| `9400778_7`, `9400797_8`, `9400798_9`, `9400799_10`, `9400800_11`, `9400814_12`, `9400815_13`, `9400816_14`, `9400817_15` | Official NEWT H200 remaining rows 7-15 | Logs show Slurm epilogs and valid official output for all listed rows; Slurm live refresh unavailable in this pass | Single-row H200 submissions using official NEWT `tdmpc2/train.py`, W&B/video/checkpoint disabled | Submitted after `903642d` and earlier records | Official NEWT 500-step smoke, `model_size=B` | Official NEWT repo/env under `/storage/project/r-agarg35-0/eliu354/external_repos/newt` and `/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602` | rows 7-15, seed/task per broad manifest | H200 / `embers` | W&B disabled | No checkpoint expected (`save_agent=false`) | Logs `logs/slurm/image_official/newt_official_h200_remaining_single_9400*.out` | Eval/train rewards by row: 7 `0.000/0.000`; 8 `18.338/37.122`; 9 `18.157/28.181`; 10 `16.060/5.912`; 11 `0.142/0.000`; 12 `10.000/0.000`; 13 `0.000/0.000`; 14 `144.140/14.021`; 15 `985.000/0.000` | None visible in logs | Yes as official NEWT smoke coverage, not performance evidence | Do not duplicate NEWT H200 rows 7-15 |

Current interpretation: the required MJLab faithful-PWM final/best eval and video gate is complete and negative; Ant/Hopper locked original-PWM supplemental evidence is usable; conservative Flow-MBPO AWR broad diagnostics are valid negative evidence; official NEWT smokes are running/completing as infrastructure checks only; official LeWM has a newly repaired HDF5 plugin issue but needs replacement submissions after Slurm becomes reachable again.

### Historical Git/Docs/GPU Record Scope Checked

This refresh also inspected the older git and documentation trail so the current
inventory is not isolated from prior work.

| Source | Scope inspected | GPU/job record takeaways |
| --- | --- | --- |
| `git log --all -- docs/git docs/goals scripts/experiments scripts/diagnostics src/flow_mbpo_pwm` | Recent history from `6474b85 Document PWM fidelity migration plan` through `f923be5 Record LeWM HDF5 compatibility repair` | Confirms a continuous English commit trail for PWM parity, faithful MJLab adapter evidence, Flow-MBPO AWR diagnostics, SIGReg prerequisite work, official NEWT/LeWM setup, Slurm repair records, and current HDF5 repair. |
| `git show --stat` for `f923be5`, `903642d`, `a0a0751`, `ce439b0`, `86df949`, `a81bc59`, `515b243`, `bd2773a`, `d473642`, `0a679b3`, `6f82e7d`, and `12fd08e` | Current 2026-06-02 scheduling/result commits | Confirms the latest commits mostly update docs plus small wrappers/shims; no hidden broad code rewrite was introduced in this refresh. |
| `docs/goals/mjlab_qs_rollout_policy_improvement_20260528.md` | BC and early Flow-MBPO history, including submitted/completed jobs in ranges `9236994`, `9237030`, `9237329`, `9237622`, `9237887`, `9238133`, `9238737`, `9239193`, `9239482`, `9239804`, `9240159`, `9240496`, `9240994`, `9241333`, `9241796`, `9270759`, `9278096`, `9304777`, `9325185`, `9326058`, `9349421`, `9349483`, `9349486`, and many `935xxxx`/`937xxxx` Flow-MBPO diagnostic jobs | Establishes the BC comparator chain. The durable baseline used by the current plan is still the 40-episode BC eval from job `9238737`: return `45.8491`, length `594.97`, fall `0.625`. Earlier 300-step videos understated BC because of the video cap; 1000-step and 40-episode evals are the relevant comparators. |
| `docs/goals/flow_mbpo_top_conf_research_plan_20260531.md` | Flow-MBPO v1 pessimistic/CQL/support diagnostics around jobs `9357006`, `9357054`, `9357126`, `9357174`, `9357227`, `9357292`, `9370468`, `9370586`, `9370641`, `9370667`, and `9370771` | Confirms those rows were W&B-disabled diagnostics/infrastructure checks on `embers`, not formal policy-improvement claims. They motivated gate-aware selection, early-stop, baseline-gate logging, and rollout/eval passthrough infrastructure. |
| `docs/goals/0601/pwm_fidelity_mjlab_flow_goal_reference_20260601.md` | Original rule set for PWM fidelity, MJLab transfer, and Flow replacement | Confirms the non-negotiable policy still applies: original PWM parity first, final and true-best actors, 40-episode real eval, MP4/W&B videos, return/length/fall, and `embers` QOS unless `inferno` is explicitly approved. |
| `docs/goals/pwm_fidelity_mjlab_flow_migration_20260601.md` | PWM parity and faithful-adapter migration trail from setup SHA `d372003` through current gate records | Confirms original source parity inspection, locked original PWM environment creation, Hopper/Ant DFlex job series, faithful MJLab adapter smoke/formal jobs `9387896`/`9387895`, required eval/video replacement jobs, and final Hopper probe `9399798`. |
| `logs/pwm_original_parity/locked_env_20260601/` | DFlex parity and repair logs for `93837xx`, `93838xx`, `93843xx`, `93874xx`, `93879xx`, `93886xx`, `93948xx`, `93983xx`, and `9399798` | Confirms the repeated DFlex failures were infrastructure/compiler/path issues until Ant fix3 and Hopper fix4; those final fixes produced usable DFlex evidence. |
| `logs/slurm/mjlab_qs/` and `logs/slurm/image_official/` | MJLab adapter eval/rollout logs, Flow-MBPO AWR logs, NEWT/LeWM official-image logs | Confirms the current recorded statuses are grounded in actual Slurm logs where accounting was unavailable. LeWM row `9400771_0` failed after passing the earlier pyarrow repair, proving the newer HDF5 plugin root cause; NEWT H200 remaining rows have valid official smoke output. |

## Continuation Inventory: Pending LeWM And Completed Shards

Preflight time: 2026-06-02 after commit `30d7450`, America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD | `e8a52e6` before the LeWM pathfix script/doc edit |
| Dirty status before this inventory edit | `scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh` modified for the pathfix |
| Slurm commands | `squeue -u $USER -o ...`; `sacct -X -S 2026-06-02 -u $USER --format=...`; `sacct -X -j 9401543,9401544 --format=...`; `seff` unavailable on PATH |
| Artifact searches | Checked `logs/slurm/mjlab_qs/flow_mbpo_awr`, `logs/slurm/image_official`, `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_shards_20260602`, and LeWM HDF5-fix log paths. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9401543_[0-5%3]` | LeWM official PushT eval after HDF5 plugin repair | Row `9401543_0` `FAILED` exit `1:0` after `00:00:24`; row `9401543_1` canceled after start; rows `9401543_[2-5%3]` canceled before start | `scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh` | `8270506` script commit; submitted at `30d7450`; failure recorded during pathfix edit after `e8a52e6` | Official LeWM `eval.py --config-name=pusht`; policies `pusht/lewm` and `random`; horizons 2/5; eval budget 30; W&B disabled | Official LeWM env `/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602`; compat root `scripts/experiments/image_official/compat`; PushT assets under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm` | row 0 seed 0 horizon 2 started; siblings canceled | H200 / `embers` | W&B disabled | Used `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/pusht/lewm_object.ckpt` | Failure log `logs/slurm/image_official/lewm_official_pusht_eval_hdf5fix_h200_9401543_0.out`; replacement will use `hdf5pathfix` logs | Not available | Official eval expected `${STABLEWM_HOME}/datasets/pusht_expert_train.h5`; dataset existed at `${STABLEWM_HOME}/pusht_expert_train.h5`. HDF5 plugin repair itself worked. | No; failure is useful diagnostic only | Symlink `${STABLEWM_HOME}/datasets/pusht_expert_train.h5 -> ../pusht_expert_train.h5`, validate eval/train dataset loading, commit, and resubmit with `hdf5pathfix` names |
| `9401544_[0-1%2]` | LeWM official PushT train smoke after HDF5 plugin repair | `CANCELLED+` before start after `9401543_0` exposed the shared cache-layout issue | `scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh` | `8270506` script commit; submitted at `30d7450`; canceled during pathfix edit after `e8a52e6` | Official LeWM `train.py data=pusht`, 1 epoch, 2 train batches, 1 val batch, W&B disabled | Same repaired official LeWM env/assets/compat root as `9401543` | seeds 0 and 1 canceled | H200 / `embers` | W&B disabled | Expected smoke train outputs did not materialize | No useful train logs because rows were canceled before start | Not available | Not directly failed; canceled because train used the same dataset location family and needed path validation before running | No | Replacement script exports `LOCAL_DATASET_DIR=${STABLEWM_HOME}` because official `load_dataset` appends `datasets`; local train dataset validation returned length `2336736`, state dim `7` |
| `9401638_[0-5%3]` | LeWM official PushT eval after HDF5 dataset pathfix | `PENDING`, reason `Priority` at first check | `scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh` | `8f7dbce` pathfix commit | Official LeWM `eval.py --config-name=pusht`; policies `pusht/lewm` and `random`; horizons 2/5; eval budget 30; W&B disabled; result suffixes include `hdf5pathfix` | Official LeWM env; compat root; PushT assets; `${STABLEWM_HOME}/datasets/pusht_expert_train.h5 -> ../pusht_expert_train.h5` symlink validated | rows cover seeds 0, 1, 2 and random baselines | H200 / `embers` | W&B disabled | Uses converted `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/pusht/lewm_object.ckpt` | Expected logs `logs/slurm/image_official/lewm_official_pusht_eval_hdf5pathfix_h200_9401638_*.{out,err}` | Not available | None yet | Pending only | Monitor first started row immediately; cancel siblings if a new shared root cause appears |
| `9401639_[0-1%2]` | LeWM official PushT train smoke after HDF5 dataset pathfix | `PENDING`, reason `Priority` at first check | `scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh` | `8f7dbce` pathfix commit | Official LeWM `train.py data=pusht`, `data.dataset.name=pusht_expert_train.h5`, 1 epoch, 2 train batches, 1 val batch, W&B disabled | Same repaired official LeWM env/assets/compat root as `9401638`; `LOCAL_DATASET_DIR=${STABLEWM_HOME}` locally validated through `load_dataset` | seeds 0 and 1 | H200 / `embers` | W&B disabled | Expected smoke train outputs under official LeWM output dirs with `hdf5pathfix` suffixes | Expected logs `logs/slurm/image_official/lewm_official_pusht_train_hdf5pathfix_h200_9401639_*.{out,err}` | Not available | None yet | Pending only | Inspect first train row when it starts |
| `9400435_[0-3]` | H100 Flow-MBPO AWR shard backfill | All 4 elements `COMPLETED`, exit `0:0` | `submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_h100_20260602.csv --gpu-type H100 --partition gpu-h100 --qos embers --max-concurrent 4` | Submitted after `a2a7aac`; completed after earlier pending inventory | H100 shard manifest: endpoint H1 seed1, residual H3/H5 seed0, trajectory H3 fall5 seed0 | Project `pwm` env; same MJLab QS H16 data, BC checkpoint, replay/support inputs as broad AWR batch | seed 0/1 by row | H100 / `embers` | W&B disabled | Per-row `final_policy_extraction.pt`, `best_policy_extraction.pt`, `best_training_loss_policy_extraction.pt`, and `final_q_critic.pt` under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_shards_20260602/h100/*/` | Per-row `summary.json`; Slurm logs `logs/slurm/mjlab_qs/flow_mbpo_awr/mjqs_flow_mbpo_awr_9400435_*.out` | 8-episode real eval rows: returns `19.6023`, `19.9737`, `21.3537`, `11.0026`; lengths `296.875`, `291.125`, `323.375`, `181.5`; fall `1.0` for all | None; valid negative runs | Usable negative confirmation; still below BC `45.8491` / `594.97` / `0.625` | Do not expand this exact conservative AWR setting; pivot to different objective/gating if continuing Flow-MBPO |
| `9400714_[0-15]` | NEWT official broad L40S smoke | All 16 elements `COMPLETED`, exit `0:0` by `sacct` and logs | `scripts/experiments/image_official/submit_newt_lewm_l40s_backups_20260602.sh` | Submitted after `ce439b0` | Official NEWT `tdmpc2/train.py`, `model_size=B`, 500 steps, W&B/video/checkpoint disabled | Official NEWT repo/env; data dir `/storage/project/r-agarg35-0/eliu354/external_data/newt_demos` | rows 0-15, seeds 0 and 1 across 8 tasks | L40S / `embers` | W&B disabled | No checkpoint expected (`save_agent=false`) | Logs `logs/slurm/image_official/newt_official_broad_l40s_9400714_*.out` | Eval/train rewards by row: 0 `42.247/51.202`; 1 `42.179/23.809`; 2 `5.871/6.516`; 3 `0.000/0.000`; 4 `34.000/0.000`; 5 `0.000/0.000`; 6 `183.107/8.803`; 7 `0.000/0.000`; 8 `18.338/37.122`; 9 `18.157/28.181`; 10 `13.833/5.912`; 11 `0.142/0.000`; 12 `10.000/0.000`; 13 `0.000/0.000`; 14 `144.140/14.021`; 15 `985.000/0.000` | None | Yes as official NEWT infrastructure smoke only | Preserve; do not interpret as performance evidence |

Current queue contains pending A100 jobs `9399799`, `9400409`, `9400442`, and `9400528`, pending H200 LeWM pathfix arrays `9401638` and `9401639`, plus unrelated `9400333 sam3_img_smoke`. LeWM HDF5-fix arrays `9401543` and `9401544` were canceled after the cache-layout failure.

## Continuation Inventory: LeWM Pathfix Pending And AWR Negative Confirmation

Preflight time: 2026-06-02 after commit `be114a9`, America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD | `be114a9` |
| Dirty status before this inventory edit | clean |
| Slurm commands | `squeue -u $USER -o ...`; `sacct -X -S 2026-06-02 -u $USER --format=...`; `seff` unavailable on PATH |
| Artifact searches | Checked `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_20260602`, `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_shards_20260602`, `logs/slurm/mjlab_qs/flow_mbpo_awr`, and current LeWM pathfix queue state. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9401638_[0-5%3]` | LeWM official PushT eval after HDF5 dataset pathfix | `PENDING`, reason `Priority` by `squeue`; `PENDING` / QOS `embers` by `sacct` | `scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh` | `8f7dbce`; submission recorded at `be114a9` | Official LeWM `eval.py --config-name=pusht`; policies `pusht/lewm` and `random`; horizons 2/5; eval budget 30 | Official LeWM env; compat root; PushT HDF5 dataset plus dataset symlink; converted LeWM checkpoint | seeds 0, 1, 2 and random baselines | H200 / `embers` | W&B disabled | `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/pusht/lewm_object.ckpt` | Expected `logs/slurm/image_official/lewm_official_pusht_eval_hdf5pathfix_h200_9401638_*.{out,err}`; no logs yet | Not available | None yet | Pending only | Monitor first started row; cancel siblings and record if a new root cause appears |
| `9401639_[0-1%2]` | LeWM official PushT train smoke after HDF5 dataset pathfix | `PENDING`, reason `Priority` by `squeue`; `PENDING` / QOS `embers` by `sacct` | Same script | `8f7dbce`; submission recorded at `be114a9` | Official LeWM `train.py data=pusht`, `data.dataset.name=pusht_expert_train.h5`, 1 epoch, 2 train batches, 1 val batch | Same official LeWM env/assets/compat root; `LOCAL_DATASET_DIR=${STABLEWM_HOME}` validated locally | seeds 0 and 1 | H200 / `embers` | W&B disabled | Expected smoke output under official LeWM output dirs | Expected `logs/slurm/image_official/lewm_official_pusht_train_hdf5pathfix_h200_9401639_*.{out,err}`; no logs yet | Not available | None yet | Pending only | Inspect first train row when it starts |
| `9400410_[0-13]` | Full H200 conservative Flow-MBPO AWR diagnostic array | All 14 elements `COMPLETED`, exit `0:0`; all `summary.json` files present | `submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_20260602.csv --gpu-type H200 --partition gpu-h200 --qos embers --max-concurrent 8` | Run summaries report `0a679b3` | Full 14-row endpoint/trajectory/residual conservative AWR sweep; 8-episode real eval every 10 updates; W&B disabled | Project `pwm` env; MJLab QS H16 dataset, BC policy checkpoint, synthetic replay/support inputs | seeds 0 and 1 by manifest row | H200 / `embers` | W&B disabled | Per-row final/best/best-training policy and critic checkpoints under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_20260602/*/` | Per-row `summary.json`; no formal videos | Best row `endpoint_h3_trunc_cql_mixed_s1`: return `25.9699`, length `360.000`, fall `1.000`; all rows below BC `45.8491` / `594.97` / `0.625` | None; valid negative result | Usable diagnostic negative evidence | Do not expand this exact conservative AWR sweep; use it as model-exploitation/fall-failure evidence |
| `9400436_[0-3]` | H200 shard duplicate of selected AWR rows | All 4 elements `COMPLETED`, exit `0:0`; all summaries present | `submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_h200_20260602.csv --gpu-type H200 --partition gpu-h200 --qos embers --max-concurrent 4` | Run summaries report `0a679b3` | H200 shard manifest | Same project env and MJLab QS inputs; independent shard output root | seed 0 rows | H200 / `embers` | W&B disabled | Under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_shards_20260602/h200/*/` | Per-row `summary.json` | Best row `endpoint_h1_trunc_cql_data_noise_s0`: return `22.9451`, length `333.750`, fall `1.000` | None | Usable negative confirmation | No duplicate submission |
| `9400525_[0-2]` | L40S AWR shard backfill | All 3 elements `COMPLETED`, exit `0:0`; all summaries present | `submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_l40s_20260602.csv --gpu-type L40S --partition gpu-l40s --qos embers --cpus 4` | Run summaries report `0a679b3` | L40S shard manifest | Same project env and MJLab QS inputs | seed 1 rows | L40S / `embers` | W&B disabled | Under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_shards_20260602/l40s/*/` | Per-row `summary.json` | Best row `traj_h3_fall5_trunc_cql_data_noise_s1`: return `16.4492`, length `256.375`, fall `1.000` | None | Usable negative confirmation | No duplicate submission |

Current queue still contains pending A100 jobs `9399799`, `9400409`, `9400442`, and `9400528`, plus unrelated `9400333 sam3_img_smoke`. No new submission is justified in this poll because active LeWM/A100 jobs are already queued and the completed AWR rows are negative diagnostics.

## Continuation Inventory: R0-R4 Matrix Preparation

Preflight time: 2026-06-02 after commit `cb819b8`, America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD | `cb819b8` |
| Dirty status before this inventory edit | clean |
| Slurm commands | `squeue -u $USER -o ...`; `sacct -X -S 2026-06-02 -u $USER --format=...`; `seff` unavailable on PATH |
| Artifact searches | Checked R0/R4 eval and rollout artifacts, Flow-MBPO candidate ranking tools, status CSVs, broad AWR summaries, and pending LeWM/A100 queue state. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9395746_[0-1]` | R0 faithful original PWM adapter final/best 40-episode eval fix2 | `COMPLETED`, exit `0:0` | `submit_array.sh --kind policy_eval --manifest scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_eval40_final_best_fix2_20260602.csv` | Recorded in prior docs; result used in `docs/git/flow_pwm_matched_evidence_inventory_20260602.md` | Faithful original PWM adapter final/best eval40 | Hybrid locked MJLab bridge with fixed QS dataset/metadata/normalization | seed 0 final/best | H200 / `embers` | W&B enabled in manifest | Formal adapter final/best checkpoints from `9387895` | Eval summaries under `scripts/outputs/mjlab_qs/policy_evals/original_pwm_adapter_phase3_eval40_fix2_20260602/` | Eval final `-0.8010`; eval best `-0.7778`; fall `1.000` | None; valid collapse evidence | Usable negative R0 eval evidence | Treat as completed R0 baseline |
| `9396189_[0-1]` | R0 faithful original PWM adapter final/best rollout video fix2 | `COMPLETED`, exit `0:0` | `submit_array.sh --kind policy_rollout --manifest scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_rollout10_final_best_fix2_20260602.csv` | Recorded in prior docs | Faithful original PWM adapter final/best rollout10 | Hybrid locked MJLab bridge with fixed QS dataset/metadata/normalization | seed 0 final/best | H200 / `embers` | W&B enabled in manifest | Same final/best checkpoints | Rollout summaries and MP4s under `scripts/outputs/mjlab_qs/policy_rollouts/original_pwm_adapter_phase3_rollout10_fix2_20260602/` | Video final `-0.5265`; video best `-0.5218`; fall `1.000` | None; valid collapse evidence | Usable negative R0 video evidence | Treat as completed R0 baseline |
| `R1/R2` | Controlled Flow WM-only and Flow policy-only rows | Not submitted; no implementable fixed-protocol row identified | New row design required | Not applicable | Must keep R0 dataset, seed, eval/video protocol, and change one variable only | Project/hybrid env decision still needed | seed 0 first | Prefer H200/H100/A100/L40S / `embers` when ready | W&B on for formal | Missing row-specific checkpoints | Missing | Not available | Missing runner/input design | Not usable yet | Build explicit row inputs before any sbatch |

Current queue is unchanged: LeWM pathfix arrays `9401638` and `9401639`, A100 jobs `9399799`, `9400409`, `9400442`, and `9400528`, plus unrelated `9400333` remain pending. The R0-R4 preparation record is `docs/git/r0_r4_controlled_matrix_status_20260602.md`; it does not justify a new submission yet.

## Continuation Inventory: PWM Comparator Boundary And LeWM HF Cache Repair

Preflight time: 2026-06-02 after commit `dbc3829`, America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD before edits | `dbc3829` |
| Dirty status before this edit | clean |
| Slurm commands | `squeue -u $USER -o ...`; `sacct -X -j 9401638,9401639 --format=...`; `seff` unavailable earlier and not used in this pass |
| Artifact/log searches | Checked `logs/slurm/image_official/lewm_official_pusht_eval_hdf5pathfix_h200_9401638_0.{out,err}`, official LeWM loader code, external LeWM cache layout, `docs/git`, `docs/goals`, and current queue. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9401638_0` | LeWM official PushT eval after HDF5 dataset pathfix | `FAILED`, exit `1:0`, elapsed `00:00:36` | `scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh` | `8f7dbce` submission; failure repaired in current edit | Official LeWM `eval.py --config-name=pusht`, policy `pusht/lewm`, seed 0, horizon 2, eval budget 30 | Official LeWM env `/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602`; PushT HDF5 dataset under `${STABLEWM_HOME}`; hdf5plugin compat shim active | seed `0` | H200 / `embers` | W&B disabled | Existing converted object checkpoint `${STABLEWM_HOME}/pusht/lewm_object.ckpt`; official loader wanted `${STABLEWM_HOME}/checkpoints/models--pusht--lewm` | Log `logs/slurm/image_official/lewm_official_pusht_eval_hdf5pathfix_h200_9401638_0.out`; no eval result | Not available | Dataset path was fixed, but official `load_pretrained('pusht/lewm')` fell through to `https://huggingface.co/pusht/lewm/...` and got HTTP 401 Unauthorized because the expected local pretrained cache was missing/invalid. Raw HF weights also need legacy ViT key normalization before this loader can load them. | Useful infrastructure diagnostic only | Replacement wrapper now prepares normalized local pretrained cache and uses `hfcachefix` names; submit after commit |
| `9401638_[1-5%3]` | LeWM official PushT eval siblings after HDF5 dataset pathfix | `CANCELLED by 3509929` at `2026-06-02T20:38:46` | Same script | `8f7dbce` | Same eval array | Same env/assets | seeds 1,2 and random baseline rows | H200 / `embers` | W&B disabled | Same cache family | No useful row logs | Not available | Canceled to avoid repeating the shared pretrained-cache root cause | No | Keep canceled; replace with hfcachefix array |
| `9401639_[0-1%2]` | LeWM official PushT train smoke after HDF5 dataset pathfix | `CANCELLED by 3509929` at `2026-06-02T20:38:46` | Same script | `8f7dbce` | Official LeWM `train.py data=pusht`, 1 epoch, 2 train batches, 1 val batch | Same env/assets | seeds 0 and 1 | H200 / `embers` | W&B disabled | Train would write official smoke checkpoints under LeWM cache | No useful row logs | Not available | Canceled because eval exposed a shared LeWM cache-layout issue before train started | No | Replace with hfcachefix train smoke after commit |

Additional documentation correction in this pass:

```text
The MJLab `original_pwm_adapter` row is not a full upstream PWM `train_dflex.py`
or `train_multitask.py` pipeline reproduction. It is adapter-level evidence:
upstream `baselines/PWM/src/pwm.algorithms.pwm.PWM` model/update code is used,
but MJLab-QS window sampling, loop orchestration, and MJLab eval are implemented
in `scripts/experiments/mjlab_qs/run_original_pwm_adapter.py`. R0 and matched
evidence docs now describe this as an upstream PWM algorithm adapter baseline,
not a full upstream-pipeline failure claim.
```

Current queue after canceling bad LeWM pathfix rows:

```text
9400409_[0-15] NEWT official broad A100 smoke: PENDING Priority.
9400528_[1-2%2] Flow-MBPO AWR A100 remaining rows: PENDING Priority.
9400442 Flow-MBPO AWR A100 single row: PENDING Priority.
```

Validation for the replacement candidate:

```text
bash -n scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh
swm.wm.utils.load_pretrained('pusht/lewm', cache_dir=${STABLEWM_HOME}) -> LeWM
swm.data.HDF5Dataset('pusht_expert_train', cache_dir=${STABLEWM_HOME}) -> length 2336736, state dim 7
swm.data.load_dataset('pusht_expert_train.h5', cache_dir=${STABLEWM_HOME}) -> length 2336736, state dim 7
sbatch --test-only accepted eval and train H200 / embers requests.
```

Submit decision: after committing this repair record and wrapper, submit the
LeWM `hfcachefix` H200 eval/train replacement arrays. No dependency is needed
because the official env, dataset, hdf5plugin compat shim, and normalized local
pretrained cache already exist.

Submitted LeWM hfcachefix replacements after commit `7ccc508`:

```text
script: scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh
cache_preparation: normalized `${STABLEWM_HOME}/checkpoints/models--pusht--lewm/weights.pt`, size 72265441 bytes.
submitted_jobs:
  9401796_[0-5%3] lewm_official_pusht_eval_hfcachefix_h200_20260602, H200 / embers, PENDING Priority at first check.
  9401797_[0-1%2] lewm_official_pusht_train_hfcachefix_h200_20260602, H200 / embers, PENDING Priority at first check.
wandb: disabled.
dependencies: none.
sacct: both arrays PENDING, QOS embers, exit 0:0.
expected_logs:
  logs/slurm/image_official/lewm_official_pusht_eval_hfcachefix_h200_%A_%a.{out,err}
  logs/slurm/image_official/lewm_official_pusht_train_hfcachefix_h200_%A_%a.{out,err}
next_action: inspect the first started row; keep canceled pathfix jobs as failed diagnostics and do not duplicate them.
```

NEWT official walker A100 smoke completion:

```text
job: 9399799 `newt_official_walker_swig_fix1_a100_20260602`.
status: COMPLETED 0:0, elapsed 00:00:48, gpu-a100 / embers.
log: logs/slurm/image_official/newt_official_walker_swig_fix1_9399799.out.
result: official NEWT walker-walk smoke created the environment, compiled the model, disabled W&B, trained for 1,000 steps, and printed `Training completed successfully`.
reported rewards: eval I=0 R=42.248; train I=500 R=51.202; train I=1000 R=45.629.
usable: yes, as official NEWT infrastructure smoke only.
next_action: preserve result and wait for broad A100 rows; do not duplicate this walker smoke.
```

LeWM hfcachefix first-row completion:

```text
job: 9401796_0 `lewm_official_pusht_eval_hfcachefix_h200_20260602`.
status: COMPLETED 0:0, elapsed 00:01:34, gpu-h200 / embers.
log: logs/slurm/image_official/lewm_official_pusht_eval_hfcachefix_h200_9401796_0.out.
result: official eval loaded the PushT HDF5 dataset and loaded `pusht/lewm` from the normalized local pretrained cache instead of hitting the unauthorized HF repo. It found 1,869,611 valid starts and reported `success_rate: 100.0` over 4 eval episodes.
usable: yes, as official LeWM smoke evidence for eval row 0.
next_action: keep `9401796_[1-5]` and `9401797_[0-1]` pending/running; no cancellation needed for this root cause.
```

LeWM hfcachefix eval sweep completion:

```text
9401796_[0-5] all COMPLETED 0:0 on gpu-h200 / embers.
success_rate by row: 100.0, 100.0, 75.0, 75.0, 0.0, 0.0.
aggregate: 14/24 successful episodes, mean row success_rate 58.33.
interpretation: the local pretrained-cache and dataset fixes are validated across the full eval array, but official eval performance is mixed across row settings. This replaces the earlier first-row-only 100% note.
next_action: monitor train smoke array `9401797_[0-1]`, running at the time of this record.
```

LeWM hfcachefix train smoke failure:

```text
9401797_0 FAILED 1:0 after 00:00:52 on gpu-h200 / embers.
9401797_1 FAILED 1:0 after 00:00:40 on gpu-h200 / embers.
root_cause: Hydra/OmegaConf struct mode rejected `trainer.limit_train_batches=2`
because the official trainer config lacks that key. Hydra reported that append
syntax should be used: `+trainer.limit_train_batches=2`.
interpretation: the train smoke failed before training; this does not invalidate
the completed eval cache/data fix.
next_action: repair the train command overrides and resubmit train-only rows.
```

## Continuation Inventory: Full Upstream PWM Pipeline On MJLab Candidate

Preflight time: 2026-06-02 after commit `6930a67`, America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD before edits | `6930a67` |
| Dirty status before this edit | clean before adding the new upstream-PWM-on-MJLab smoke files |
| Slurm commands | `squeue -u $USER -o ...`; `sbatch --test-only ...`; LeWM `9401796_0` also started while this candidate was being prepared |
| Artifact/log searches | Checked `baselines/PWM/scripts/train_dflex.py`, upstream `baselines/PWM/src/pwm/algorithms/pwm.py`, existing `src/flow_mbpo_pwm/envs/mjlab_pwm_adapter.py`, `scripts/experiments/mjlab_qs/locked_mjlab_python.py`, and prior docs for `original_pwm_adapter` limitations. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9401871` | Full upstream PWM orchestration smoke on MJLab | `COMPLETED`, exit `0:0`, elapsed `00:00:27` | `scripts/experiments/mjlab_qs/submit_upstream_pwm_mjlab_full_pipeline_smoke_20260602.sh`, running `scripts/experiments/mjlab_qs/locked_mjlab_python.py train_dflex.py env=mjlab_velocity_flat_unitree_g1 alg=pwm ...` from `baselines/PWM/scripts` | `f79cb1d`; result recorded after `98d025b` | The wrapper writes `cfg/env/mjlab_velocity_flat_unitree_g1.yaml` at job startup because `baselines/PWM` is a nested repo ignored by the main repo; `alg=pwm` resolves to upstream `pwm.algorithms.pwm.PWM`; smoke overrides `alg.max_epochs=8`, `alg.horizon=8`, `critic_iterations=1`, `wm_iterations=1`, `wm_batch_size=16`, W&B disabled | Base Python/torch/PWM from locked original PWM env `/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4`; MJLab and local env adapter exposed by `locked_mjlab_python.py`; MJLab task `Mjlab-Velocity-Flat-Unitree-G1`, online env interaction, no QS-window dataset | seed `0` | H200 / `embers` | W&B disabled | `baselines/PWM/scripts/outputs/2026-06-02/21-19-04/logs/upstream_pwm_mjlab_full_smoke_h200_seed0_20260602/{init_policy.pt,best_policy.pt,final_policy.pt,final_policy.buffer/}` | `logs/slurm/mjlab_qs/upstream_pwm_full_pipeline/upstream_pwm_mjlab_full_smoke_h200_9401871.{out,err}` | Last update row `[4/8]` R `0.10`, H `8.0`, S `1024`; eval mean episode loss `0.05`, mean discounted loss `0.05`, mean episode length `32.00` | None; warnings only from W&B/Gym/functorch deprecations | Yes: full-pipeline feasibility evidence | Promote to longer full-PWM MJLab run/eval only after defining performance gate |

## Continuation Preflight After Full PWM Smoke

Preflight time: 2026-06-02 after commit `7a854ab`, America/New_York.

| Item | Evidence |
| --- | --- |
| Branch / SHA | `mjlab-qs-rollout-policy-improvement` / `7a854ab4ff35dc76a1ba384ed099e1a5d9244820` |
| Worktree | clean at preflight (`git status --short --branch`) |
| Recent commits | `7a854ab Record upstream PWM MJLab smoke result`; `dc35bec Record LeWM hfcachefix eval sweep`; `98d025b Record upstream PWM MJLab smoke submission`; `f79cb1d Fix upstream PWM MJLab smoke config setup`; `0ab2aef Add upstream PWM MJLab smoke`; `6930a67 Record NEWT walker smoke result`; `e9b9bb9 Record LeWM hfcachefix submissions`; `7ccc508 Fix LeWM pretrained cache setup` |
| Slurm commands | `squeue -u $USER -o ...`; `sacct -X -S 2026-06-02 -u $USER --format=... -P`; `seff` unavailable on PATH |
| Search scope | Searched `docs/git`, `docs/goals`, `scripts`, `logs/slurm`, W&B/offline roots, checkpoints, and video/eval roots for recent Flow-MBPO AWR, NEWT, LeWM, and full upstream PWM job IDs/artifacts. |
| Active scheduler state | `9400409_[0-15]` NEWT A100 broad smoke pending Priority; `9400528_[1-2%2]` Flow-MBPO AWR A100 remaining pending Priority; `9400442` Flow-MBPO AWR A100 single pending Priority. No active H200/H100 jobs at this preflight. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9401871` | Full upstream PWM orchestration smoke on MJLab | `COMPLETED`, exit `0:0` | `scripts/experiments/mjlab_qs/submit_upstream_pwm_mjlab_full_pipeline_smoke_20260602.sh` | `f79cb1d` submission wrapper; result recorded by `7a854ab` | Wrapper-generated `env=mjlab_velocity_flat_unitree_g1`; `alg=pwm`; `alg.max_epochs=8`; `alg.horizon=8`; W&B disabled | Locked original PWM env via `locked_mjlab_python.py`; upstream PWM from `baselines/PWM/src`; MJLab G1 adapter; online interaction, no QS dataset | `0` | H200 / `embers` | Disabled | `baselines/PWM/scripts/outputs/2026-06-02/21-19-04/logs/upstream_pwm_mjlab_full_smoke_h200_seed0_20260602/{init_policy.pt,best_policy.pt,final_policy.pt,final_policy.buffer/}` | Slurm log `logs/slurm/mjlab_qs/upstream_pwm_full_pipeline/upstream_pwm_mjlab_full_smoke_h200_9401871.out` | Last update row `[4/8]` R `0.10`, eval mean episode loss `0.05`, mean episode length `32.00` | None | Yes, feasibility evidence only | Submit a longer W&B-disabled diagnostic full-PWM MJLab run before any formal claim. |
| `9401796_[0-5]` | Official LeWM PushT eval after local pretrained-cache fix | All six elements `COMPLETED`, exit `0:0` | `scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh` eval array | `7ccc508` repair; results recorded by `dc35bec` | Official `eval.py --config-name=pusht`; policies `pusht/lewm` and `random`; horizons 2/5; `eval.num_eval=4`; W&B disabled | Official LeWM env `/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602`; local HDF5 data and normalized `models--pusht--lewm` cache | rows 0-5 map to seeds/policies in wrapper | H200 / `embers` | Disabled | Local pretrained cache under `${STABLEWM_HOME}/checkpoints/models--pusht--lewm` | Logs under `logs/slurm/image_official/lewm_official_pusht_eval_hfcachefix_h200_9401796_*.out` | Row success rates `100,100,75,75,0,0`; aggregate 14/24 successes, mean row success `58.33` | None | Yes, mixed official LeWM eval evidence | Do not call this a stable 100% result; keep as mixed eval reference. |
| `9401797_[0-1]` | Official LeWM PushT train smoke after local pretrained-cache fix | Both elements `FAILED`, exit `1:0` | `scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh` train array | `7ccc508`; failure recorded by `7a854ab` | Official `train.py data=pusht`; `trainer.max_epochs=1`; attempted `trainer.limit_train_batches=2`, `trainer.limit_val_batches=1`; W&B disabled | Official LeWM env and local HDF5 data/cache | seeds `0`, `1` | H200 / `embers` | Disabled | None from failed train | Logs `logs/slurm/image_official/lewm_official_pusht_train_hfcachefix_h200_9401797_{0,1}.out` | Not available | Hydra/OmegaConf struct rejected `trainer.limit_train_batches`; append syntax `+trainer.limit_train_batches=2` is required | No, command-line failure only | Patch train-only replacement wrapper with `+trainer.limit_*` overrides and resubmit after candidate record. |
| `9400409_[0-15]` | Official NEWT broad A100 smoke | Pending Priority | `scripts/experiments/image_official/submit_newt_lewm_broad_gpu_20260602.sh` | `3a3e161` submission; still pending | 16 official NEWT train smokes, 500 steps, W&B/video/checkpoint off | Official NEWT env; DMControl tasks | seeds `0`, `1` | A100 / `embers` | Disabled | Expected local NEWT logs only | Slurm logs expected under `logs/slurm/image_official/newt_official_broad_smoke_%A_%a.{out,err}` | Not available | None yet | Pending | Leave queued; L40S/H200 completed rows already validate the infrastructure. |
| `9400528_[1-2%2]` and `9400442` | Flow-MBPO AWR A100 remaining shards | Pending Priority | `submit_array.sh --kind flow_mbpo_awr ... --gpu-type A100` | Earlier broad AWR shard commits; pending at preflight | Conservative AWR diagnostics from existing manifests | Project `pwm` env; MJLab QS dataset and synthetic replays | manifest rows | A100 / `embers` | Disabled | Expected per-row AWR summaries/checkpoints | Expected under `logs/slurm/mjlab_qs/flow_mbpo_awr/` and row output dirs | Not available | None yet | Pending | Leave queued; completed H200/H100/L40S AWR rows are already usable negative evidence. |

Continuation submissions after candidate commit `daff684`:

```text
9401906 upstream_pwm_mjlab_full_longdiag_h200_20260602, H200 / embers,
PENDING Resources at first check.

9401907_[0-1%2] lewm_official_pusht_train_hfcachefix_fix1_h200_20260602,
H200 / embers, PENDING Resources at first check.
```

Continuation poll after commit `45a4355`:

```text
queue: `9401906`, `9401907_[0-1]`, `9400409_[0-15]`, `9400528_[1-2]`, and
`9400442` all remained pending; no new Slurm logs were available.
seff: unavailable on PATH.
static progress: R4 existing-candidate selection was completed in
`docs/git/r0_r4_controlled_matrix_status_20260602.md`.
selected R4 scalar reproduction:
  flow_endpoint_seed0_h1_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0
reason: strongest existing 40-episode scalar eval final checkpoint
  return 60.8721, length 759.30, fall 0.450.
submit_decision: no duplicate eval/video job because final/best eval40 and
rollout10 artifacts already exist for the top R4 candidates.
```

Running poll after R4 selection commit:

```text
9401906 upstream_pwm_mjlab_full_longdiag_h200_20260602 moved from PENDING to
RUNNING on gpu-h200 / embers at 2026-06-02T21:34:33.
early stdout showed `Setting seed: 0` and `Running sim with no_grad=True`;
stderr showed only the known locked-env W&B/Gym/functorch deprecation warnings.
No result or failure is available yet.
9401907_[0-1] remained PENDING Resources.
```

Continuation results after the running poll:

```text
9401906 upstream_pwm_mjlab_full_longdiag_h200_20260602 COMPLETED 0:0 after
00:02:15 on gpu-h200 / embers.
usable: yes, as full-upstream-pipeline diagnostic evidence and checkpoint
producer; no, as a formal MJLab performance claim.
outputs: baselines/PWM/scripts/outputs/2026-06-02/21-35-04/logs/upstream_pwm_mjlab_full_longdiag_h200_seed0_20260602/
result: last printed update row `[196/200]` had R `-1.77`, mean horizon `8.8`,
step `102400`; internal eval mean episode loss `0.44`, mean discounted loss
`0.38`, mean episode length `35.33`.
failure_reason: none. Caution: repeated terminal-observation fallback warnings.
next_action: determine whether the upstream PWM checkpoint format can be
evaluated by the existing MJLab eval/render bridge before submitting any
fall-aware final/best eval/video gates.

9401907_0 and 9401907_1 LeWM train hfcachefix fix1 both COMPLETED 0:0 on
gpu-h200 / embers.
usable: yes, official LeWM train-smoke infrastructure evidence.
outputs: `${STABLEWM_HOME}/checkpoints/lewm_train_smoke_h200_hfcachefix_fix1_seed{0,1}/weights_epoch_1.pt`.
result: both rows reached Trainer.fit, ran the intended 2 train batches and 1
validation batch, saved epoch-1 weights, and stopped at max_epochs=1.
failure_reason: none; the Hydra append override repair fixed the prior
`trainer.limit_train_batches` struct error.
next_action: no train-smoke repair needed.
```

Pre-submit compatibility check for full upstream PWM real-env eval:

```text
existing eval bridge compatibility:
  `eval_policy_checkpoint.py` supports generic Flow-MBPO checkpoints with
  embedded `args` and original PWM adapter checkpoints with QS metadata. The
  full upstream PWM checkpoint from `9401906` has keys
  actor, critic, world_model, obs_rms, rew_rms, ret_rms, actor_opt,
  critic_opt, world_model_opt and no `args`; auto-detection would route it to
  the QS adapter format even though it was trained on 210-dim online MJLab
  observations.
new diagnostic script:
  scripts/experiments/mjlab_qs/eval_upstream_pwm_mjlab_checkpoint.py
validation:
  py_compile passed; locked bridge `--help` passed; H200/embers test-only
  accepted.
candidate: submit W&B-disabled final/best 8-episode real-env eval smoke before
  considering any formal 40-episode eval/video gate for full upstream PWM.
```

Submission record for full upstream PWM real-env eval smoke:

```text
commit: b889f7b Add upstream PWM real eval smoke.
submitted_job: 9401975_[0-1%2] upstream_pwm_mjlab_real_eval_smoke_h200_20260602.
resource: gpu-h200 / embers, gpu:h200:1, 8 CPUs, 96G, 01:00:00.
initial_state: PENDING, reason Resources.
inputs_verified_before_submit: Hydra config, final_policy.pt, and best_policy.pt
from `9401906` exist and are non-empty.
```

Backup submission for full upstream PWM real-env eval smoke:

```text
reason: H200 job `9401975_[0-1%2]` was pending for Resources; H100 test-only
reported a concrete eligible start time, so the same eval smoke was submitted
to H100 with a distinct output root to avoid artifact overlap.
commit: 31e9212 Parameterize upstream PWM eval smoke GPU label.
submitted_job: 9401980_[0-1%2] upstream_pwm_mjlab_real_eval_smoke_h100_20260602.
resource: gpu-h100 / embers, gpu:h100:1, 8 CPUs, 96G, 01:00:00.
initial_state: PENDING, reason Priority.
```

Additional scheduler preflight for duplicate PWM eval backups:

```text
checked: A100 and L40S as lower-priority alternatives while H200/H100 PWM
real-env eval smoke jobs were pending.
A100_result: test-only accepted but predicted 2026-06-07T00:27:42.
L40S_result: 8 CPU request rejected by the partition CPU:GPU ratio; 4 CPU / 64G
test-only accepted but predicted 2026-06-07T13:13:41.
decision: no additional backup submitted because both alternatives were much
later than the already-submitted H100 backup and would duplicate the same
final/best checkpoint eval.
```

Validation details:

```text
direct locked env:
  python: /storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4/bin/python
  torch: 2.3.1 from locked env
  pwm: baselines/PWM/src/pwm
  mjlab: ModuleNotFoundError
  flow_mbpo_pwm: ModuleNotFoundError

locked bridge:
  scripts/experiments/mjlab_qs/locked_mjlab_python.py imports locked torch/PWM first,
  then exposes project-env site-packages for MJLab and local adapter imports.
  Verified torch 2.3.1 from locked env; PWM class module `pwm.algorithms.pwm`;
  MJLab import from project env; adapter factory from `flow_mbpo_pwm.envs.mjlab_pwm_adapter`.

Hydra compose:
  `locked_mjlab_python.py train_dflex.py --cfg job env=mjlab_velocity_flat_unitree_g1 alg=pwm ...`
  shows `alg._target_: pwm.algorithms.pwm.PWM` and env target
  `flow_mbpo_pwm.envs.mjlab_pwm_adapter.create_mjlab_pwm_env`.

Scheduler validation:
  `sbatch --test-only` accepted the H200 / embers resource request.
```

## Continuation Inventory: Full PWM Eval Smoke Pending

Preflight time: 2026-06-02 after commit `79ec3a0`, America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD before this edit | `79ec3a0` |
| Dirty status before this inventory edit | clean |
| Slurm commands | `squeue -u $USER -o ...`; `sacct -j 9401975,9401980 --format=... -P`; `seff` unavailable earlier and not used in this pass |
| Artifact searches | Checked full upstream PWM real-env eval output roots, Slurm queue state, and current docs/git evidence records after updating the longdiag status. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9401975_[0-1%2]` | Full upstream PWM final/best real-env eval smoke after longdiag `9401906` | `PENDING`, reason `Resources`; no output files yet | `scripts/experiments/mjlab_qs/submit_upstream_pwm_mjlab_real_eval_smoke_20260602.sh` | `b889f7b` script commit; `6318d7e` submission record | Hydra config from `baselines/PWM/scripts/outputs/2026-06-02/21-35-04/.hydra/config.yaml`; evaluator overrides `eval_episodes=8`, `eval_num_envs=16`, `max_steps=1000`, W&B disabled | Locked original PWM env via `locked_mjlab_python.py`, upstream PWM checkpoint schema, MJLab online task `Mjlab-Velocity-Flat-Unitree-G1`; no QS-window dataset | seed `0`; array row 0 final, row 1 best | H200 / `embers` | W&B disabled | `baselines/PWM/scripts/outputs/2026-06-02/21-35-04/logs/upstream_pwm_mjlab_full_longdiag_h200_seed0_20260602/{final_policy.pt,best_policy.pt}` | Expected under `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_real_eval_smoke_20260602/{final,best}/` | Not available | None yet | Pending only | Wait for first completed row; do not submit formal eval/video until this checkpoint loader smoke succeeds |
| `9401980_[0-1%2]` | H100 backup for the same full upstream PWM final/best real-env eval smoke | `PENDING`, reason `Priority`; no output files yet | Same script with `PARTITION=gpu-h100`, `GPU_GRES=gpu:h100:1`, `GPU_LABEL=h100`, and a distinct `OUTPUT_ROOT` | `31e9212` GPU-label parameterization commit; `8af421d` backup submission record | Same Hydra config/checkpoints and evaluator overrides as `9401975` | Same locked-PWM/MJLab bridge and online task | seed `0`; array row 0 final, row 1 best | H100 / `embers` | W&B disabled | Same final/best checkpoints from `9401906` | Expected under `scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_real_eval_smoke_h100_20260602/{final,best}/` | Not available | None yet | Pending only | Keep as backup because A100/L40S test-only starts were much later; first completed smoke determines whether formal gates are worth submitting |
| `9400409_[0-15]` | Official NEWT broad A100 smoke | `PENDING`, reason `Priority` | Existing official NEWT broad submission | Earlier image-official submission commits; no new change in this pass | Official NEWT 500-step broad smoke, W&B disabled | Official NEWT env/repo | seeds `0`, `1` across tasks | A100 / `embers` | W&B disabled | No checkpoint expected | Logs expected under `logs/slurm/image_official/` | Not available | None yet | Pending only | Leave queued; do not duplicate because L40S/H200 NEWT smoke coverage already exists |
| `9400442`, `9400528_[1-2%2]` | Remaining A100 Flow-MBPO AWR shard rows | `PENDING`, reason `Priority` | Existing A100 shard submissions | Earlier AWR shard submission commits; no new change in this pass | Conservative Flow-MBPO AWR shard manifests | Project `pwm` env, MJLab QS H16 data and replay/support inputs | seed `1` rows | A100 / `embers` | W&B disabled | Expected per-row AWR checkpoints under A100 shard output root | Expected summaries under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_shards_20260602/a100/` | Not available | None yet | Pending only | Leave queued; completed H200/H100/L40S AWR rows are already negative diagnostics |

Submit decision in this poll: no new sbatch submission. The two full upstream
PWM real-env eval smokes are already queued, no summary exists yet, and
additional A100/L40S duplicate backup preflight was previously worse than the
H100 backup.

Full upstream PWM eval smoke completion:

```text
9401975_0 final_policy.pt COMPLETED 0:0, elapsed 00:00:26, gpu-h200 / embers.
result: return_mean -1.7458, episode_length_mean 38.125, fall_rate_mean 1.000,
timeout_rate_mean 0.000, baseline_gate_pass false.

9401975_1 best_policy.pt COMPLETED 0:0, elapsed 00:00:39, gpu-h200 / embers.
result: return_mean -1.6728, episode_length_mean 47.500, fall_rate_mean 1.000,
timeout_rate_mean 0.000, baseline_gate_pass false.

9401980_[0-1%2] H100 backup was canceled after the H200 final/best smoke
completed, because it would duplicate the same checkpoint/evaluator evidence.

usable: yes, as proof that full upstream PWM longdiag checkpoints can be loaded
and evaluated on real MJLab through the locked-PWM bridge. No, as a performance
claim or formal MJLab gate.
next_action: record this as negative full-upstream-pipeline smoke evidence; do
not submit W&B/formal eval40/video gates for this diagnostic checkpoint.
```

## Continuation Inventory: Existing V1 Flow-MBPO Support Results

Preflight time: 2026-06-02 after commit `9ae1c06`, America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD before this edit | `9ae1c06` |
| Dirty status before this inventory edit | clean |
| Slurm commands | `squeue -u $USER -o ...`; `sacct -j 9400409,9400528,9400442,9401975,9401980 --format=... -P`; `seff` unavailable earlier and not used in this pass |
| Artifact searches | Checked existing Flow-MBPO v1 AWR/support summaries, v1 CQL/OOD manifest, broad AWR manifests, support penalty/truncation scripts, and active A100 queue state. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local export | Aggregate existing Flow-MBPO v1 AWR/support results | Completed locally; 26 summaries exported | `python scripts/experiments/mjlab_qs/export_flow_mbpo_awr_summary.py --root scripts/outputs/mjlab_qs/flow_mbpo_v1_awr_smoke --root scripts/outputs/mjlab_qs/flow_mbpo_v1_awr --output-csv scripts/outputs/mjlab_qs/status/flow_mbpo_v1_awr_support_summary_20260602.csv --output-md scripts/outputs/mjlab_qs/status/flow_mbpo_v1_awr_support_summary_20260602.md` | Mixed historic run SHAs in summaries; current export run after `9ae1c06` | Existing v1 AWR/support summaries, no new training | Project `pwm` env; MJLab QS H16 dataset and existing Flow-MBPO synthetic replay artifacts | mostly seed `0` | No GPU for export | Existing rows mostly W&B disabled; one v1 W&B row had W&B enabled in its original summary | Existing final/best/checkpoints under `scripts/outputs/mjlab_qs/flow_mbpo_v1_awr_smoke/` and `scripts/outputs/mjlab_qs/flow_mbpo_v1_awr/` | Existing real-eval snapshots where present | High-return rows are 1-update / 2-episode gate-logging smokes from the BC checkpoint (`84.35`-`84.43`, length `1000`, fall `0`), not Flow-MBPO improvement evidence. Substantive rows are negative: CQL random-action eval8 return `18.7727`, length `283.5`, fall `1.0`; 500-iter action-deviation row best return `19.577`, length `295.0`, fall `1.0`. | None for export | Yes as branch-decision evidence; not a formal claim | Do not submit another duplicate support/AWR row. Design a stricter fall-stop/support-truncation manifest before any new GPU job. |
| `9400409_[0-15]` | Official NEWT broad A100 smoke | `PENDING`, reason `Priority` | Existing official NEWT broad submission | Earlier image-official submission commits | Official NEWT 500-step broad smoke | Official NEWT env/repo | seeds `0`, `1` across tasks | A100 / `embers` | W&B disabled | No checkpoint expected | Logs expected under `logs/slurm/image_official/` | Not available | None yet | Pending only | Leave queued |
| `9400442`, `9400528_[1-2%2]` | Remaining A100 Flow-MBPO AWR shard rows | `PENDING`, reason `Priority` | Existing A100 shard submissions | Earlier AWR shard submission commits | Conservative Flow-MBPO AWR shard manifests | Project `pwm` env, MJLab QS H16 data and replay/support inputs | seed `1` rows | A100 / `embers` | W&B disabled | Expected per-row AWR checkpoints under A100 shard output root | Expected summaries under `scripts/outputs/mjlab_qs/flow_mbpo_broad_embers_awr_shards_20260602/a100/` | Not available | None yet | Pending only | Leave queued; no duplicate broad AWR submission |

Submit decision in this poll: no new sbatch submission. Existing v1
support/pessimistic smoke evidence is already negative for substantive update
rows, and active A100 jobs remain queued.

Support-truncation diagnostic candidate preflight:

```text
new_manifest:
  scripts/experiments/mjlab_qs/manifests/flow_mbpo_support_trunc_awr_diag_20260602.csv
rows: 2.
inputs_verified:
  MJLab QS H16 dataset/metadata/normalization exist.
  BC seed0 final policy checkpoint exists.
  q0.90 and q0.50 state-support truncated trajectory H3 synthetic replays exist.
  q0.90 replay done fraction after support truncation: 0.171875.
  q0.50 replay done fraction after support truncation: 0.484375.
validation:
  CSV sanity check passed.
  `run_flow_mbpo_awr_row.py --check-inputs --dry-run` passed for rows 0 and 1.
  `submit_array.sh --kind flow_mbpo_awr --dry-run` produced a valid H200 array command.
  H200/embers `sbatch --test-only` accepted the request.
submit_decision: commit first, then submit W&B-disabled H200 diagnostic array
because it changes the fall/OOD intervention beyond the already-negative broad
AWR/support rows.
```

Support-truncation diagnostic submission record:

```text
commit: 23cbdbd Add Flow MBPO support truncation diagnostic manifest.
submitted_job: 9402080_[0-1%2] mjqs_flow_mbpo_awr_H200.
resource: gpu-h200 / embers, gpu:h200:1, 8 CPUs, 128G, 02:00:00.
initial_state: PENDING, reason None.
array rows:
  0 q0.90 state-support truncation, mixed CQL, real_eval_every 20.
  1 q0.50 state-support truncation, mixed CQL, real_eval_every 20.
wandb: disabled.
dependencies: none.
```

## Continuation Inventory: Support-Truncation H100 Backup

Preflight time: 2026-06-02 after commit `aaf657d`, America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD before this edit | `aaf657d` |
| Dirty status before this inventory edit | new H100 backup manifest only |
| Slurm commands | `squeue -u $USER -o ...`; H100/A100/L40S `sbatch --test-only` probes |
| Artifact searches | Checked H200 support-truncation queue state, H100 backup manifest rows/output roots, and existing Flow-MBPO support-truncation replay inputs. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9402080_0` | H200 Flow-MBPO support-truncation diagnostic, q0.90 replay | `RUNNING` at preflight poll | `submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_support_trunc_awr_diag_20260602.csv --gpu-type H200 --partition gpu-h200 --qos embers --max-concurrent 2 --time 02:00:00 --mem 128G --cpus 8` | `23cbdbd` manifest commit; submission recorded at `aaf657d` | q0.90 state-support truncated trajectory H3 replay, mixed CQL, real eval every 20, early stop on poor return/length/fall score | Project `pwm` env; MJLab QS H16 data; BC seed0 policy; support-truncated synthetic replay | seed `0` | H200 / `embers` | W&B disabled | Expected under `scripts/outputs/mjlab_qs/flow_mbpo_support_trunc_awr_diag_20260602/state_support_q90_trunc_cql_mixed_evalstop_s0/` | Expected `summary.json` and real-eval snapshots | Not available yet | None yet | Running only | Inspect summary/log after completion |
| `9402080_1` | H200 Flow-MBPO support-truncation diagnostic, q0.50 replay | `PENDING`, reason `Resources` at preflight poll | Same H200 array | `23cbdbd` / `aaf657d` | q0.50 state-support truncated trajectory H3 replay, mixed CQL, real eval every 20, early stop on poor return/length/fall score | Same project env and MJLab QS inputs | seed `0` | H200 / `embers` | W&B disabled | Expected under `scripts/outputs/mjlab_qs/flow_mbpo_support_trunc_awr_diag_20260602/state_support_q50_trunc_cql_mixed_evalstop_s0/` | Expected `summary.json` and real-eval snapshots | Not available yet | None yet | Pending only | H100 backup is justified if committed before submission because this row is still blocked on H200 resources |
| `flow_mbpo_support_trunc_awr_diag_h100_20260602` | H100 backup manifest for the same two support-truncation diagnostics | Prepared locally; not submitted before this commit | `submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_support_trunc_awr_diag_h100_20260602.csv --gpu-type H100 --partition gpu-h100 --qos embers --max-concurrent 2 --time 02:00:00 --mem 128G --cpus 8` | Pending commit | Same two q0.90/q0.50 support-truncated replay rows as H200, but with distinct output root `scripts/outputs/mjlab_qs/flow_mbpo_support_trunc_awr_diag_h100_20260602/` | Same project `pwm` env and MJLab QS inputs; no upstream-PWM locked env is needed for this Flow-MBPO AWR diagnostic | seed `0` | H100 / `embers` | W&B disabled | Expected per-row final/best/best-training policies and critic checkpoints under the H100 output root | Expected per-row `summary.json` and real-eval snapshots | Not available yet | None yet | Candidate only | Commit manifest and docs, then submit H100 backup; cancel duplicate pending/running work if either H200 or H100 completes first with decisive summaries |

Scheduler alternative notes:

```text
H100 / embers, 8 CPUs, 128G, 02:00:00: accepted by sbatch --test-only,
predicted start 2026-06-03T14:10:03.
A100 / embers: accepted but predicted 2026-06-06T19:13:42.
L40S / embers with 4 CPUs / 96G: accepted but predicted 2026-06-07T12:49:41.
decision: use H100 backup only; do not add A100/L40S duplicates for this
two-row diagnostic.
```

## Continuation Inventory: Support-Truncation Env Fix

Preflight time: 2026-06-02 after commit `eceae57`, America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD before this edit | `eceae57` |
| Dirty status before this inventory edit | clean |
| Slurm commands | `sacct -j 9402080 --format=... -P`; `squeue -u $USER -o ...`; H200 `sbatch --test-only` |
| Artifact searches | Checked `logs/slurm/mjlab_qs/flow_mbpo_awr/mjqs_flow_mbpo_awr_9402080_{0,1}.{out,err}` and the expected support-truncation output root. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9402080_0` | H200 Flow-MBPO support-truncation diagnostic, q0.90 replay | `FAILED`, exit `1:0`, elapsed `00:00:13` | Original H200 support-truncation submission without explicit conda activation | `23cbdbd` manifest; submission recorded at `aaf657d` | q0.90 state-support truncated trajectory H3 replay, mixed CQL, real eval every 20, early stop | Incorrect default login Python 3.13 was used by the Slurm wrapper; intended env was project `pwm` | seed `0` | H200 / `embers` | W&B disabled | None written | None written | Not available | `ModuleNotFoundError: No module named 'omegaconf'` before training because the job did not activate `pwm` | No; infrastructure failure only | Replace with the same manifest plus `--conda-env pwm` |
| `9402080_1` | H200 Flow-MBPO support-truncation diagnostic, q0.50 replay | `FAILED`, exit `1:0`, elapsed `00:00:03` | Same original H200 array | `23cbdbd` / `aaf657d` | q0.50 state-support truncated trajectory H3 replay, mixed CQL, real eval every 20, early stop | Same incorrect default login Python 3.13; intended env was project `pwm` | seed `0` | H200 / `embers` | W&B disabled | None written | None written | Not available | Same missing `omegaconf` import before training | No; infrastructure failure only | Replace with the same manifest plus `--conda-env pwm` |
| `flow_mbpo_support_trunc_awr_diag_20260602_fixenv` | H200 replacement for the same support-truncation diagnostics | Prepared; not submitted before this commit | `submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_support_trunc_awr_diag_20260602.csv --gpu-type H200 --partition gpu-h200 --qos embers --max-concurrent 2 --time 02:00:00 --mem 128G --cpus 8 --conda-env pwm` | Pending commit | Same two support-truncated replay rows; output roots were not created by the failed job, so reuse is clean | Project `pwm` env validated: Python 3.10.19, OmegaConf 2.3.0, torch 2.10.0+cu128 | seed `0` | H200 / `embers` | W&B disabled | Expected per-row AWR checkpoints under `scripts/outputs/mjlab_qs/flow_mbpo_support_trunc_awr_diag_20260602/` | Expected per-row `summary.json` and real-eval snapshots | Not available yet | None yet | Candidate replacement | Commit failure record, submit H200 replacement, then monitor first row |

Replacement validation:

```text
`conda activate pwm` import check passed for omegaconf and torch.
`submit_array.sh ... --conda-env pwm --dry-run` showed the Slurm wrap now
activates `pwm` before running `run_flow_mbpo_awr_row.py`.
H200 / embers `sbatch --test-only` accepted the fixed-env resource request.
```

Fixed-env replacement submission:

```text
commit_before_submit: f6beceb Record Flow MBPO support truncation env failure.
submitted_job: 9402128_[0-1%2] mjqs_flow_mbpo_awr_H200.
resource: gpu-h200 / embers, gpu:h200:1, 8 CPUs, 128G, 02:00:00.
command:
  scripts/experiments/mjlab_qs/submit_array.sh --kind flow_mbpo_awr \
    --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_support_trunc_awr_diag_20260602.csv \
    --gpu-type H200 --partition gpu-h200 --qos embers --max-concurrent 2 \
    --time 02:00:00 --mem 128G --cpus 8 --conda-env pwm
initial_squeue: PENDING, reason Resources.
initial_sacct: PENDING, QOS embers, partition gpu-h200.
wandb: disabled.
next_action: inspect logs as soon as a row starts; if H200 remains
resource-blocked, use the already-committed H100 backup manifest.
```

H100 backup preflight before submission:

```text
current_h200_replacement: 9402128_[0-1%2] PENDING, reason Resources.
H100 / embers `sbatch --test-only`: accepted, predicted start
2026-06-03T10:37:39.
`submit_array.sh ... --gpu-type H100 --conda-env pwm --dry-run` confirmed the
Slurm wrapper activates the project `pwm` env before row execution.
H100 backup row 0 and row 1 `--check-inputs --dry-run` passed.
decision: submit the H100 backup manifest because it uses distinct output dirs
and is the earliest scheduler-backed replacement path.
```

H100 backup submission:

```text
commit_before_submit: 661ccac Record Flow MBPO support truncation H100 preflight.
submitted_job: 9402136_[0-1%2] mjqs_flow_mbpo_awr_H100.
resource: gpu-h100 / embers, gpu:h100:1, 8 CPUs, 128G, 02:00:00.
command:
  scripts/experiments/mjlab_qs/submit_array.sh --kind flow_mbpo_awr \
    --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_support_trunc_awr_diag_h100_20260602.csv \
    --gpu-type H100 --partition gpu-h100 --qos embers --max-concurrent 2 \
    --time 02:00:00 --mem 128G --cpus 8 --conda-env pwm
initial_squeue: PENDING, reason Priority.
initial_sacct: PENDING, QOS embers, partition gpu-h100.
paired_h200_replacement: 9402128_[0-1%2] PENDING, reason Resources.
wandb: disabled.
next_action: monitor both arrays; cancel duplicate pending/running work after
the first complete diagnostic summaries are available.
```

## Continuation Inventory: NEWT A100 Row 0 And Pending Support-Truncation

Preflight time: 2026-06-02 after commit `7a32e49`, America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD before this edit | `7a32e49` |
| Dirty status before this inventory edit | clean |
| Slurm commands | `squeue -u $USER -o ...`; `sacct -j 9400409,9402128,9402136 --format=... -P`; `seff` unavailable earlier and not used |
| Artifact searches | Checked NEWT A100 Slurm logs, support-truncation output roots, and current pending H200/H100 support-truncation arrays. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9400409_0` | Official NEWT broad A100 smoke, walker-walk seed0 | `COMPLETED`, exit `0:0`, elapsed `00:00:36` | Official NEWT broad smoke payload from `scripts/experiments/image_official/submit_newt_lewm_broad_gpu_20260602.sh` | Earlier official-image submission commits; no new code in this poll | Official NEWT `tdmpc2/train.py`, task `walker-walk`, seed 0, `steps=500`, `model_size=B`, W&B disabled, checkpoint/video disabled | Official NEWT repo `/storage/project/r-agarg35-0/eliu354/external_repos/newt`; official env `/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602`; data dir `/storage/project/r-agarg35-0/eliu354/external_data/newt_demos` | seed `0` | A100 / `embers` | W&B disabled | No checkpoint expected | Log `logs/slurm/image_official/newt_official_broad_smoke_9400409_0.out`; no video expected | Official smoke log reports eval reward `42.248` at step 0 and train reward `51.202` at step 500; no MJLab length/fall metrics because this is not an MJLab job | None; stderr only includes upstream Gym deprecation warnings | Yes as official NEWT infrastructure smoke evidence only; not performance evidence and not a PWM/MJLab result | Keep remaining `9400409_[1-15%8]` rows queued; do not duplicate while pending |
| `9400409_1` | Official NEWT broad A100 smoke, walker-run seed0 | `COMPLETED`, exit `0:0`, elapsed `00:00:29` | Same official NEWT broad smoke payload | Earlier official-image submission commits; no new code in this poll | Official NEWT `tdmpc2/train.py`, task `walker-run`, seed 0, `steps=500`, `model_size=B`, W&B disabled, checkpoint/video disabled | Same official NEWT repo/env/data as row 0 | seed `0` | A100 / `embers` | W&B disabled | No checkpoint expected | Log `logs/slurm/image_official/newt_official_broad_smoke_9400409_1.out`; no video expected | Official smoke log reports eval reward `42.179` at step 0 and train reward `23.809` at step 500; no MJLab length/fall metrics | None; stderr only includes upstream Gym deprecation warnings | Yes as official NEWT infrastructure smoke evidence only; not performance evidence and not a PWM/MJLab result | Keep remaining rows queued; do not duplicate while pending |
| `9400409_2` | Official NEWT broad A100 smoke, cheetah-run seed0 | `COMPLETED`, exit `0:0`, elapsed `00:00:36` | Same official NEWT broad smoke payload | Earlier official-image submission commits; no new code in this poll | Official NEWT `tdmpc2/train.py`, task `cheetah-run`, seed 0, `steps=500`, `model_size=B`, W&B disabled, checkpoint/video disabled | Same official NEWT repo/env/data as row 0 | seed `0` | A100 / `embers` | W&B disabled | No checkpoint expected | Log `logs/slurm/image_official/newt_official_broad_smoke_9400409_2.out`; no video expected | Official smoke log reports eval reward `5.871` at step 0 and train reward `6.516` at step 500; no MJLab length/fall metrics | None; stderr only includes upstream Gym deprecation warnings | Yes as official NEWT infrastructure smoke evidence only; not performance evidence and not a PWM/MJLab result | Keep remaining rows queued; do not duplicate while pending |
| `9400409_3` | Official NEWT broad A100 smoke, hopper-hop seed0 | `COMPLETED`, exit `0:0`, elapsed `00:00:32` | Same official NEWT broad smoke payload | Earlier official-image submission commits; no new code in this poll | Official NEWT `tdmpc2/train.py`, task `hopper-hop`, seed 0, `steps=500`, `model_size=B`, W&B disabled, checkpoint/video disabled | Same official NEWT repo/env/data as row 0 | seed `0` | A100 / `embers` | W&B disabled | No checkpoint expected | Log `logs/slurm/image_official/newt_official_broad_smoke_9400409_3.out`; no video expected | Official smoke log reports eval reward `0.000` at step 0 and train reward `0.000` at step 500; no MJLab length/fall metrics | None; stderr only includes upstream Gym deprecation warnings | Yes as official NEWT infrastructure smoke evidence only; not performance evidence and not a PWM/MJLab result | Keep remaining rows queued; do not duplicate while pending |
| `9400409_4` | Official NEWT broad A100 smoke, reacher-easy seed0 | `COMPLETED`, exit `0:0`, elapsed `00:00:28` | Same official NEWT broad smoke payload | Earlier official-image submission commits; no new code in this poll | Official NEWT `tdmpc2/train.py`, task `reacher-easy`, seed 0, `steps=500`, `model_size=B`, W&B disabled, checkpoint/video disabled | Same official NEWT repo/env/data as row 0 | seed `0` | A100 / `embers` | W&B disabled | No checkpoint expected | Log `logs/slurm/image_official/newt_official_broad_smoke_9400409_4.out`; no video expected | Official smoke log reports eval reward `34.000` at step 0 and train reward `0.000` at step 500; no MJLab length/fall metrics | None; stderr only includes upstream Gym deprecation warnings | Yes as official NEWT infrastructure smoke evidence only; not performance evidence and not a PWM/MJLab result | Keep remaining rows queued; do not duplicate while pending |
| `9400409_5` | Official NEWT broad A100 smoke, pendulum-swingup seed0 | `COMPLETED`, exit `0:0`, elapsed `00:00:27` | Same official NEWT broad smoke payload | Earlier official-image submission commits; no new code in this poll | Official NEWT `tdmpc2/train.py`, task `pendulum-swingup`, seed 0, `steps=500`, `model_size=B`, W&B disabled, checkpoint/video disabled | Same official NEWT repo/env/data as row 0 | seed `0` | A100 / `embers` | W&B disabled | No checkpoint expected | Log `logs/slurm/image_official/newt_official_broad_smoke_9400409_5.out`; no video expected | Official smoke log reports eval reward `0.000` at step 0 and train reward `0.000` at step 500; no MJLab length/fall metrics | None; stderr only includes upstream Gym deprecation warnings | Yes as official NEWT infrastructure smoke evidence only; not performance evidence and not a PWM/MJLab result | Keep remaining rows queued; do not duplicate while pending |
| `9400409_6` | Official NEWT broad A100 smoke, cartpole-swingup seed0 | `COMPLETED`, exit `0:0`, elapsed `00:00:28` | Same official NEWT broad smoke payload | Earlier official-image submission commits; no new code in this poll | Official NEWT `tdmpc2/train.py`, task `cartpole-swingup`, seed 0, `steps=500`, `model_size=B`, W&B disabled, checkpoint/video disabled | Same official NEWT repo/env/data as row 0 | seed `0` | A100 / `embers` | W&B disabled | No checkpoint expected | Log `logs/slurm/image_official/newt_official_broad_smoke_9400409_6.out`; no video expected | Official smoke log reports eval reward `183.107` at step 0 and train reward `8.803` at step 500; no MJLab length/fall metrics | None; stderr only includes upstream Gym deprecation warnings | Yes as official NEWT infrastructure smoke evidence only; not performance evidence and not a PWM/MJLab result | Keep remaining rows queued; do not duplicate while pending |
| `9400409_[7-15%8]` | Remaining official NEWT broad A100 smoke rows | `PENDING`, reason `Priority` | Same official NEWT broad smoke payload | Existing submission record | Official NEWT broad smoke rows 7-15 | Same official NEWT repo/env/data | seeds/tasks by row | A100 / `embers` | W&B disabled | No checkpoint expected | Logs pending | Not available | None yet | Pending only | Wait; L40S/H200 coverage already provides infrastructure signal, so no duplicate submission is needed |
| `9402128_[0-1%2]` | H200 fixed-env Flow-MBPO support-truncation diagnostic | `PENDING`, reason `Resources` | `submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_support_trunc_awr_diag_20260602.csv --gpu-type H200 --partition gpu-h200 --qos embers --max-concurrent 2 --time 02:00:00 --mem 128G --cpus 8 --conda-env pwm` | `f6beceb` failure record, `dc08e7a` replacement submission | q0.90/q0.50 state-support truncated replay diagnostics | Project `pwm` env; MJLab QS H16 data; BC seed0 policy; support-truncated replays | seed `0` | H200 / `embers` | W&B disabled | Expected under `scripts/outputs/mjlab_qs/flow_mbpo_support_trunc_awr_diag_20260602/` | Expected per-row `summary.json` and real-eval snapshots | Not available | None yet | Pending only | Keep pending; cancel if H100 completes first and answers the diagnostic |
| `9402136_[0-1%2]` | H100 backup fixed-env Flow-MBPO support-truncation diagnostic | `PENDING`, reason `Priority` | `submit_array.sh --kind flow_mbpo_awr --manifest scripts/experiments/mjlab_qs/manifests/flow_mbpo_support_trunc_awr_diag_h100_20260602.csv --gpu-type H100 --partition gpu-h100 --qos embers --max-concurrent 2 --time 02:00:00 --mem 128G --cpus 8 --conda-env pwm` | `eceae57` H100 manifest, `7a32e49` H100 submission record | Same support-truncation diagnostics with distinct H100 output root | Same project `pwm` env and MJLab QS inputs | seed `0` | H100 / `embers` | W&B disabled | Expected under `scripts/outputs/mjlab_qs/flow_mbpo_support_trunc_awr_diag_h100_20260602/` | Expected per-row `summary.json` and real-eval snapshots | Not available | None yet | Pending only | Monitor both H200/H100 arrays; first complete summaries decide whether duplicate work should be canceled |

Submit decision in this poll: no new submission. The support-truncation H200
replacement and H100 backup are already queued with the corrected `pwm`
environment, and only official NEWT smoke rows completed.

## Continuation Inventory: Conda Wrapper Repair

Preflight time: 2026-06-02 after commit `5b0d3d7`, America/New_York.

| Field | Value |
| --- | --- |
| Branch | `mjlab-qs-rollout-policy-improvement` |
| Current HEAD before this edit | `5b0d3d7` |
| Dirty status before this inventory edit | clean before `submit_array.sh` repair |
| Slurm commands | `squeue -u $USER -o ...`; `sacct -j 9402128,9402136 --format=... -P`; `scancel 9402128` |
| Artifact searches | Checked H100 backup Slurm logs, support-truncation output roots, and current H200/H100 support-truncation queue state. |

| Job ID | Purpose | Status | Command / script | Git SHA | Config | Env / dataset / version | Seed | GPU / QOS | W&B link or offline dir | Checkpoint paths | Eval / video paths | Return / length / fall | Failure reason | Usable? | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `9402136_0` | H100 backup Flow-MBPO support-truncation diagnostic, q0.90 replay | `FAILED`, exit `2:0`, elapsed `00:00:04` | H100 backup submission using `submit_array.sh --conda-env pwm` before wrapper repair | `7a32e49` submission record | q0.90 state-support truncated trajectory H3 replay, mixed CQL, real eval every 20, early stop | Intended project `pwm` env, but conda activation never ran | seed `0` | H100 / `embers` | W&B disabled | None written | None written | Not available | Slurm wrapper sourced `~/.bashrc`; compute-node `.codex_project_profiles.sh` exited with `codex-project: not a valid identifier` before `conda activate pwm` | No; infrastructure failure only | Fix conda activation wrapper and replace |
| `9402136_1` | H100 backup Flow-MBPO support-truncation diagnostic, q0.50 replay | `FAILED`, exit `2:0`, elapsed `00:00:04` | Same H100 backup submission before wrapper repair | `7a32e49` submission record | q0.50 state-support truncated trajectory H3 replay, mixed CQL, real eval every 20, early stop | Intended project `pwm` env, but conda activation never ran | seed `0` | H100 / `embers` | W&B disabled | None written | None written | Not available | Same `~/.bashrc` / `.codex_project_profiles.sh` failure before row execution | No; infrastructure failure only | Fix conda activation wrapper and replace |
| `9402128_[0-1%2]` | H200 fixed-env support-truncation replacement affected by same wrapper bug | `CANCELLED by 3509929` while pending | H200 fixed-env replacement using `submit_array.sh --conda-env pwm` before wrapper repair | `dc08e7a` submission record | Same two support-truncation diagnostics | Would have used same failing `~/.bashrc` wrapper if started | seed `0` | H200 / `embers` | W&B disabled | None written | None written | Not available | Canceled proactively after H100 exposed shared wrapper failure | No; canceled infrastructure prevention | Replace after committing wrapper repair |
| local wrapper repair | Fix `submit_array.sh --conda-env` for MJLab QS arrays | Completed locally before commit | `scripts/experiments/mjlab_qs/submit_array.sh` | Pending commit | Conda activation snippet now uses `${HOME}/miniconda3/etc/profile.d/conda.sh` or `conda shell.bash hook`; no `source ~/.bashrc` | Local validation activates `pwm`: Python from `/storage/home/hcoda1/9/eliu354/r-agarg35-0/envs/pwm/bin/python`, OmegaConf `2.3.0` | n/a | n/a | n/a | n/a | n/a | n/a | None | Yes as infrastructure repair | Commit, then resubmit support-truncation diagnostics with the repaired wrapper |

Validation:

```text
bash -n scripts/experiments/mjlab_qs/submit_array.sh
submit_array.sh --kind flow_mbpo_awr ... --gpu-type H100 --conda-env pwm --dry-run
  confirmed the Slurm wrap no longer contains `source ~/.bashrc`.
local conda snippet activated `pwm` and imported OmegaConf 2.3.0.
```
