# Candidate Jobs Before New Submission

Date: 2026-06-02

Preflight inventory commit: `c44c53c Record PWM Flow preflight inventory`.

| Candidate | Type | Purpose | Inputs exist? | W&B mode | Expected artifacts | GPU / QOS | Dependency required? | Submit decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `original_pwm_adapter_phase3_eval40_final_best_20260602` | eval | Run 40-episode real MJLab eval for the faithful original PWM adapter final and best-imagined checkpoints from formal job `9387895` so fall metrics are logged by `eval_policy_checkpoint.py`. | Yes: final and best checkpoints exist under `scripts/outputs/mjlab_qs/original_pwm_adapter/original_pwm_adapter_phase3_formal_20260601/.../seed_0/`. | Enabled; project `flow-mbpo-mjlab-original-pwm-adapter-eval40`. | `summary.json`, `eval_episodes.csv`, W&B runs for final and best; output root `scripts/outputs/mjlab_qs/policy_evals/original_pwm_adapter_phase3_eval40_20260602/...`. | H200 preferred; `embers`. | No. | Submit. |
| `original_pwm_adapter_phase3_rollout10_final_best_20260602` | eval | Render 10-episode, 1000-step MP4/W&B rollout videos for the same final and best-imagined checkpoints. | Yes: same checkpoints as above. | Enabled; project `flow-mbpo-mjlab-original-pwm-adapter-rollout1000`. | `summary.json`, `rollout.mp4`, W&B video media for final and best; output root `scripts/outputs/mjlab_qs/policy_rollouts/original_pwm_adapter_phase3_rollout10_20260602/...`. | H200 preferred; `embers`. | No. | Submit. |
| `ant_locked_true_eval_repair` | diagnostic | Repair/resubmit Ant final/best true DFlex eval after jobs `9387422` and `9387942` failed during DFlex kernel rebuild. | Yes: final and best Ant checkpoints from job `9384344` exist. The repair is to use the known-good locked DFlex wrapper exports: job-local sandbox, `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`, `CUDAHOSTCXX=/usr/bin/g++`, and unset `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH`, `LIBRARY_PATH`, `GCC_EXEC_PREFIX`, and `COMPILER_PATH`. | Enabled for parity evidence under `flow-mbpo-pwm-fidelity`. | Eval summaries under `eval_results/pwm_phase1_ant_locked_h200_realenv_*_20260602`. | H200; `embers`. | No: checkpoints exist and the wrapper fix is known from prior successful locked jobs. | Submit repaired replacement. |
| `hopper_wm_vs_real_probe_repair` | diagnostic | Repair/resubmit Hopper WM-vs-real probe after jobs `9387423` and `9387949` failed during DFlex kernel rebuild. | Yes: Hopper final and best checkpoints from job `9383814` exist. Use the same repaired locked DFlex wrapper exports as the Ant eval. | Disabled. | Probe JSON under `eval_results/pwm_phase2_hopper_locked_probe_20260602/`. | H100; `embers`. | No: checkpoints exist and the wrapper fix is known from prior successful locked jobs. | Submit repaired replacement. |
| `R0-R4 controlled matrix` | formal | Matched one-variable A/B rows for faithful PWM, Flow WM only, Flow policy only, Flow WM+policy, and best current Flow reproduction. | Not fully prepared in this turn; faithful-PWM collapse package still needs eval/video and Flow rows need matched manifests. | W&B enabled for formal rows, disabled for any new-code smoke. | WM/prediction/calibration/grad/action/OOD/real eval/video metrics. | H200/H100/A100/L40S; `embers`. | Only if row artifacts are missing. | Defer until final/best evidence from the faithful adapter is recorded. |
| `pessimistic_short_horizon_flow_mbpo` | exploratory / diagnostic | If collapse/OOD is confirmed, run H=1/3/5 AWR/AWAC plus support/fall/OOD diagnostics. | Existing prior Flow-MBPO infrastructure and some artifacts exist, but this branch should wait for the faithful adapter final/best package and candidate ranking refresh. | W&B disabled for new smokes, enabled for formal candidates. | Replay diagnostics, support/fall/OOD metrics, eval/video artifacts. | H200/H100/A100/L40S; `embers`. | No for smokes with existing inputs; yes only for missing replay/checkpoint artifacts. | Defer for this submission batch. |
| `SIGReg_state_latent_tests` | diagnostic / code | Add LeWM-inspired SIGReg only after documenting objective, tensor shapes, and tests. | Implemented and tested in `src/flow_mbpo_pwm/utils/sigreg.py` with documentation in `docs/git/sigreg_objective_shapes_tests_20260602.md`. | W&B off for tests/smokes. | Unit tests cover finite loss, finite gradients, zero-weight no-op, constant-latent anti-collapse penalty, and latent variance/isotropy stats. | No GPU required unless later smoke needs it. | No Slurm dependency. | Done for CPU prerequisite; do not submit SIGReg GPU rows until the pending faithful-PWM evidence package is recorded and a fresh candidate list selects a specific row. |

## Replacement Candidates After Failed Infrastructure Jobs

Preflight before this replacement batch:

```text
branch: mjlab-qs-rollout-policy-improvement
head_sha: 4d823b32967b2c56803399b07de1f5522e87541a
dirty_before_edits: docs/goals/pwm_flow_sigreg_image_research_plan_20260602.md
active_related_squeue: none
```

Correct environments:

```text
original DFlex/PWM parity environment:
  /storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4
  called directly as ${ENV_DIR}/bin/python after exporting locked CUDA/compiler
  variables and copying env site-packages/dflex into a job-local sandbox.

MJLab adapter eval/video environment:
  conda environment pwm
  called through scripts/experiments/mjlab_qs/submit_array.sh after exporting
  PYTHONPATH=${PROJECT_ROOT}/src:${PROJECT_ROOT}/baselines/PWM/src:$PYTHONPATH.
  The upstream PWM path is required because the faithful adapter checkpoints
  pickle/import objects from the upstream `pwm` package.
```

| Candidate | Type | Purpose | Inputs exist? | W&B mode | Expected artifacts | GPU / QOS | Dependency required? | Submit decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `original_pwm_adapter_phase3_eval40_final_best_fix1_20260602` | eval | Re-run faithful original PWM adapter final/best 40-episode MJLab eval after jobs `9388552_[0-1]` failed before evaluation with `ModuleNotFoundError: No module named 'pwm'`. | Yes: final and best checkpoints from job `9387895` exist and CPU `torch.load(..., weights_only=False)` succeeds with `PYTHONPATH=src:baselines/PWM/src`. | Enabled; project `flow-mbpo-mjlab-original-pwm-adapter-eval40`. | `summary.json`, `eval_episodes.csv`, W&B runs under `scripts/outputs/mjlab_qs/policy_evals/original_pwm_adapter_phase3_eval40_fix1_20260602/...`. | H200; `embers`. | No. | Submit after committing wrapper/manifests. |
| `original_pwm_adapter_phase3_rollout10_final_best_fix1_20260602` | eval / video | Re-run faithful original PWM adapter final/best 10-episode, 1000-step rollout videos after jobs `9388553_[0-1]` failed before rendering with `ModuleNotFoundError: No module named 'pwm'`. | Yes: same checkpoints as above; fixed wrapper includes upstream PWM import path. | Enabled; project `flow-mbpo-mjlab-original-pwm-adapter-rollout1000`. | `summary.json`, `rollout.mp4`, W&B video media under `scripts/outputs/mjlab_qs/policy_rollouts/original_pwm_adapter_phase3_rollout10_fix1_20260602/...`. | H200; `embers`. | No. | Submit after committing wrapper/manifests. |
| `pwm_ant_locked_realenv_eval_h200_fix3` | diagnostic / eval | Re-run Ant locked final/best true DFlex eval after jobs `9387942` and `9388605` failed during infrastructure setup. Fix2 got past missing `cc1plus` but failed because system header `CPATH` was not restored. | Yes: final and best Ant actors from job `9384344` exist; config `baselines/PWM/scripts/outputs/2026-06-01/23-06-24/.hydra/config.yaml` exists. | Enabled; project `flow-mbpo-pwm-fidelity`. | Eval summaries under `eval_results/pwm_phase1_ant_locked_h200_realenv_{final,best}_20260602_fix3/`. | H200; `embers`. | No. | Submit after committing wrapper. |
| `pwm_hopper_locked_wmprobe_h100_fix3` | diagnostic | Re-run Hopper final/best WM-vs-real probe after jobs `9387949` and `9388606` failed during DFlex kernel rebuild. Fix3 restores the explicit GCC 11 system include `CPATH`. | Yes: final and best Hopper actors from job `9383814` exist. | Disabled. | Probe JSON files `final_actor_wm_vs_real_fix3.json` and `best_actor_wm_vs_real_fix3.json`. | H100; `embers`. | No. | Submit after committing wrapper. |
| `original_pwm_adapter_phase3_eval40_final_best_fix2_20260602` | eval | Re-run faithful original PWM adapter final/best 40-episode MJLab eval after fix1 jobs `9394870_[0-1]` failed with `KeyError: 'args'` before evaluation. | Yes: final and best upstream PWM adapter checkpoints from job `9387895` exist; metadata reports `phys_obs_dim=96`, `command_dim=3`, `act_dim=29`; checkpoint world model input is 99 and CPU state-dict load succeeds after rebuilding the upstream PWM agent. | Enabled; project `flow-mbpo-mjlab-original-pwm-adapter-eval40`. | `summary.json`, `eval_episodes.csv`, W&B runs under `scripts/outputs/mjlab_qs/policy_evals/original_pwm_adapter_phase3_eval40_fix2_20260602/...`. | H200; `embers`. | No. | Submit after committing schema repair/manifests. |
| `original_pwm_adapter_phase3_rollout10_final_best_fix2_20260602` | eval / video | Re-run faithful original PWM adapter final/best 10-episode, 1000-step rollout videos after fix1 rollout row `9394871_0` failed with `KeyError: 'args'` and sibling row `9394871_1` was canceled. | Yes: same checkpoints and schema validation as eval fix2; row runners now forward explicit dataset, metadata, normalization, task, command, and obs-mode fields. | Enabled; project `flow-mbpo-mjlab-original-pwm-adapter-rollout1000`. | `summary.json`, `rollout.mp4`, W&B video media under `scripts/outputs/mjlab_qs/policy_rollouts/original_pwm_adapter_phase3_rollout10_fix2_20260602/...`. | H200; `embers`. | No. | Submit after committing schema repair/manifests. |

Submitted replacement job IDs:

```text
9394870 original_pwm_adapter_phase3_eval40_final_best_fix1_20260602
9394871 original_pwm_adapter_phase3_rollout10_final_best_fix1_20260602
9394869 pwm_ant_locked_realenv_eval_h200_fix3
9394872 pwm_hopper_locked_wmprobe_h100_fix3
```

Fix1 follow-up:

```text
9394870_[0-1] failed with KeyError: 'args' before evaluation artifacts.
9394871_0 failed with KeyError: 'args' before rollout artifacts.
9394871_1 was canceled after the same schema root cause was confirmed.
```

Submitted fix2 replacement job IDs:

```text
9395746 original_pwm_adapter_phase3_eval40_final_best_fix2_20260602
9395745 original_pwm_adapter_phase3_rollout10_final_best_fix2_20260602
```

## Replacement Candidates After Faithful Gate Results

Preflight before this replacement batch:

```text
branch: mjlab-qs-rollout-policy-improvement
head_sha_before_edits: ef6ab1f9ae3cd90591cf041086a0eedcdb28f61e
dirty_before_edits: docs/goals/pwm_flow_sigreg_image_research_plan_20260602.md
active_related_squeue: none
checked_recent_jobs: 9394872 failed, 9396164 failed, 9396165 failed, 9395746 completed, 9396189 completed
```

Candidate list before submission:

| Candidate | Type | Purpose | Inputs exist? | W&B mode | Expected artifacts | GPU / QOS | Dependency required? | Submit decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pwm_hopper_locked_wmprobe_h100_fix4` | diagnostic | Re-run Hopper final/best WM-vs-real probe after job `9394872` reached DFlex kernel/eval setup but failed because `PWM.load(..., with_buffer=False)` is incompatible with locked original PWM, whose API is `PWM.load(path, buffer=False)`. | Yes: final and best Hopper actors from locked formal job `9383814` exist under `baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601/`. | Disabled. | `eval_results/pwm_phase2_hopper_locked_probe_20260602/final_actor_wm_vs_real_fix4.json` and `best_actor_wm_vs_real_fix4.json`, plus Slurm logs under `logs/pwm_original_parity/locked_env_20260601/`. | H100; `embers`. | No. | Submit after committing probe compatibility and wrapper. |
| `original_pwm_adapter_hybrid_locked_nowandb_eval4_final_best_20260602` | smoke / diagnostic | Re-run short final/best hybrid locked MJLab eval after jobs `9396164_[0-1]` failed before evaluation because the mixed runtime W&B backend could not import `wandb.sdk.internal.internal`. This isolates runtime/eval behavior from W&B packaging. | Yes: final and best faithful adapter checkpoints from job `9387895` exist; dataset, metadata, and normalization files exist. | Disabled by manifest field `disable_wandb=true`. | Final/best `summary.json` and `eval_episodes.csv` under `scripts/outputs/mjlab_qs/policy_evals/original_pwm_adapter_hybrid_locked_nowandb_eval4_20260602/...`. | H200; `embers`. | No. | Submit after committing manifests and wrapper user-site isolation. |
| `original_pwm_adapter_hybrid_locked_nowandb_rollout1_final_best_20260602` | smoke / diagnostic video | Re-run short final/best hybrid locked MJLab render after jobs `9396165_[0-1]` failed before rollout because of the same W&B backend import problem. This checks whether hybrid locked runtime can render MJLab without W&B. | Yes: same checkpoints and MJLab QS files as eval smoke. | Disabled by manifest field `disable_wandb=true`. | Final/best `summary.json`, `rollout.mp4`, and rollout CSVs under `scripts/outputs/mjlab_qs/policy_rollouts/original_pwm_adapter_hybrid_locked_nowandb_rollout1_20260602/...`. | H200; `embers`. | No. | Submit after committing manifests and wrapper user-site isolation. |

Validation before submission:

```text
python -m py_compile scripts/diagnostics/pwm_dflex_checkpoint_probe.py scripts/experiments/mjlab_qs/locked_mjlab_python.py
bash -n scripts/experiments/mjlab_qs/submit_hopper_wmprobe_fix4_20260602.sh
locked env helper smoke: load_full_checkpoint calls PWM.load(path, buffer=False) for the locked API and PWM.load(path, with_buffer=False) for newer local APIs
csv.DictReader parsed both W&B-disabled hybrid locked manifests: 2 rows each, final and best, disable_wandb=true
dry-run submit_array.sh produced H200/embers arrays for eval and rollout without --require-formal-metadata
locked_mjlab_python.py user-site isolation check: user_site_paths []
locked_mjlab_python.py stack check: torch 2.3.1/cu118, tensordict 0.4.0, torchrl 0.4.0
```

Submitted replacement job IDs after commit
`28401ce79addfdd3881c68acd22e6ae5aba4d244`:

```text
9398352 pwm_hopper_locked_wmprobe_h100_fix4
9398353 original_pwm_adapter_hybrid_locked_nowandb_eval4_final_best_20260602
9398354 original_pwm_adapter_hybrid_locked_nowandb_rollout1_final_best_20260602
```

Completed MJLab W&B-disabled hybrid locked results:

```text
9398353_0 eval final COMPLETED 0:0: return_mean -0.671470582485199, episode_length_mean 47.75, fall_rate_mean 1.0, baseline_gate_pass false
9398353_1 eval best COMPLETED 0:0: return_mean -0.6604552865028381, episode_length_mean 48.0, fall_rate_mean 1.0, baseline_gate_pass false
9398354_0 rollout final COMPLETED 0:0: return_mean -0.10419569909572601, episode_length_mean 48.0, fall_rate_mean 1.0, num_frames 48, baseline_gate_pass false
9398354_1 rollout best COMPLETED 0:0: return_mean -0.11118776351213455, episode_length_mean 48.0, fall_rate_mean 1.0, num_frames 48, baseline_gate_pass false
interpretation: disabling W&B fixed the previous runtime packaging failure, but the faithful original-PWM MJLab adapter still collapses immediately; W&B was not the cause of poor MJLab policy behavior.
```

## Phase E Official Image Track Candidates

The user explicitly requested running official NEWT and LeWM environments in
parallel. Candidate details are recorded in:

```text
docs/git/image_official_newt_lewm_candidates_20260602.md
```

Summary before submission:

| Candidate | Type | Purpose | W&B mode | GPU / QOS | Dependency required? | Submit decision |
| --- | --- | --- | --- | --- | --- | --- |
| `newt_official_env_setup_20260602` | setup | Official NEWT conda env from `docker/environment.yaml` plus `ale_py==0.10`. | Off. | CPU. | No. | Submit. |
| `lewm_official_env_setup_20260602` | setup | Official LeWM uv venv with `stable-worldmodel[train,env]`. | Off. | CPU. | No. | Submit. |
| `newt_official_import_config_smoke_20260602` | smoke | Official NEWT import/config smoke. | Off. | CPU. | After NEWT env. | Submit. |
| `newt_official_walker_smoke_a100_20260602` | smoke / exploratory | Smallest official NEWT DMControl `walker-walk` train smoke. | Off. | A100 / `embers`. | After NEWT env. | Submit. |
| `lewm_official_import_config_smoke_20260602` | smoke | Official LeWM import/config/SIGReg-class smoke without dataset. | Off. | CPU. | After LeWM env. | Submit. |
| `lewm_official_data_checkpoint_download_20260602` | setup / diagnostic | Identify and download the smallest official LeWM data/checkpoint for train/eval. | Off. | CPU. | After LeWM env preferred. | Prepare next. |

Submitted Phase E official image job IDs after commit
`a428513610a636cde59f9cb8ce5621b204115760`:

```text
9398480 newt_official_env_setup_20260602
9398481 lewm_official_env_setup_20260602
9398482 newt_official_import_config_smoke_20260602, dependency=afterok:9398480
9398483 newt_official_walker_smoke_a100_20260602, dependency=afterok:9398480
9398484 lewm_official_import_config_smoke_20260602, dependency=afterok:9398481
```

Follow-up before replacement:

```text
9398480, 9398481, 9398482, and 9398484 were submitted on CPU without an explicit QOS and inherited inferno.
9398483 used embers but depended on the NEWT setup job.
All five jobs were canceled with scancel before setup completed.
Wrapper fix: all CPU jobs now pass CPU_QOS=embers explicitly; incomplete env directories are removed before recreation.
Replacement batch should use the same official env/setup smoke candidates with CPU_QOS=embers and GPU_QOS=embers.
```

Replacement submitted after QOS repair commit
`9e2e0a3c9942844823b20b99db1862b3f7c7d564`:

```text
9398555 newt_official_env_setup_20260602, QOS embers, FAILED 1:0
9398556 lewm_official_env_setup_20260602, QOS embers, RUNNING at first check
9398557 newt_official_import_config_smoke_20260602, dependency=afterok:9398555, canceled after setup failure
9398558 newt_official_walker_smoke_a100_20260602, dependency=afterok:9398555, canceled after setup failure
9398559 lewm_official_import_config_smoke_20260602, dependency=afterok:9398556, pending at first check
root_cause_newt: the canceled first submission left a partial conda env with `bin/python` present but no working standard-library `encodings` module.
fix_newt: before reusing an env, run `python -c 'import encodings'`; remove and recreate the env if the sanity check fails.
next_newt: resubmit only the NEWT setup/import/walker smoke with CPU_QOS=embers and GPU_QOS=embers after committing the sanity repair.
```

NEWT repair resubmission after commit
`0c75280e05d529bff0d3dd380a2da8c36fdd6e4c`:

```text
9398617 newt_official_env_setup_repair_20260602, QOS embers, FAILED 1:0
9398618 newt_official_import_config_repair_20260602, dependency=afterok:9398617, canceled after setup failure
9398619 newt_official_walker_repair_a100_20260602, dependency=afterok:9398617, canceled after setup failure
root_cause_newt_repair: official `conda env create -f docker/environment.yaml` reached pip dependency installation but `box2d-py` failed to build because the `swig` executable was not on PATH.
fix_newt_repair: keep the official YAML unchanged, but load the cluster `swig/4.1.1` module before `conda env create`; export `PYTHONNOUSERSITE=1`; add a setup completion marker so failed partial envs are removed before reuse.
next_newt_repair: resubmit only the NEWT setup/import/walker smoke with CPU_QOS=embers and GPU_QOS=embers after committing the SWIG/module and marker fix.
```

LeWM replacement result:

```text
9398556 lewm_official_env_setup_20260602 COMPLETED 0:0, QOS embers
9398559 lewm_official_import_config_smoke_20260602 COMPLETED 0:0, QOS embers
evidence: official LeWM uv env imports torch 2.12.0+cu130, hydra 1.3.2, stable_worldmodel, stable_pretraining; config smoke composes data=pusht_expert_train.lance, trainer.max_epochs=1, loss.sigreg.weight=0.09, and imports JEPA/ARPredictor/Embedder/MLP/SIGReg.
next_lewm: prepare official data/checkpoint inventory/download before train/eval because local LeWM `.h5`/`.lance` inputs were absent.
```

Next LeWM official assets candidate:

```text
lewm_official_pusht_assets_20260602
purpose: use the healthy official LeWM uv env to inventory `quentinll/lewm-pusht`, download `config.json`, `weights.pt`, and `pusht_expert_train.h5.zst`, decompress the dataset, convert the HF checkpoint into `$STABLEWM_HOME/pusht/lewm_object.ckpt`, and smoke-load `AutoCostModel('pusht/lewm')`.
validation: HF model repo lists `config.json` and `weights.pt`; HF dataset repo lists `pusht_expert_train.h5.zst`; official README describes the same HF mirror and conversion path.
resources: CPU / `embers`, W&B disabled, no train/eval yet.
script: `scripts/experiments/image_official/submit_lewm_official_pusht_assets_20260602.sh`
submit_decision: submit after committing the script and candidate record.
```

Submitted after commit
`dd41bea5f3a34537043dbe5170d02b2e4107ada2`:

```text
9398711 newt_official_env_setup_swig_20260602, QOS embers, RUNNING at first check
9398712 lewm_official_pusht_assets_20260602, QOS embers, RUNNING at first check
9398713 newt_official_import_config_swig_20260602, dependency=afterok:9398711, QOS embers, PENDING at first check
9398714 newt_official_walker_swig_a100_20260602, dependency=afterok:9398711, QOS embers, PENDING at first check
```

Hopper WM probe failure and replacement candidate:

```text
9398352 pwm_hopper_locked_wmprobe_h100_fix4 FAILED 1:0 after 00:00:54, QOS embers.
failure: DFlex kernels built, but `pwm_dflex_checkpoint_probe.py` resolved the relative checkpoint path after entering `baselines/PWM/scripts`, producing a duplicated path `.../baselines/PWM/scripts/baselines/PWM/scripts/outputs/.../final_policy.pt`.
replacement_candidate: `pwm_hopper_locked_wmprobe_h100_fix4` resubmission after script repair; pass absolute final/best checkpoint paths from `${ROOT}/baselines/PWM/scripts/outputs/...`.
resources: H100 / `embers`; W&B disabled; no dependency because checkpoints already exist.
expected_artifacts: `eval_results/pwm_phase2_hopper_locked_probe_20260602/final_actor_wm_vs_real_fix4.json` and `best_actor_wm_vs_real_fix4.json`.
submit_decision: submit after committing the absolute-path fix.
```

LeWM assets failure and replacement candidate:

```text
9398712 lewm_official_pusht_assets_20260602 FAILED 1:0 after 00:04:09, QOS embers.
completed_before_failure: HF model and dataset inventory succeeded; `config.json`, `weights.pt`, `pusht_expert_train.h5.zst`, and decompressed `pusht_expert_train.h5` now exist under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm`.
failure: checkpoint conversion triggered `stable_pretraining.backbone`, which imported old `datasets` code incompatible with the installed `pyarrow`/datasets API.
replacement_candidate: `lewm_official_pusht_assets_fix1_20260602`; load the official installed `stable_pretraining/backbone/utils.py` file directly for `vit_hf` and reuse downloaded files.
resources: CPU / `embers`; W&B disabled; no dependency because env/assets already exist.
expected_artifacts: `$STABLEWM_HOME/pusht/lewm_object.ckpt` and `AutoCostModel('pusht/lewm')` smoke-load evidence.
submit_decision: submit after committing the conversion import fix.
```

NEWT SWIG setup result and replacement candidate:

```text
9398711 newt_official_env_setup_swig_20260602 COMPLETED 0:0 after 00:08:03, QOS embers.
env evidence: official conda env completed after loading cluster `swig/4.1.1`; `box2d-py` built successfully; `ale_py==0.10.0` installed; torch 2.8.0+cu128, torchvision 0.23.0+cu128, hydra 1.3.2, gymnasium 0.29.1; marker `.newt_official_setup_ok_20260602` exists.
9398713 newt_official_import_config_swig_20260602 FAILED 1:0 after 00:00:04, QOS embers.
9398714 newt_official_walker_swig_a100_20260602 canceled before start.
failure: import/config smoke called `parse_cfg` outside Hydra runtime, causing `ValueError: get_original_cwd() must only be used after HydraConfig is initialized`.
replacement_candidate: `newt_official_import_config_swig_fix1_20260602` and `newt_official_walker_swig_fix1_a100_20260602`; import smoke should avoid `parse_cfg`, walker smoke should continue to use official `train.py`.
resources: CPU/A100 / `embers`; W&B disabled; no dependency because env setup succeeded.
script: `scripts/experiments/image_official/submit_newt_official_swig_followups_20260602.sh`.
submit_decision: submit after committing the wrapper repair and record.
```

Replacement submissions after commit
`c7c33b54d256d07355a88f95bc7c4cf506fd85b8`:

```text
9399731 lewm_official_pusht_assets_fix1_20260602, CPU / embers, PENDING Priority at first check
9399797 newt_official_import_config_swig_fix1_20260602, CPU / embers, PENDING Priority at first check
9399798 pwm_hopper_locked_wmprobe_h100_fix4, H100 / embers, PENDING Priority at first check
9399799 newt_official_walker_swig_fix1_a100_20260602, A100 / embers, PENDING Priority at first check
```

NEWT import/config fix1 result:

```text
9399797 newt_official_import_config_swig_fix1_20260602 COMPLETED 0:0 after 00:00:09, QOS embers.
evidence: `newt_import_config_ok`; task_count 234; model_sizes ['B', 'L', 'S', 'XL']; walker_action_dim 6; Config(task='walker-walk', obs='state', model_size='B', num_envs=10); walker_in_task_set True.
next_newt: wait for A100 walker smoke `9399799`.
```

LeWM assets fix1 failure and fix2 candidate:

```text
9399731 lewm_official_pusht_assets_fix1_20260602 FAILED 1:0 after 00:00:43, QOS embers.
progress: direct official `vit_hf` load worked and created ViT-tiny with patch_size 14 from `config.json`; downloads and decompressed dataset were reused.
failure: model construction failed at `ARPredictor(**cfg['predictor'])` because the Hugging Face `config.json` stores Hydra `_target_` keys that the plain class constructors do not accept.
replacement_candidate: `lewm_official_pusht_assets_fix2_20260602`; strip Hydra-only `_target_` and `_partial_` keys before passing config dictionaries to plain constructors; reuse already-downloaded assets.
resources: CPU / `embers`; W&B disabled; no dependency.
submit_decision: submit after committing the conversion helper fix.
```

Broad embers GPU candidate batch requested 2026-06-02:

```text
request: submit GPU tasks more aggressively; use embers broadly because it is not charged.
manifest: scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_20260602.csv
candidate: `mjqs_flow_mbpo_awr_H200`, 14 array elements, H200 / embers, array 0-13%8.
purpose: broaden the next Flow-MBPO exploration after MJLab adapter collapse by probing endpoint H=1/3/5, trajectory-chunk support-risk/fall-penalized, and residual-flow synthetic replays with conservative critic, action/support penalties, and real MJLab eval every 10/10 update steps.
inputs: all rows passed path validation for dataset, metadata, normalization, BC policy checkpoint, synthetic replay, and support-risk feature path where used.
W&B: disabled; these are diagnostic smokes rather than formal claim rows.
validation: `run_flow_mbpo_awr_row.py --check-inputs --dry-run` succeeded for row 0; `submit_array.sh --dry-run` produced `--array=0-13%8 --partition=gpu-h200 --qos=embers --gres=gpu:h200:1`.
submit_decision: submit after committing manifest and candidate record.
```

Broad embers submissions after commit `3a3e161`:

```text
9400408 lewm_official_pusht_assets_fix2_20260602, CPU / embers, PENDING Priority at first check.
9400410_[0-13%8] mjqs_flow_mbpo_awr_H200, H200 / embers, PENDING Resources at first check.
array_payload: 14 Flow-MBPO AWR diagnostics from scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_20260602.csv, max concurrent 8.
scheduler_validation: `squeue` and `sacct` both showed QOS `embers` for 9400408 and 9400410.
```

LeWM fix2 completion:

```text
9400408 lewm_official_pusht_assets_fix2_20260602 COMPLETED 0:0 after 00:00:41, QOS embers.
evidence: HF inventory succeeded; `config.json`, `weights.pt`, and `pusht_expert_train.h5.zst` were reused/downloaded; `dataset_h5` size was 46300921856 bytes; converted checkpoint `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/pusht/lewm_object.ckpt` size was 72334197 bytes; `autocost_load_ok JEPA`; `lewm_official_pusht_assets_ok`.
```

Additional shard GPU candidates after the user asked to submit as much embers GPU work as possible:

```text
reason: the first broad H200 array covers all 14 rows but only uses the H200 partition; additional shard manifests use independent output roots under `flow_mbpo_broad_embers_awr_shards_20260602/{gpu}/` to avoid overwriting the full-array outputs.
manifests:
  scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_h200_20260602.csv, 4 rows, H200 / embers, array 0-3%4.
  scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_h100_20260602.csv, 4 rows, H100 / embers, array 0-3%4.
  scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_a100_20260602.csv, 3 rows, A100 / embers, array 0-2%3.
  scripts/experiments/mjlab_qs/manifests/flow_mbpo_broad_embers_awr_l40s_20260602.csv, 3 rows, L40S / embers, array 0-2%3.
validation: all four shard manifests passed input-path validation; H100 dry-run produced `--array=0-3%4 --partition=gpu-h100 --qos=embers --gres=gpu:h100:1`.
submit_decision: submit all four shard arrays after committing the manifests and this record.
```

Shard submission result after commit `a2a7aac`:

```text
9400436_[0-3%4] mjqs_flow_mbpo_awr_H200, H200 / embers, PENDING Resources at first check.
9400435_[0-3%4] mjqs_flow_mbpo_awr_H100, H100 / embers, PENDING Priority at first check.
first_a100_array_attempt: failed with `QOSMaxSubmitJobPerUserLimit`.
first_l40s_array_attempt: failed with `Maximum CPU:GPU ratio of 4:1 for gpu-l40s,gpu-l40s node class`; the retry reduced CPUs from 8 to 4.
9400442 mjqs_flow_mbpo_awr_A100_single, A100 / embers, PENDING Priority at first check.
l40s_single_retry: failed with `QOSMaxSubmitJobPerUserLimit`; no further L40S/A100 array submission possible until queued embers jobs start or finish.
active_broad_gpu_payload_after_retry: 9400410 has 14 H200 AWR rows, 9400436 has 4 H200 shard rows, 9400435 has 4 H100 shard rows, 9400442 has 1 A100 shard row.
scheduler_validation: `squeue` and `sacct` showed QOS `embers` for 9400435, 9400436, and 9400442.
```
