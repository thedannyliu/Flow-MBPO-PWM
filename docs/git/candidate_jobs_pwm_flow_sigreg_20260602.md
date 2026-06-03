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
  2026-06-02 audit: Python 3.10.14, torch 2.3.1/cu118, Hydra 1.2.0,
  OmegaConf 2.2.3, W&B 0.12.21; CUDA unavailable on login node.
  called directly as ${ENV_DIR}/bin/python after exporting locked CUDA/compiler
  variables and copying env site-packages/dflex into a job-local sandbox.
  This is the preferred environment for credible original PWM reproduction
  claims. Direct login-node DFlex import fails because the environment is
  read-only and DFlex tries to rebuild kernels in site-packages; use the Slurm
  sandbox wrappers for DFlex jobs.

project/current Flow-MBPO and MJLab environment:
  conda environment pwm
  2026-06-02 audit: Python 3.10.19, torch 2.10.0+cu128, Hydra 1.3.2,
  OmegaConf 2.3.0, W&B 0.23.0; CUDA unavailable on login node.
  used for current Flow-MBPO diagnostics, MJLab QS runners, policy eval/render
  tools, manifest utilities, and new-code smokes.

MJLab adapter eval/video bridge:
  called through scripts/experiments/mjlab_qs/submit_array.sh after exporting
  PYTHONPATH=${PROJECT_ROOT}/src:${PROJECT_ROOT}/baselines/PWM/src:$PYTHONPATH.
  The upstream PWM path is required because the faithful adapter checkpoints
  pickle/import objects from the upstream `pwm` package.

Hybrid locked MJLab bridge:
  scripts/experiments/mjlab_qs/locked_mjlab_python.py
  loads locked torch/tensordict/torchrl/PWM first, then exposes project-env
  MJLab packages for W&B-disabled faithful-adapter checks.
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

Continuation poll after commit `12fd08e`:

```text
branch: mjlab-qs-rollout-policy-improvement
head_sha: 12fd08e
git_status: only pre-existing uncommitted `docs/goals/pwm_flow_sigreg_image_research_plan_20260602.md` remained dirty; left unstaged.
seff: unavailable on PATH (`seff: command not found`), so status used `squeue` and `sacct`.
known_gate_jobs: 9387895_0 COMPLETED 0:0; 9387896_0 COMPLETED 0:0; 9387942 FAILED 1:0 with locked DFlex `cc1plus` rebuild failure; 9387949 FAILED 1:0 with the same locked DFlex `cc1plus` rebuild failure.
active_related_queue:
  9399798 pwm_hopper_locked_wmprobe_h100_fix4 PENDING Priority, H100 / embers
  9399799 newt_official_walker_swig_fix1_a100_20260602 PENDING Priority, A100 / embers
  9400409_[0-15%8] newt_official_broad_smoke_a100_20260602 PENDING Priority, A100 / embers
  9400410_[0-13%8] mjqs_flow_mbpo_awr_H200 PENDING Resources, H200 / embers
  9400411_[0-5%6] lewm_official_pusht_eval_h100_20260602 PENDING Priority, H100 / embers
  9400412_[0-1%2] lewm_official_pusht_train_smoke_h100_20260602 PENDING Priority, H100 / embers
  9400435_[0-3%4] mjqs_flow_mbpo_awr_H100 PENDING Priority, H100 / embers
  9400436_[0-3%4] mjqs_flow_mbpo_awr_H200 PENDING Resources, H200 / embers
  9400442 mjqs_flow_mbpo_awr_A100_single PENDING Priority, A100 / embers
no_new_submission_reason: the previous L40S retry and A100 array attempt hit `QOSMaxSubmitJobPerUserLimit`; all useful validated broad rows are already queued or blocked by submit quota.
next_action: wait for any pending GPU job to start/finish, then inspect logs, cancel/fix/resubmit bad rows, and reattempt L40S/A100 shard coverage only after embers submit slots free.
```

Embers quota reopened and shard backfill:

```text
poll_time: 2026-06-02 18:31-18:34 America/New_York.
submit_probe: `sbatch --test-only` on gpu-l40s / embers / 1 GPU / 4 CPUs succeeded after earlier `QOSMaxSubmitJobPerUserLimit`.
9400410 H200 full AWR array: all 14 elements completed 0:0; early real-eval diagnostics were valid runs, not Slurm crashes, but all sampled rows underperformed the BC baseline with fall_rate_mean 1.0. Example row 0 best observed in log was return_mean 23.0109, length 324.625, fall 1.0 versus baseline return 45.8491, length 594.97, fall 0.625.
9400436 H200 shard AWR array: elements 1-3 completed 0:0 and element 0 was still running at first poll; sampled returns were also below the BC baseline with fall_rate_mean 1.0.
9400525 mjqs_flow_mbpo_awr_L40S, L40S / embers, array 0-2%3, submitted after lowering CPUs to 4 for the L40S CPU:GPU policy.
9400528 mjqs_flow_mbpo_awr_A100_remaining, A100 / embers, array 1-2%2, submitted manually against the A100 shard manifest to avoid duplicating pending single-row job 9400442 for row 0.
```

Additional official image GPU backup candidates:

```text
reason: NEWT A100 and LeWM H100 official jobs are still pending under Priority; embers submit quota reopened, so backup submissions on H200/L40S can produce signal sooner without touching official environments or code.
NEWT H200 backup: official NEWT env `/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602`, official repo `/storage/project/r-agarg35-0/eliu354/external_repos/newt`, same 16 task/seed array as 9400409, unique `exp_name=official_broad_h200_*`, W&B disabled.
NEWT L40S backup: same payload with L40S-compatible 4 CPUs and unique `exp_name=official_broad_l40s_*`, W&B disabled.
LeWM H200 eval backup: official LeWM env `/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602`, official repo `/storage/project/r-agarg35-0/eliu354/external_repos/le-wm`, converted PushT assets under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm`, same six eval rows as 9400411, unique output filenames with `_h200`.
LeWM L40S eval backup: same eval payload with L40S-compatible 4 CPUs and unique output filenames with `_l40s`.
submit_decision: submit these backup arrays while embers quota accepts them; stop once Slurm returns `QOSMaxSubmitJobPerUserLimit`.
```

Official image backup submission result after commit `d473642`:

```text
9400532_[0-5%6] lewm_official_pusht_eval_h200_20260602, H200 / embers, PENDING Resources at first check.
newt_h200_full_array_attempt: failed with `QOSMaxSubmitJobPerUserLimit`, so NEWT backup was split into smaller chunks.
9400537_[0-3%4] newt_official_broad_h200_chunk0_20260602, H200 / embers, PENDING Resources at first check.
newt_h200_chunk1_array_attempt: 4-row array 4-7%4 failed with `QOSMaxSubmitJobPerUserLimit`, so remaining NEWT backup rows were attempted as single-row arrays.
9400543_[4] newt_official_h200_row4_20260602, H200 / embers, PENDING Resources at first check.
9400544_[5] newt_official_h200_row5_20260602, H200 / embers, PENDING Resources at first check.
9400545_[6] newt_official_h200_row6_20260602, H200 / embers, PENDING Resources at first check.
newt_h200_row7_attempt: failed with `QOSMaxSubmitJobPerUserLimit`.
status_after_backfill: Slurm queue showed all newly accepted jobs with QOS `embers`; remaining unsent backup candidates are NEWT H200 rows 7-15 and the L40S NEWT/LeWM backup arrays.
```

H200 Flow-MBPO AWR aggregate result:

```text
9400410 H200 full AWR array: all 14 elements completed 0:0.
9400436 H200 shard AWR array: all 4 elements completed 0:0.
aggregate_interpretation: these were valid runs with real MJLab eval output, not Slurm/runtime failures, but none beat the BC baseline.
baseline: return 45.8491, length 594.97, fall 0.625.
best_selection_score_row: 9400410_8 at iter 10, return_mean 25.969, length 360.0, fall 1.0, selection_score -70.430.
best_return_row: same as best selection score, 9400410_8.
common_failure_mode: fall_rate_mean remained 1.0 across all sampled best and final evals, so the conservative AWR sweep did not fix MJLab collapse.
next_action: do not expand this exact AWR setting blindly; wait for A100/L40S/H100 shard confirmation and image official jobs, then pivot Flow-MBPO toward stronger pessimism/OOD gating or shorter-horizon model rollouts if all shards show the same fall pattern.
```

Post-inventory submit probe:

```text
preflight_inventory_commit: a81bc59.
probe: `sbatch --test-only` on gpu-l40s / embers / 1 GPU / 4 CPUs after the continuation inventory commit.
result: failed with `QOSMaxSubmitJobPerUserLimit`.
current_related_queue_count: 15 user jobs from `squeue -u $USER -h | wc -l`.
new_submission_decision: do not submit more jobs at this poll; remaining useful candidates are still NEWT H200 rows 7-15 and the L40S NEWT/LeWM backup arrays, but they must wait for submit slots to free.
```

NEWT/LeWM L40S backup readiness:

```text
script_added: scripts/experiments/image_official/submit_newt_lewm_l40s_backups_20260602.sh
payload: NEWT 16-row official broad train smoke on L40S; LeWM 6-row PushT eval on L40S; LeWM 2-row PushT train smoke on L40S.
validation: `bash -n` passed; input check passed for NEWT official setup marker, LeWM env python, converted PushT object checkpoint, and 46GB PushT h5 dataset.
submission_status: not submitted because `sbatch --test-only` on both gpu-l40s and gpu-h200 still failed with `QOSMaxSubmitJobPerUserLimit`.
next_action: as soon as any embers submit slot frees, submit this script or the remaining H200 NEWT rows, then record accepted job IDs in `docs/git/image_official_newt_lewm_candidates_20260602.md`.
```

LeWM H200 eval failure, repair, and replacement candidates:

```text
failed_jobs: 9400532_0..5 `lewm_official_pusht_eval_h200_20260602`, H200 / embers, FAILED 1:0 between 2026-06-02 18:49:49 and 18:50:23.
root_cause: the official LeWM eval entrypoint imported `stable_pretraining.data.datasets`, which imported the old HuggingFace `datasets` API against `pyarrow 24.0.0`; all rows failed before evaluation with `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'`.
canceled_jobs: 9400411 `lewm_official_pusht_eval_h100_20260602` and 9400412 `lewm_official_pusht_train_smoke_h100_20260602` were canceled before start to avoid repeating the same known environment failure.
env_repair: bootstrapped pip in `/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602`, restored `pyarrow==24.0.0` after a failed `pyarrow==12.0.1` ABI attempt against NumPy 2.2.6, added a lightweight `PyExtensionType` compatibility shim, and upgraded `datasets` to 3.6.0.
local_validation: `pyarrow 24.0.0 has_PyExtensionType True`; `datasets 3.6.0 has_config True`; `stable_pretraining` imports; `stable_worldmodel.policy.AutoCostModel` imports; `eval.py --config-name=pusht --cfg job` composes the intended PushT eval config with the converted checkpoint cache.
tracked_script_fix: `scripts/experiments/image_official/compat/sitecustomize.py` plus `PYTHONPATH=${COMPAT_ROOT}:${LEWM_ROOT}` in LeWM GPU scripts makes the pyarrow shim visible under batch jobs.
replacement_candidate: `scripts/experiments/image_official/submit_newt_h200_remaining_lewm_h200_fix_20260602.sh` submits LeWM H200 eval fix rows 0-5 and LeWM H200 train-smoke fix rows 0-1 with unique log/result names.
additional_candidate: the same script submits remaining NEWT H200 rows 7-15 that were previously blocked by the embers submit quota.
validation: `bash -n` passed for the broad GPU wrapper, L40S backup wrapper, and H200 replacement wrapper.
submit_decision: submit the H200 replacement wrapper after this repair/candidate record is committed, subject to the embers submit quota.
```

L40S Flow-MBPO AWR completion and L40S official-image submission:

```text
9400525_0 COMPLETED 0:0, L40S / embers, final real-eval return_mean 11.3848, episode_length_mean 194.0, fall_rate_mean 1.0.
9400525_1 COMPLETED 0:0, L40S / embers, final real-eval return_mean 10.2740, episode_length_mean about 172.1, fall_rate_mean 1.0.
9400525_2 COMPLETED 0:0, L40S / embers, final real-eval return_mean 12.9974, episode_length_mean 208.25, fall_rate_mean 1.0.
interpretation: these are valid negative MJLab runs; the L40S shard agrees with the H200 conservative AWR result that this setting still collapses/falls and does not beat the BC baseline return 45.8491 / length 594.97 / fall 0.625.
new_official_image_jobs: 9400714 `newt_official_broad_l40s_20260602` array 0-15%4, 9400715 `lewm_official_pusht_eval_l40s_20260602` array 0-5%3, and 9400716 `lewm_official_pusht_train_l40s_20260602` array 0-1%2 were accepted on gpu-l40s / embers after quota reopened.
early_newt_l40s_evidence: 9400714_0 completed walker-walk seed0 in 00:01:01 with eval R 42.247 and train R 51.202; 9400714_1 completed walker-run seed0 in 00:00:32 with eval R 42.179 and train R 23.809.
next_action: continue monitoring 9400714/9400715/9400716; record LeWM L40S results or failures, and submit H200 replacement jobs if the embers quota allows.
```

H200 single-row repair submissions:

```text
pre_submit_head: ce439b0.
array_attempt: `scripts/experiments/image_official/submit_newt_h200_remaining_lewm_h200_fix_20260602.sh` failed immediately with `QOSMaxSubmitJobPerUserLimit`.
single_row_probe: H200 and L40S `sbatch --test-only` single-row probes succeeded, so the large H200 array was split into single-row submissions.
accepted_lewm_h200_eval_fix: 9400771_[0], 9400772_[1], 9400773_[2], 9400774_[3], 9400775_[4], 9400776_[5], all gpu-h200 / embers, pending Resources at first check.
accepted_newt_h200_remaining: 9400778_[7] `newt_official_h200_row7_fix_20260602`, gpu-h200 / embers, pending Resources at first check.
blocked_newt_h200_remaining: row 8 submission failed with `QOSMaxSubmitJobPerUserLimit`; rows 8-15 remain unsent on H200.
lewm_train_h200_status: not submitted in this pass because quota was exhausted after the LeWM eval fix rows and one NEWT row.
newt_l40s_progress: 9400714_0..6 completed 0:0 with valid official NEWT train/eval smoke output; row 6 cartpole-swingup seed0 reached eval R 183.107 and train R 8.803.
next_action: monitor 9400771..9400776 immediately when H200 starts to verify the LeWM repair under Slurm; keep submitting rows 8-15 and LeWM H200 train singles as embers submit slots free.
```

NEWT/LeWM official-image continuation on 2026-06-02:

```text
newt_status: H200 official NEWT rows `9400537_0-3`, `9400543_4`, `9400544_5`, and `9400545_6` completed 0:0; the same first four rows completed again on L40S as `9400714_0-3`.
lewm_failure: H200 LeWM eval rows `9400532_0-5` failed 1:0 because `datasets` hit a legacy `pyarrow.PyExtensionType` import path; canceled H100 rows `9400411` and `9400412` produced no eval/train evidence.
fix: added `scripts/experiments/image_official/compat/sitecustomize.py` and patched LeWM submit payloads to run with `PYTHONPATH=${COMPAT_ROOT}:${LEWM_ROOT}`.
submitted: `9400714` NEWT L40S broad official array, `9400715` LeWM PushT eval L40S array, and `9400716` LeWM PushT train-smoke L40S array, all QOS embers.
prepared_not_submitted: `scripts/experiments/image_official/submit_newt_h200_remaining_lewm_h200_fix_20260602.sh` covers NEWT H200 rows 7-15 plus shimmed LeWM H200 eval/train replacements; `sbatch --test-only` is currently blocked by `QOSMaxSubmitJobPerUserLimit`.
priority_next: watch `9400715` logs for the pyarrow root-cause replacement; submit the H200 fix wrapper as soon as embers submit quota reopens.
```

Continuation candidate refresh after preflight commit `ce439b0`:

```text
poll_time: 2026-06-02 19:10-19:13 America/New_York.
hopper_probe_result: `9399798` completed 0:0 on H100 / embers and wrote `eval_results/pwm_phase2_hopper_locked_probe_20260602/final_actor_wm_vs_real_fix4.json` plus `best_actor_wm_vs_real_fix4.json`. Final actor normalized WM-real reward correlation was 0.999928 with MAE 0.004424; best actor correlation was 0.999961 with MAE 0.003968; both had termination/truncation fraction 0.0.
newt_l40s_progress: `9400714_0-6` completed 0:0 with valid official NEWT train/eval output; row rewards match the already completed H200 seed0 rows for the overlapping tasks.
lewm_l40s_status: `9400715_[0-5]` and `9400716_[0-1]` remain pending Priority; no repaired LeWM GPU row has started yet.
h200_submit_probe: gpu-h200 / embers `sbatch --test-only` succeeded after the inventory check.
candidate_to_submit: `scripts/experiments/image_official/submit_newt_h200_remaining_lewm_h200_fix_20260602.sh`.
candidate_payload: NEWT H200 rows 7-15; repaired LeWM H200 eval rows 0-5; repaired LeWM H200 train-smoke rows 0-1.
dependency_required: no; NEWT env marker, LeWM env, converted PushT checkpoint, and PushT h5 dataset already exist.
wandb_mode: disabled for these official-image smoke/eval replacement rows.
submit_decision: submit after committing this inventory/candidate refresh if the embers submit quota still accepts it.
```

Additional H200 single-row submission after commit `a0a0751`:

```text
submit_probe: gpu-h200 / embers single-row `sbatch --test-only` still succeeded after `a0a0751`.
accepted_newt_h200_rows: 9400797_[8] walker-walk seed1, 9400798_[9] walker-run seed1, 9400799_[10] cheetah-run seed1, 9400800_[11] hopper-hop seed1; all pending Resources at first check.
blocked_newt_h200_rows: row 12 submission failed with `QOSMaxSubmitJobPerUserLimit`; rows 12-15 remain unsent on H200.
newt_l40s_progress: 9400714_7 cup-catch seed0 completed 0:0 with eval/train `0.0/0.0`; 9400714_8 walker-walk seed1 completed 0:0 with eval/train `18.338/37.122`; 9400714_9 walker-run seed1 completed 0:0 with eval/train `18.157/28.181`.
lewm_status: repaired LeWM H200 eval rows 9400771..9400776 are still pending Resources; LeWM L40S eval/train arrays 9400715/9400716 are still pending Priority.
next_action: continue single-row H200 submissions for NEWT rows 12-15 and LeWM H200 train rows when submit quota reopens; do not duplicate already accepted rows 7-11.
```

NEWT H200 remaining completion of submission set:

```text
accepted_after_903642d: 9400814_[12] reacher-easy seed1; 9400815_[13] pendulum-swingup seed1; 9400816_[14] cartpole-swingup seed1; 9400817_[15] cup-catch seed1.
status_at_first_check: all four H200 / embers rows were pending Resources.
newt_h200_remaining_status: rows 7-15 are now all accepted on H200 / embers; no further NEWT H200 remaining rows should be submitted.
lewm_eval_status: 9400771_0 started on H200 / embers at 2026-06-02 19:16; stdout initially contained the Slurm prolog only, with no immediate pyarrow crash.
remaining_h200_candidate: LeWM H200 train-smoke singles for seeds 0 and 1 are still useful if submit quota opens and no repaired LeWM eval row fails first.
```

LeWM H200 eval fix row0 failure and hdf5plugin repair:

```text
failed_job: 9400771_0 `lewm_official_pusht_eval_h200_fix_row0_20260602`, gpu-h200 / embers, FAILED 1:0 after 00:01:29.
progress_before_failure: pyarrow/datasets import progressed past the earlier `PyExtensionType` failure; this proves the first LeWM compatibility repair worked.
new_root_cause: official LeWM `eval.py` called `stable_worldmodel.data.HDF5Dataset`, but the installed `stable_worldmodel.data` did not export it because optional import of `stable_worldmodel.data.formats.hdf5` failed without `hdf5plugin`.
affected_jobs_canceled: 9400772_[1]..9400776_[5] repaired LeWM H200 eval rows, plus L40S LeWM eval/train arrays 9400715 and 9400716, were canceled to avoid repeating the same environment failure.
dataset_validation: direct HDF5 read showed `pixels` uses an unknown plugin compression filter; without `hdf5plugin`, reading `pixels[0]` fails with `can't open directory (/usr/local/lib/plugin)`.
repair: installed `hdf5plugin==6.0.0` into repo-local `scripts/experiments/image_official/compat/vendor/` because the official LeWM env site-packages is read-only; `.gitignore` excludes that binary vendor directory.
tracked_rebuild_script: `scripts/experiments/image_official/install_lewm_compat_vendor_20260602.sh`.
tracked_shim_update: `scripts/experiments/image_official/compat/sitecustomize.py` prepends the vendor directory to `sys.path` and re-exports `HDF5Dataset` / `HDF5Writer` onto `stable_worldmodel.data` when missing.
local_validation: with `PYTHONPATH=scripts/experiments/image_official/compat:${LEWM_ROOT}`, `hdf5plugin` imported from the vendor directory, `stable_worldmodel.data.HDF5Dataset` was visible, and `pixels[0]` from `pusht_expert_train.h5` read successfully with shape `(224, 224, 3)`.
next_action: commit this repair record, then resubmit LeWM eval/train replacement rows only after confirming submit quota; do not resubmit NEWT H200 rows because 7-15 are already queued.
```

Current candidate refresh after preflight commit `f923be5`:

```text
preflight_time: 2026-06-02 after the LeWM HDF5 compatibility repair.
branch: mjlab-qs-rollout-policy-improvement.
head_sha: f923be59b563498e3e9dc65ddcb2d06a6044201a.
slurm_status: `squeue` could not contact the Slurm controller; `sacct` could not contact SlurmDB; `seff` is unavailable on PATH.
submission_decision: no new sbatch submission during this refresh because live Slurm status and submit quota could not be verified.
```

| Candidate | Type | Purpose | Inputs exist? | W&B mode | Expected artifacts | GPU / QOS | Dependency required? | Submit decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lewm_official_pusht_eval_h200_hdf5fix_20260602` | diagnostic / official eval smoke | Re-run the six official LeWM PushT eval rows after commit `f923be5` added repo-local `hdf5plugin` vendor loading and `stable_worldmodel.data.HDF5Dataset` re-export. This replaces failed/canceled rows `9400771_0` and `9400772_1` through `9400776_5`. | Yes: official LeWM env exists; converted checkpoint `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/pusht/lewm_object.ckpt` exists; `pusht_expert_train.h5` exists; local validation can read `pixels[0]` with shape `(224, 224, 3)`. | Disabled. | Per-row official eval logs/results for `pusht/lewm` and random policies at horizons 2/5 with eval budget 30; unique result filenames should include `hdf5fix` to avoid overwriting failed fix rows. | H200 preferred, then L40S/H100 if H200 quota stays blocked; `embers`. | No. | Submit only after Slurm controller/accounting is reachable and submit quota can be checked. |
| `lewm_official_pusht_train_h200_hdf5fix_20260602` | diagnostic / official train smoke | Run the two official LeWM PushT train-smoke rows after the same HDF5 plugin repair. Prior L40S train array `9400716` was canceled before useful evidence because row `9400771_0` revealed the shared HDF5 plugin root cause. | Yes: same repaired official LeWM env/assets; training h5 pixel reads now work under the compat `PYTHONPATH`. | Disabled. | Official train logs for 1 epoch, 2 train batches, 1 val batch; no performance claim. | H200 preferred, then L40S/H100; `embers`. | No. | Submit only after Slurm is reachable and after or alongside the eval replacement, with unique output/log names. |
| `newt_official_h200_remaining_rows` | smoke / official infrastructure | No new submission needed: rows 7-15 are already accepted and logs show valid official NEWT output for `9400778`, `9400797`, `9400798`, `9400799`, `9400800`, `9400814`, `9400815`, `9400816`, and `9400817`. | Existing rows already ran or produced logs. | Disabled. | Existing logs under `logs/slurm/image_official/newt_official_h200_remaining_single_9400*.out`. | H200 / `embers`. | No. | Do not submit duplicates. |

LeWM HDF5-fix submission preparation after commit `cea9b37`:

```text
script: scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh
scope: LeWM only; it intentionally does not resubmit NEWT rows 7-15, which already completed.
validation:
  bash -n scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh
  direct LeWM env without compat still fails with ModuleNotFoundError: hdf5plugin, proving the compat path is required.
  PYTHONPATH=${ROOT}/scripts/experiments/image_official/compat:${LEWM_ROOT} ${LEWM_ENV}/bin/python can import hdf5plugin, exposes stable_worldmodel.data.HDF5Dataset, and reads pusht_expert_train.h5 pixels[0] with shape (224, 224, 3).
  sbatch --test-only on gpu-h200 / embers / 4 CPU / 64G succeeded, predicted start 2026-06-05T14:18:24.
submit_decision: submit the LeWM HDF5-fix H200 eval and train-smoke arrays after committing this script/candidate record.
```

Submitted LeWM HDF5-fix replacements after commit `8270506`:

```text
9401543_[0-5%3] lewm_official_pusht_eval_hdf5fix_h200_20260602, H200 / embers, PENDING Priority at first check.
9401544_[0-1%2] lewm_official_pusht_train_hdf5fix_h200_20260602, H200 / embers, PENDING Priority at first check.
script: scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh.
wandb: disabled.
dependencies: none; official env, checkpoint, dataset, and compat vendor are present.
expected_logs:
  logs/slurm/image_official/lewm_official_pusht_eval_hdf5fix_h200_%A_%a.{out,err}
  logs/slurm/image_official/lewm_official_pusht_train_hdf5fix_h200_%A_%a.{out,err}
next_action: monitor first started row; cancel siblings and record root cause if the HDF5 repair exposes another official LeWM issue.
```

Continuation poll after LeWM HDF5-fix submission:

```text
poll_time: 2026-06-02 after commit 30d7450.
queue:
  9401543_[0-5%3] LeWM HDF5-fix eval H200 / embers remains PENDING Priority; no logs yet.
  9401544_[0-1%2] LeWM HDF5-fix train H200 / embers remains PENDING Priority; no logs yet.
new_results:
  9400435_[0-3] H100 Flow-MBPO AWR shard completed 0:0. All rows were valid negative diagnostics with fall_rate_mean 1.0 and returns 19.6023, 19.9737, 21.3537, and 11.0026.
  9400714_[0-15] NEWT official broad L40S smoke completed 0:0. All rows produced official train/eval output; this is infrastructure evidence only.
submit_decision: no new sbatch submission in this poll. Useful LeWM replacements are already queued, A100 Flow/NEWT jobs are already queued, and the completed H100 AWR shard confirms the same negative pattern rather than motivating a duplicate run.
```

LeWM HDF5-fix start failure and dataset path repair:

```text
failure_time: 2026-06-02 after commit e8a52e6.
failed_row: 9401543_0 `lewm_official_pusht_eval_hdf5fix_h200_20260602`, FAILED 1:0 after 00:00:24.
canceled_rows: 9401543_1, 9401543_[2-5%3], and 9401544_[0-1%2] were canceled to avoid repeating the same official LeWM cache-layout issue.
root_cause: the HDF5 plugin repair worked, and `stable_worldmodel.data.HDF5Dataset` was available, but official eval looked for `${STABLEWM_HOME}/datasets/pusht_expert_train.h5` while the decompressed dataset lived at `${STABLEWM_HOME}/pusht_expert_train.h5`.
repair: `submit_lewm_hdf5fix_h200_20260602.sh` now creates `${STABLEWM_HOME}/datasets/pusht_expert_train.h5 -> ../pusht_expert_train.h5`, uses unique `hdf5pathfix` job/log/result names, and exports `LOCAL_DATASET_DIR=${STABLEWM_HOME}` for train because official `load_dataset` appends `datasets` itself.
validation:
  bash -n scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh passed.
  eval path: `HDF5Dataset('pusht_expert_train', cache_dir=${STABLEWM_HOME})` returned length 2336736 and state dim 7.
  train path: `load_dataset('pusht_expert_train.h5', cache_dir=${STABLEWM_HOME})` returned length 2336736 and state dim 7.
  sbatch --test-only with account gts-agarg35, gpu-h200, QOS embers, 8 CPU, 64G, 01:00:00 accepted the request and predicted a 2026-06-05T13:18:24 start.
submit_decision: commit the path repair and then resubmit the LeWM eval/train replacement arrays with the `hdf5pathfix` names.
```

Submitted LeWM HDF5 pathfix replacements after commit `8f7dbce`:

```text
9401638_[0-5%3] lewm_official_pusht_eval_hdf5pathfix_h200_20260602, H200 / embers, PENDING Priority at first check.
9401639_[0-1%2] lewm_official_pusht_train_hdf5pathfix_h200_20260602, H200 / embers, PENDING Priority at first check.
script: scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh.
wandb: disabled.
dependencies: none; official env, converted checkpoint, dataset, repo-local hdf5plugin vendor, and dataset compatibility symlink are present.
expected_logs:
  logs/slurm/image_official/lewm_official_pusht_eval_hdf5pathfix_h200_%A_%a.{out,err}
  logs/slurm/image_official/lewm_official_pusht_train_hdf5pathfix_h200_%A_%a.{out,err}
initial_scheduler_check:
  squeue showed both arrays PENDING Priority.
  sacct showed both arrays PENDING, QOS embers, exit 0:0.
next_action: monitor the first started eval/train row immediately; cancel siblings if a new shared official LeWM root cause appears.
```

Continuation poll after LeWM pathfix submission and AWR artifact extraction:

```text
poll_time: 2026-06-02 after commit be114a9.
repo_state: branch mjlab-qs-rollout-policy-improvement, worktree clean.
queue:
  9401638_[0-5%3] LeWM HDF5 pathfix eval remains PENDING Priority on gpu-h200 / embers.
  9401639_[0-1%2] LeWM HDF5 pathfix train remains PENDING Priority on gpu-h200 / embers.
  A100 jobs 9399799, 9400409, 9400442, and 9400528 remain PENDING Priority; unrelated 9400333 remains PENDING Priority.
artifact_extraction:
  Recomputed Flow-MBPO AWR metrics from summary.json files, not just logs.
  9400410 full H200 AWR array: 14/14 summaries present; best row by return and selection score is row 8 endpoint_h3_trunc_cql_mixed_s1, best-real iter 10, return 25.9699, length 360.000, fall 1.000, score -70.430.
  9400436 H200 shard: 4/4 summaries present; best row is endpoint_h1_trunc_cql_data_noise_s0, return 22.9451, length 333.750, fall 1.000.
  9400435 H100 shard: 4/4 summaries present; best row is residual_h5_trunc_cql_data_noise_s0, return 22.6907, length 330.625, fall 1.000.
  9400525 L40S shard: 3/3 summaries present; best row is traj_h3_fall5_trunc_cql_data_noise_s1, return 16.4492, length 256.375, fall 1.000.
interpretation: all conservative AWR rows remain below BC return 45.8491, length 594.97, fall 0.625; this confirms model-exploitation/fall failure rather than a policy-improvement signal.
submit_decision: no new sbatch submission from this poll. LeWM replacements and A100 NEWT/Flow rows are already queued, and completed AWR rows are usable negative diagnostics rather than formal candidates.
```

SIGReg CPU prerequisite recheck:

```text
time: 2026-06-02 after commit 405d8f3.
command: pytest -q tests/test_sigreg.py
result: 5 passed in 27.61s.
interpretation: the local SIGReg utility still satisfies the documented no-GPU prerequisite tests: finite loss, finite gradients, constant-latent anti-collapse penalty, zero-weight no-op, and latent Gaussian/isotropy diagnostics.
submit_decision: no SIGReg GPU row yet; the next SIGReg job still needs a selected controlled Flow-PWM row and the same real-eval/video evidence protocol as its no-SIGReg baseline.
```

R0-R4 controlled matrix preparation:

```text
time: 2026-06-02 after commit cb819b8.
record: docs/git/r0_r4_controlled_matrix_status_20260602.md.
result: R0 faithful original PWM is complete as a negative MJLab baseline; R1 and R2 are not satisfied by old 2x2 artifacts because they do not preserve the faithful R0 update/protocol; R3 has old diagnostic and broad AWR negative evidence but not a clean causal claim; R4 remains exploratory and must be selected from existing candidate evidence before any missing eval/video submission.
submit_decision: no new sbatch submission from this preparation step. The next useful GPU work is either a concrete R1/R2 runner with explicit inputs, an R4 missing eval/video package for a selected existing checkpoint, or a short-horizon pessimistic Flow-MBPO row list, not another duplicate conservative AWR sweep.
```

PWM comparator clarification and LeWM hfcachefix candidate:

```text
time: 2026-06-02 after commit dbc3829.
source_audit: compared `scripts/experiments/mjlab_qs/run_original_pwm_adapter.py` with `baselines/PWM/scripts/train_dflex.py` and `baselines/PWM/src/pwm/algorithms/pwm.py`.
clarification: the MJLab `original_pwm_adapter` rows import upstream PWM and use upstream model/update primitives, but they do not execute the full upstream Hydra `train_dflex.py` pipeline or `agent.train()` loop. They are adapter-level PWM algorithm evidence, not full upstream-pipeline evidence.
docs_updated:
  docs/goals/pwm_flow_sigreg_image_research_plan_20260602.md
  docs/git/flow_pwm_matched_evidence_inventory_20260602.md
  docs/git/r0_r4_controlled_matrix_status_20260602.md
```

| Candidate | Type | Purpose | Inputs exist? | W&B mode | Expected artifacts | GPU / QOS | Dependency required? | Submit decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lewm_official_pusht_eval_hfcachefix_h200_20260602` | smoke / exploratory eval | Replace failed/canceled LeWM pathfix eval rows `9401638_[0-5]` after `9401638_0` proved the dataset path is fixed but official `load_pretrained('pusht/lewm')` needs a local normalized `${STABLEWM_HOME}/checkpoints/models--pusht--lewm` cache. | Yes: official LeWM env, PushT HDF5 dataset, hdf5plugin compat shim, HF `config.json`/`weights.pt`, and normalized local pretrained cache were validated. | Disabled. | Logs `logs/slurm/image_official/lewm_official_pusht_eval_hfcachefix_h200_%A_%a.{out,err}` and official eval result files with `hfcachefix` suffixes. | H200 / `embers`. | No. | Submit after committing wrapper/docs. |
| `lewm_official_pusht_train_hfcachefix_h200_20260602` | smoke / exploratory train | Replace canceled train rows `9401639_[0-1]` after the shared pretrained-cache repair; run the same short official train smoke. | Yes: `swm.data.load_dataset('pusht_expert_train.h5', cache_dir=${STABLEWM_HOME})` returned length 2336736 and state dim 7. | Disabled. | Logs `logs/slurm/image_official/lewm_official_pusht_train_hfcachefix_h200_%A_%a.{out,err}` and official train smoke checkpoint/output dirs with `hfcachefix` suffixes. | H200 / `embers`. | No. | Submit after committing wrapper/docs. |

Submitted after commit `7ccc508`:

```text
9401796_[0-5%3] lewm_official_pusht_eval_hfcachefix_h200_20260602, H200 / embers, PENDING Priority at first check.
9401797_[0-1%2] lewm_official_pusht_train_hfcachefix_h200_20260602, H200 / embers, PENDING Priority at first check.
script: scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh.
wandb: disabled.
dependencies: none.
```

NEWT walker fix1 completion:

```text
9399799 newt_official_walker_swig_fix1_a100_20260602 COMPLETED 0:0 after 00:00:48 on gpu-a100 / embers.
evidence: official NEWT walker-walk smoke reached `Training completed successfully`; W&B disabled; rewards eval I=0 42.248, train I=500 51.202, train I=1000 45.629.
interpretation: official NEWT infrastructure smoke passed; not performance evidence.
submit_decision: do not submit another single walker smoke; wait for broad A100 array `9400409_[1-15]`.
```

LeWM hfcachefix first-row completion:

```text
9401796_0 lewm_official_pusht_eval_hfcachefix_h200_20260602 COMPLETED 0:0 after 00:01:34 on gpu-h200 / embers.
evidence: official eval loaded the HDF5 dataset and local normalized `pusht/lewm` cache successfully; reported `success_rate: 100.0` over 4 eval episodes.
interpretation: pretrained-cache fix is validated for the first eval row; no sibling cancellation needed.
```

Full upstream PWM pipeline on MJLab candidate:

```text
time: 2026-06-02 after user request to test a complete PWM pipeline on MJLab.
requirement: use the PWM-tested successful environment, not the normal project `pwm` env as the base runtime.
environment_decision:
  Direct `/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4/bin/python` imports torch 2.3.1 and upstream `pwm`, but does not include `mjlab` or `flow_mbpo_pwm`.
  Use `scripts/experiments/mjlab_qs/locked_mjlab_python.py` instead. It starts from the locked original PWM env, imports locked torch/tensordict/torchrl and `baselines/PWM/src/pwm.algorithms.pwm.PWM` first, then exposes project-env site-packages only so MJLab and the local MJLab env adapter are importable.
validation:
  `locked_mjlab_python.py -` reported torch 2.3.1 from `/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4`, `pwm` from `baselines/PWM/src`, `mjlab` from the project env site-packages, and `flow_mbpo_pwm` from this repo.
  Hydra config compose through `baselines/PWM/scripts/train_dflex.py --cfg job env=mjlab_velocity_flat_unitree_g1 alg=pwm` shows `alg._target_: pwm.algorithms.pwm.PWM`, not `flow_mbpo_pwm.algorithms.pwm.PWM`.
  The submit wrapper now writes the MJLab Hydra env config at job startup because `baselines/PWM` is a nested repo and ignored by the main repo.
  `bash -n scripts/experiments/mjlab_qs/submit_upstream_pwm_mjlab_full_pipeline_smoke_20260602.sh` passed after that wrapper-local config generation change.
  `sbatch --test-only` accepted H200 / embers, 8 CPU, 96G, 01:00:00.
```

| Candidate | Type | Purpose | Inputs exist? | W&B mode | Expected artifacts | GPU / QOS | Dependency required? | Submit decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `upstream_pwm_mjlab_full_smoke_h200_20260602` | smoke / diagnostic | Run the closest available full upstream PWM pipeline on MJLab: `baselines/PWM/scripts/train_dflex.py`, `alg=pwm`, `pwm.algorithms.pwm.PWM`, and `agent.train()` orchestration, with only the env config swapped to `flow_mbpo_pwm.envs.mjlab_pwm_adapter.create_mjlab_pwm_env`. This is distinct from the previous QS-window `original_pwm_adapter`. | Yes: locked PWM env, upstream PWM source, project MJLab packages via locked bridge, and wrapper-generated MJLab env config exist. | Disabled. | Slurm logs under `logs/slurm/mjlab_qs/upstream_pwm_full_pipeline/`; upstream PWM logdir `baselines/PWM/scripts/outputs/.../logs/upstream_pwm_mjlab_full_smoke_h200_seed0_20260602`; init/final/best policy files if the smoke reaches saving. | H200 / `embers`. | No. | Submit after committing wrapper and candidate/preflight docs. |

Submitted full upstream PWM MJLab smoke:

```text
9401871 upstream_pwm_mjlab_full_smoke_h200_20260602 submitted after commit f79cb1d.
initial scheduler state: PENDING, reason Priority, gpu-h200 / embers, 01:00:00.
runtime: locked original PWM env through `locked_mjlab_python.py`, upstream
`pwm.algorithms.pwm.PWM`, wrapper-generated MJLab Hydra env config, W&B disabled.
```

Full upstream PWM MJLab smoke result:

```text
9401871 COMPLETED 0:0 after 00:00:27 on gpu-h200 / embers.
log: logs/slurm/mjlab_qs/upstream_pwm_full_pipeline/upstream_pwm_mjlab_full_smoke_h200_9401871.out
Hydra output: baselines/PWM/scripts/outputs/2026-06-02/21-19-04
policy artifacts:
  logs/upstream_pwm_mjlab_full_smoke_h200_seed0_20260602/init_policy.pt
  logs/upstream_pwm_mjlab_full_smoke_h200_seed0_20260602/best_policy.pt
  logs/upstream_pwm_mjlab_full_smoke_h200_seed0_20260602/final_policy.pt
  logs/upstream_pwm_mjlab_full_smoke_h200_seed0_20260602/final_policy.buffer/
evidence:
  MJLab adapter initialized task `Mjlab-Velocity-Flat-Unitree-G1`, num_envs 16,
  obs_dim 210, act_dim 29.
  Hydra config target `alg._target_` was `pwm.algorithms.pwm.PWM`; env target
  was `flow_mbpo_pwm.envs.mjlab_pwm_adapter.create_mjlab_pwm_env`.
  The upstream `PWM.train()` loop ran actor/critic/world-model updates and
  wrote init/best/final policies. Printed update rows reached `[4/8]` because
  early epochs with an empty buffer only collect rollout data and do not print
  update metrics.
last smoke metrics:
  [4/8] R:0.10, T:0.0, H:8.0, S:1024, FPS:780,
  pi_loss:4.02, pi_grad:4.59/4.59, v_loss:0.08,
  wm_loss:0.01, rew_loss:0.00, dyn_loss:0.01.
  Eval summary printed mean episode loss 0.05, mean discounted loss 0.05,
  mean episode length 32.00 over eval_runs=2.
interpretation: a complete upstream PWM orchestration smoke now runs on MJLab
using the PWM-tested locked runtime bridge. This is a pipeline feasibility
success only; it is not yet a performance claim.
next_action: run a longer full-PWM MJLab formal/eval pass if this smoke should
be promoted beyond feasibility.
```

## Continuation Candidates After Full PWM Smoke

Preflight before this candidate batch is recorded in
`docs/git/preflight_inventory_pwm_flow_sigreg_20260602.md` after commit
`31fa5c0`.

| Candidate | Type | Purpose | Inputs exist? | W&B mode | Expected artifacts | GPU / QOS | Dependency required? | Submit decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lewm_official_pusht_train_hfcachefix_fix1_h200_20260602` | smoke / repair | Re-run only the official LeWM train smoke rows after `9401797_[0-1]` failed before training because Hydra struct mode rejected non-append `trainer.limit_train_batches` overrides. The replacement uses `+trainer.limit_train_batches=2` and `+trainer.limit_val_batches=1`. | Yes: official LeWM env exists; local HDF5 dataset exists; repo-local `hdf5plugin` compat path exists; cache/data path already validated by completed eval array `9401796`. | Disabled. | Slurm logs under `logs/slurm/image_official/lewm_official_pusht_train_hfcachefix_fix1_h200_%A_%a.{out,err}` and official train smoke output under LeWM output/cache paths if training starts. | H200 / `embers`. | No. | Submit after committing wrapper and candidate record. |
| `upstream_pwm_mjlab_full_longdiag_h200_20260602` | diagnostic | Run a longer complete upstream PWM pipeline on MJLab after smoke `9401871` proved feasibility. This keeps W&B disabled and remains diagnostic: `train_dflex.py`, upstream `pwm.algorithms.pwm.PWM`, wrapper-generated MJLab env config, 32 envs, episode length 64, horizon 16, max epochs 200. | Yes: locked PWM runtime bridge, upstream PWM source, MJLab adapter, and wrapper-generated env config are available; smoke `9401871` produced init/best/final policies without runtime failures. | Disabled. | Slurm logs under `logs/slurm/mjlab_qs/upstream_pwm_full_pipeline/upstream_pwm_mjlab_full_longdiag_h200_%j.{out,err}`; upstream PWM output under `baselines/PWM/scripts/outputs/.../logs/upstream_pwm_mjlab_full_longdiag_h200_seed0_20260602/`; init/best/final policies and intermediate saves if reached. | H200 / `embers`. | No. | Submit after committing wrapper and candidate record. |

Validation before submission:

```text
bash -n scripts/experiments/image_official/submit_lewm_train_hfcachefix_fix1_h200_20260602.sh
bash -n scripts/experiments/mjlab_qs/submit_upstream_pwm_mjlab_full_pipeline_longdiag_20260602.sh
LeWM Hydra compose check with append overrides showed:
  trainer.max_epochs: 1
  trainer.limit_train_batches: 2
  trainer.limit_val_batches: 1
  loader.batch_size: 8
  wandb.enabled: false
Full-PWM Hydra compose through locked bridge showed:
  env target `flow_mbpo_pwm.envs.mjlab_pwm_adapter.create_mjlab_pwm_env`
  alg target `pwm.algorithms.pwm.PWM`
  eval_runs: 4
  critic_iterations: 2
  max_epochs: 200
  horizon: 16
  wm_batch_size: 64
  wm_buffer_size: 50000
Scheduler validation:
  `sbatch --test-only` accepted both H200 / embers requests:
  LeWM train fix1, 4 CPUs, 96G, 02:00:00;
  full-PWM longdiag, 8 CPUs, 128G, 02:00:00.
```

Submitted continuation jobs after commit `daff684`:

```text
9401906 upstream_pwm_mjlab_full_longdiag_h200_20260602
  initial state: PENDING, reason Resources, gpu-h200 / embers.
  W&B disabled; locked original PWM env through `locked_mjlab_python.py`;
  upstream `pwm.algorithms.pwm.PWM`; MJLab adapter env config generated by wrapper.

9401907_[0-1%2] lewm_official_pusht_train_hfcachefix_fix1_h200_20260602
  initial state: PENDING, reason Resources, gpu-h200 / embers.
  W&B disabled; official LeWM env; Hydra append overrides for
  `+trainer.limit_train_batches=2` and `+trainer.limit_val_batches=1`.
```
