# Image Official NEWT And LeWM Candidates

Date: 2026-06-02

Purpose: start Phase E with official-environment setup and cheap official smokes
while the state-based MJLab jobs continue. These rows are feasibility and
diagnostic evidence only. They are not image-task performance claims.

## Preflight

```text
branch: mjlab-qs-rollout-policy-improvement
head_sha_before_image_edits: 28401ce79addfdd3881c68acd22e6ae5aba4d244
dirty_before_image_edits: docs/goals/pwm_flow_sigreg_image_research_plan_20260602.md
active_squeue_before_image_setup:
  9398352 pwm_hopper_locked_wmprobe_h100_fix4 PENDING Priority
  9398353_[0-1] mjqs_policy_eval_H200 PENDING Resources
  9398354_[0-1] mjqs_policy_rollout_H200 PENDING Resources
```

Phase E trigger from the active plan is satisfied by the current evidence:
faithful original PWM collapses on MJLab while original DFlex parity holds, and
official NEWT/LeWM setup appears cheap enough to reproduce in parallel.

## Official Repositories

| Project | URL | Local path | Commit | Official env setup |
| --- | --- | --- | --- | --- |
| NEWT | `https://github.com/nicklashansen/newt` | `/storage/project/r-agarg35-0/eliu354/external_repos/newt` | `1d3fc058b81ddf8d36a5457c29ea407dd374c1b8` | `conda env create -f docker/environment.yaml`; `conda activate newt`; `pip install --no-cache-dir 'ale_py==0.10'` |
| LeWM | `https://github.com/lucas-maes/le-wm` | `/storage/project/r-agarg35-0/eliu354/external_repos/le-wm` | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` | `uv venv --python=3.10`; `source .venv/bin/activate`; `uv pip install stable-worldmodel[train,env]` |

Environment targets use the official dependency specifications but install into
explicit project paths:

```text
NEWT env: /storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602
LeWM env: /storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602
LeWM uv tool prefix: /storage/project/r-agarg35-0/eliu354/tools/uv_official_20260602
External data root: /storage/project/r-agarg35-0/eliu354/external_data
```

## Candidates Before Submission

| Candidate | Type | Purpose | Inputs exist? | W&B mode | Expected artifacts | GPU / QOS | Dependency required? | Submit decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `newt_official_env_setup_20260602` | setup | Create the official NEWT conda environment from `docker/environment.yaml` and install `ale_py==0.10` as specified by the README. | Yes: official repo and environment file exist. | Off. | Slurm logs under `logs/slurm/image_official/`; env under `/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602`. | CPU / `embers`. | No. | Submit. |
| `lewm_official_env_setup_20260602` | setup | Create the official LeWM uv virtualenv and install `stable-worldmodel[train,env]`. | Yes: official repo exists. `uv` is not globally installed, so the wrapper installs the `uv` tool into a project-local prefix before running the official venv command. | Off. | Slurm logs under `logs/slurm/image_official/`; env under `/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602`. | CPU / `embers`. | No. | Submit. |
| `newt_official_import_config_smoke_20260602` | smoke | Import official NEWT modules, parse `walker-walk` config, and list task/model metadata. | Depends on NEWT env setup. Official repo and `tasks.json` exist. | Off. | Slurm logs with task count, model sizes, action dim, parsed config. | CPU / `embers`. | `afterok:newt_official_env_setup_20260602`. | Submit. |
| `newt_official_walker_smoke_a100_20260602` | smoke / exploratory | Run the smallest official NEWT single-task train smoke on DMControl `walker-walk`, W&B disabled, no demos, no video, no checkpoint save. This is an official runner smoke, not a performance row. | Depends on NEWT env setup. Uses DMControl task to avoid ManiSkill asset dependency in the first smoke. | Off (`enable_wandb=false`, `WANDB_MODE=disabled`). | Slurm logs and NEWT local `logs/walker-walk/0/official_walker_smoke_20260602` if the smoke reaches logger setup. | A100 / `embers`. | `afterok:newt_official_env_setup_20260602`. | Submit. |
| `lewm_official_import_config_smoke_20260602` | smoke | Import official LeWM modules, compose the training config, and verify SIGReg/model classes without requiring datasets. | Depends on LeWM env setup. Official repo/configs exist. | Off. | Slurm logs with torch/stable-worldmodel/stable-pretraining imports, train data name, max epochs, SIGReg weight, and class names. | CPU / `embers`. | `afterok:lewm_official_env_setup_20260602`. | Submit. |
| `lewm_official_data_checkpoint_download_20260602` | setup / diagnostic | Identify and download the smallest official LeWM dataset/checkpoint needed for an official train/eval smoke. | Missing locally: no `.h5`/`.lance` LeWM datasets were found under `/storage/project/r-agarg35-0/eliu354`. | Off. | HF repo/file inventory, downloaded dataset/checkpoint paths, or failure reason. | CPU / `embers`. | Depends on LeWM env only if using its HF tooling. | Prepare after env setup; do not submit a train/eval job until inputs are known. |
| `lewm_official_pusht_assets_20260602` | setup / diagnostic | Use the healthy official LeWM env to inventory `quentinll/lewm-pusht`, download PushT dataset/model assets, decompress `pusht_expert_train.h5.zst`, convert `weights.pt + config.json` into `pusht/lewm_object.ckpt`, and smoke-load `AutoCostModel('pusht/lewm')`. | Yes: LeWM env `9398556` completed; HF model repo lists `config.json` and `weights.pt`; HF dataset repo lists `pusht_expert_train.h5.zst`. | Off. | Files under `$STABLEWM_HOME=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm`, Slurm logs under `logs/slurm/image_official/`, converted checkpoint or failure reason. | CPU / `embers`. | No Slurm dependency needed because the env already exists and passed smoke. | Submit after committing script. |

Submission wrapper:

```text
scripts/experiments/image_official/submit_newt_lewm_official_smokes_20260602.sh
scripts/experiments/image_official/submit_lewm_official_pusht_assets_20260602.sh
```

Validation before submission:

```text
bash -n scripts/experiments/image_official/submit_newt_lewm_official_smokes_20260602.sh
official NEWT remote HEAD: 1d3fc058b81ddf8d36a5457c29ea407dd374c1b8
official LeWM remote HEAD: 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
local search found no existing NEWT/LeWM clone before cloning
local search found no existing LeWM `.h5` or `.lance` datasets under /storage/project/r-agarg35-0/eliu354
```

## Submitted Jobs

Submitted after commit `a428513610a636cde59f9cb8ce5621b204115760`:

```text
9398480 newt_official_env_setup_20260602
9398481 lewm_official_env_setup_20260602
9398482 newt_official_import_config_smoke_20260602, dependency=afterok:9398480
9398483 newt_official_walker_smoke_a100_20260602, dependency=afterok:9398480
9398484 lewm_official_import_config_smoke_20260602, dependency=afterok:9398481
```

Initial scheduler status:

```text
9398480 RUNNING on cpu-small
9398481 RUNNING on cpu-small
9398482 PENDING Dependency
9398483 PENDING Dependency on gpu-a100 / embers
9398484 PENDING Dependency
```

Cancellation before replacement:

```text
9398480 CANCELLED after 00:01:00?, QOS inferno
9398481 CANCELLED after 00:01:15, QOS inferno
9398482 CANCELLED before start, dependency child of 9398480, QOS inferno
9398483 CANCELLED before start, dependency child of 9398480, QOS embers
9398484 CANCELLED before start, dependency child of 9398481, QOS inferno
root_cause: CPU jobs inherited the account default QOS `inferno` because the first wrapper did not explicitly pass a CPU QOS.
fix: add CPU_QOS=embers and `--qos=${CPU_QOS}` to all CPU submissions, keep GPU_QOS=embers for GPU submissions, and remove incomplete env directories before recreating them.
validation: `sbatch --test-only --partition=cpu-small --qos=embers ...` succeeded.
replacement_decision: resubmit the same official-env/setup smoke batch after committing the QOS repair.
```

Replacement submitted after commit
`9e2e0a3c9942844823b20b99db1862b3f7c7d564`:

```text
9398555 newt_official_env_setup_20260602, QOS embers, FAILED 1:0
9398556 lewm_official_env_setup_20260602, QOS embers, RUNNING at first check
9398557 newt_official_import_config_smoke_20260602, dependency=afterok:9398555, canceled after setup failure
9398558 newt_official_walker_smoke_a100_20260602, dependency=afterok:9398555, canceled after setup failure
9398559 lewm_official_import_config_smoke_20260602, dependency=afterok:9398556, pending at first check
root_cause_newt: the canceled first submission left a partial conda env with `bin/python` present but no working standard-library `encodings` module.
fix_newt: before reusing an env, run `python -c 'import encodings'`; remove and recreate the env if the sanity check fails.
replacement_decision_newt: resubmit only the NEWT setup/import/walker smoke with the official conda YAML after committing the env sanity repair.
```

NEWT repair resubmission after commit
`0c75280e05d529bff0d3dd380a2da8c36fdd6e4c`:

```text
9398617 newt_official_env_setup_repair_20260602, QOS embers, FAILED 1:0
9398618 newt_official_import_config_repair_20260602, dependency=afterok:9398617, canceled after setup failure
9398619 newt_official_walker_repair_a100_20260602, dependency=afterok:9398617, canceled after setup failure
root_cause_newt_repair: official `conda env create -f docker/environment.yaml` reached pip dependency installation but `box2d-py` failed to build because the `swig` executable was not on PATH.
fix_newt_repair: keep the official YAML unchanged, but load the cluster `swig/4.1.1` module before `conda env create`; export `PYTHONNOUSERSITE=1`; add a setup completion marker so failed partial envs are removed before reuse.
replacement_decision_newt_repair: resubmit only the NEWT setup/import/walker smoke after committing the SWIG/module and marker fix.
```

LeWM replacement result:

```text
9398556 lewm_official_env_setup_20260602 COMPLETED 0:0, QOS embers, elapsed 00:01:49
env evidence: lewm_env_python_ok; torch 2.12.0+cu130; hydra 1.3.2; stable_worldmodel and stable_pretraining import from /storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602
9398559 lewm_official_import_config_smoke_20260602 COMPLETED 0:0, QOS embers, elapsed 00:00:11
config evidence: lewm_import_config_ok; data=pusht_expert_train.lance; trainer.max_epochs=1; loss.sigreg.weight=0.09; JEPA/ARPredictor/Embedder/MLP/SIGReg classes import successfully.
next_lewm: official env/config smoke is healthy; prepare a data/checkpoint inventory/download job before any LeWM train/eval submission because no local `.h5`/`.lance` data was found earlier.
```

LeWM PushT assets candidate validation before submission:

```text
official README data path: download data from HuggingFace collection, decompress, place extracted files under $STABLEWM_HOME.
official README checkpoint path: `hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht`, then convert `weights.pt + config.json` into `$STABLEWM_HOME/pusht/lewm_object.ckpt`.
HF model inventory via official env: `quentinll/lewm-pusht` model repo has `.gitattributes`, `README.md`, `config.json`, `weights.pt`.
HF dataset inventory via official env: `quentinll/lewm-pusht` dataset repo has `.gitattributes`, `README.md`, `pusht_expert_train.h5.zst`.
script: `scripts/experiments/image_official/submit_lewm_official_pusht_assets_20260602.sh`
```

Submitted after commit
`dd41bea5f3a34537043dbe5170d02b2e4107ada2`:

```text
9398711 newt_official_env_setup_swig_20260602, QOS embers, RUNNING at first check
9398712 lewm_official_pusht_assets_20260602, QOS embers, RUNNING at first check
9398713 newt_official_import_config_swig_20260602, dependency=afterok:9398711, QOS embers, PENDING at first check
9398714 newt_official_walker_swig_a100_20260602, dependency=afterok:9398711, QOS embers, PENDING at first check
```

LeWM assets failure and replacement candidate:

```text
9398712 lewm_official_pusht_assets_20260602 FAILED 1:0 after 00:04:09, QOS embers.
completed_before_failure: HF inventory succeeded; model `config.json` and `weights.pt` downloaded to `$STABLEWM_HOME/hf_pusht`; dataset archive `pusht_expert_train.h5.zst` downloaded to `$STABLEWM_HOME/downloads`; dataset decompressed to `$STABLEWM_HOME/pusht_expert_train.h5` with size 46300921856 bytes.
failure: checkpoint conversion failed when `stable_pretraining.backbone` imported old `datasets` code against installed `pyarrow`, first missing `PyExtensionType`, then missing `datasets.config` during direct compatibility probing.
replacement_candidate: `lewm_official_pusht_assets_fix1_20260602`; use the same official env and already-downloaded assets, but load the official installed `stable_pretraining/backbone/utils.py` file directly for `vit_hf` so conversion avoids unrelated `stable_pretraining.data.datasets` imports.
resources: CPU / `embers`; W&B disabled; no Slurm dependency because assets and env already exist.
expected_artifacts: `$STABLEWM_HOME/pusht/lewm_object.ckpt`, `AutoCostModel('pusht/lewm')` smoke-load evidence, Slurm logs.
submit_decision: submit after committing the script and record.
```

NEWT SWIG setup result and import replacement candidate:

```text
9398711 newt_official_env_setup_swig_20260602 COMPLETED 0:0 after 00:08:03, QOS embers.
env evidence: official conda env completed after loading cluster `swig/4.1.1`; `box2d-py` built successfully; `ale_py==0.10.0` installed; import smoke printed torch 2.8.0+cu128, torchvision 0.23.0+cu128, hydra 1.3.2, gymnasium 0.29.1; marker `.newt_official_setup_ok_20260602` exists.
9398713 newt_official_import_config_swig_20260602 FAILED 1:0 after 00:00:04, QOS embers.
9398714 newt_official_walker_swig_a100_20260602 was canceled before start because it depended only on setup and would likely reuse the same bad smoke assumptions.
failure: import/config smoke called `parse_cfg` outside Hydra runtime, causing `ValueError: get_original_cwd() must only be used after HydraConfig is initialized`.
replacement_candidate: `newt_official_import_config_swig_fix1_20260602` plus `newt_official_walker_swig_fix1_a100_20260602`; the import smoke should verify official modules/tasks/config dataclass without calling `parse_cfg`, while the walker smoke should continue to use the official `train.py` Hydra entrypoint.
resources: import CPU / `embers`; walker A100 / `embers`; no dependency needed because the official env setup succeeded.
script: `scripts/experiments/image_official/submit_newt_official_swig_followups_20260602.sh`.
submit_decision: submit after committing the wrapper repair and record.
```

Replacement submissions after commit
`c7c33b54d256d07355a88f95bc7c4cf506fd85b8`:

```text
9399731 lewm_official_pusht_assets_fix1_20260602, CPU / embers, PENDING Priority at first check
9399797 newt_official_import_config_swig_fix1_20260602, CPU / embers, PENDING Priority at first check
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
local_fix2_validation: strict object conversion succeeded under `/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602/bin/python`; the local conversion also normalized legacy HuggingFace ViT state-dict keys such as `encoder.encoder.layer.*.attention.attention.query.*` to the installed Transformers key schema `encoder.layers.*.attention.q_proj.*`, then loaded with `strict=True`.
resources: CPU / `embers`; W&B disabled; no dependency.
submit_decision: submit after committing the conversion helper fix.
```

Broad GPU submission candidates requested 2026-06-02:

```text
request: increase GPU submission count; submit embers GPU tasks aggressively because embers is not charged.
preflight: current running/pending user jobs were only 9399798 Hopper H100 probe and 9399799 NEWT walker A100 smoke.
script: scripts/experiments/image_official/submit_newt_lewm_broad_gpu_20260602.sh
candidate_newt_broad: `newt_official_broad_smoke_a100_20260602`, 16 array elements, A100 / embers, array 0-15%8, tasks walker-walk/walker-run/cheetah-run/hopper-hop/reacher-easy/pendulum-swingup/cartpole-swingup/cup-catch x seeds 0/1, official NEWT env, W&B/video/checkpoint off, 500 steps per task.
candidate_lewm_eval: `lewm_official_pusht_eval_h100_20260602`, 6 array elements, H100 / embers, array 0-5%6, official LeWM env, uses converted `pusht/lewm_object.ckpt` and `pusht_expert_train.h5`, CEM eval rows plus random baseline rows.
candidate_lewm_train: `lewm_official_pusht_train_smoke_h100_20260602`, 2 array elements, H100 / embers, array 0-1%2, official `python train.py data=pusht` entrypoint, short run smoke with `max_epochs=1`, `limit_train_batches=2`, `limit_val_batches=1`, W&B off.
inputs: NEWT marker exists; LeWM env exists; LeWM PushT object checkpoint and 44GB dataset exist after local strict conversion validation.
submit_decision: submit after committing the broad wrapper and candidate record.
```

Broad official image submissions after commit `3a3e161`:

```text
9400408 lewm_official_pusht_assets_fix2_20260602, CPU / embers, PENDING Priority at first check.
9400409_[0-15%8] newt_official_broad_smoke_a100_20260602, A100 / embers, PENDING Priority at first check.
9400411_[0-5%6] lewm_official_pusht_eval_h100_20260602, H100 / embers, PENDING Priority at first check.
9400412_[0-1%2] lewm_official_pusht_train_smoke_h100_20260602, H100 / embers, PENDING Priority at first check.
array_payload: 16 NEWT official train smokes, 6 LeWM official eval rows, 2 LeWM official train smokes.
scheduler_validation: `squeue` and `sacct` both showed QOS `embers` for all submitted official-image jobs.
```

LeWM fix2 completion:

```text
9400408 lewm_official_pusht_assets_fix2_20260602 COMPLETED 0:0 after 00:00:41, QOS embers.
evidence: HF inventory succeeded; `config.json`, `weights.pt`, and `pusht_expert_train.h5.zst` were reused/downloaded; `dataset_h5` size was 46300921856 bytes; converted checkpoint `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/pusht/lewm_object.ckpt` size was 72334197 bytes; `autocost_load_ok JEPA`; `lewm_official_pusht_assets_ok`.
```

Continuation poll after commit `12fd08e`:

```text
seff: unavailable on PATH; status used `squeue` and `sacct`.
9400409_[0-15%8] newt_official_broad_smoke_a100_20260602 remained PENDING Priority, A100 / embers.
9399799 newt_official_walker_swig_fix1_a100_20260602 remained PENDING Priority, A100 / embers.
9400411_[0-5%6] lewm_official_pusht_eval_h100_20260602 remained PENDING Priority, H100 / embers.
9400412_[0-1%2] lewm_official_pusht_train_smoke_h100_20260602 remained PENDING Priority, H100 / embers.
no_new_official_image_submission_reason: useful official NEWT/LeWM GPU smokes are already queued, and extra submissions are blocked by embers submit quota until pending jobs start or finish.
next_action: inspect official-image logs as soon as array elements leave pending; record failures and replacement IDs before any repair submission.
```

L40S official backup preparation:

```text
script: scripts/experiments/image_official/submit_newt_lewm_l40s_backups_20260602.sh
purpose: keep NEWT and LeWM moving on the L40S tier once embers submit quota reopens, without overwriting the existing A100/H100/H200 official-image jobs.
candidate_newt_l40s: 16 official NEWT train-smoke rows, tasks walker-walk/walker-run/cheetah-run/hopper-hop/reacher-easy/pendulum-swingup/cartpole-swingup/cup-catch x seeds 0/1, W&B/video/checkpoint disabled, unique `official_broad_l40s_*` experiment names, L40S / embers, 4 CPUs per GPU.
candidate_lewm_eval_l40s: 6 official LeWM PushT eval rows, same CEM/random payload as the H100/H200 eval jobs, unique `_l40s_results.txt` output filenames, L40S / embers, 4 CPUs per GPU.
candidate_lewm_train_l40s: 2 official LeWM PushT train-smoke rows, seeds 0/1, `max_epochs=1`, `limit_train_batches=2`, `limit_val_batches=1`, W&B disabled, unique `official_train_smoke_l40s_*` subdirs, L40S / embers, 4 CPUs per GPU.
validation: `bash -n` passed; input check passed for NEWT marker, LeWM env python, converted `pusht/lewm_object.ckpt`, and `pusht_expert_train.h5`.
submit_probe: `sbatch --test-only` on both gpu-l40s and gpu-h200 returned `QOSMaxSubmitJobPerUserLimit`, so no new official image jobs were submitted in this poll.
next_action: run the script after any queued embers job starts or finishes and `sbatch --test-only` succeeds; record accepted job IDs before interpreting results.
```

LeWM official eval environment repair:

```text
9400532_0..5 `lewm_official_pusht_eval_h200_20260602` FAILED 1:0 on H200 / embers before evaluation.
failure_signature: `stable_pretraining.data.datasets` imported HuggingFace `datasets`, which failed under the official env with `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'`.
prevented_repeat_failures: 9400411 `lewm_official_pusht_eval_h100_20260602` and 9400412 `lewm_official_pusht_train_smoke_h100_20260602` were canceled before start.
repair: enabled pip in `/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602`; kept `pyarrow==24.0.0`; added `scripts/experiments/image_official/compat/sitecustomize.py` as a batch-visible compatibility shim; upgraded `datasets` to 3.6.0.
validation: local official-env import smoke passed for `pyarrow`, `datasets`, `stable_pretraining`, `stable_worldmodel`, and `stable_worldmodel.policy.AutoCostModel`; `eval.py --config-name=pusht --cfg job` composes the intended PushT config.
script_updates: LeWM GPU wrappers now prepend `${COMPAT_ROOT}` to `PYTHONPATH`.
replacement_script: scripts/experiments/image_official/submit_newt_h200_remaining_lewm_h200_fix_20260602.sh
replacement_payload: remaining NEWT H200 rows 7-15, LeWM H200 eval fix rows 0-5, and LeWM H200 train-smoke fix rows 0-1 with unique output names.
validation_command: `bash -n scripts/experiments/image_official/submit_newt_lewm_broad_gpu_20260602.sh scripts/experiments/image_official/submit_newt_lewm_l40s_backups_20260602.sh scripts/experiments/image_official/submit_newt_h200_remaining_lewm_h200_fix_20260602.sh`
next_action: submit the H200 replacement script after this record is committed, if embers submit quota accepts it.
```

L40S official backup submission:

```text
accepted_jobs: 9400714 `newt_official_broad_l40s_20260602` array 0-15%4; 9400715 `lewm_official_pusht_eval_l40s_20260602` array 0-5%3; 9400716 `lewm_official_pusht_train_l40s_20260602` array 0-1%2.
qos_gpu: gpu-l40s / embers for all three arrays.
early_result: 9400714_0 walker-walk seed0 COMPLETED 0:0 in 00:01:01 with eval R 42.247 and train R 51.202.
early_result: 9400714_1 walker-run seed0 COMPLETED 0:0 in 00:00:32 with eval R 42.179 and train R 23.809.
pending_at_record: 9400714_2..15, 9400715_0..5, and 9400716_0..1 were pending Priority behind the running L40S array work.
next_action: inspect NEWT L40S row logs as they complete; inspect LeWM L40S logs immediately when they start because they exercise the repaired LeWM env under Slurm.
```

H200 repair submission after commit `ce439b0`:

```text
large_array_attempt: the committed H200 repair wrapper failed at its first array with `QOSMaxSubmitJobPerUserLimit`.
single_row_strategy: single-row `sbatch --test-only` probes passed on H200 and L40S, so H200 repair work was split into one-row array submissions.
submitted_lewm_eval_fix_rows:
  9400771_[0] lewm seed0 horizon2
  9400772_[1] lewm seed1 horizon2
  9400773_[2] lewm seed2 horizon2
  9400774_[3] lewm seed0 horizon5
  9400775_[4] random seed0 horizon2
  9400776_[5] random seed1 horizon2
status_at_first_check: all six H200 / embers rows were pending Resources.
submitted_newt_h200_remaining: 9400778_[7] cup-catch seed0, H200 / embers, pending Resources.
blocked_submission: NEWT H200 row 8 failed with `QOSMaxSubmitJobPerUserLimit`; NEWT H200 rows 8-15 and LeWM H200 train-smoke singles remain candidates for the next free submit slots.
concurrent_l40s_progress: 9400714_0..6 completed 0:0; rows map to walker-walk, walker-run, cheetah-run, hopper-hop, reacher-easy, pendulum-swingup, and cartpole-swingup seed0; all produced official NEWT training output.
next_action: do not cancel pending H200 repair rows; inspect their logs as soon as they start, then resubmit only failed rows with recorded root causes.
```

NEWT/LeWM continuation on 2026-06-02:

```text
completed_h200_newt_rows: `9400537_0-3`, `9400543_4`, `9400544_5`, and `9400545_6` completed 0:0 on gpu-h200 / embers using the official NEWT runner. Observed smoke metrics: walker-walk seed0 eval R 42.247 train R 51.202; walker-run seed0 eval R 42.179 train R 23.809; cheetah-run seed0 eval R 5.871 train R 6.516; hopper-hop seed0 eval/train 0.0; reacher-easy seed0 eval R 34.000 train R 0.0; pendulum-swingup seed0 eval/train 0.0; cartpole-swingup seed0 eval R 183.107 train R 8.803.
failed_h200_lewm_eval: `9400532_0-5` failed 1:0 on gpu-h200 / embers before evaluation because `datasets` imported against a `pyarrow` build without `PyExtensionType`.
canceled_prior_lewm_gpu: `9400411_[0-5]` and `9400412_[0-1]` were canceled before start, so they produced no LeWM eval/train evidence.
compat_fix: added `scripts/experiments/image_official/compat/sitecustomize.py` and patched LeWM eval/train submit payloads to prepend that directory to `PYTHONPATH`; the shim aliases `pyarrow.PyExtensionType` to `pyarrow.ExtensionType` only when the old name is absent.
compat_validation: lightweight LeWM env check loaded the repo `sitecustomize.py` and printed `pyarrow 24.0.0`, `has_PyExtensionType True`, `has_ExtensionType True`. A full `stable_pretraining` import on the login node was killed after heavy PyTorch/JAX import probing, so final validation is delegated to GPU Slurm rows.
l40s_submission: submitted `9400714` NEWT official broad L40S array, `9400715` LeWM official PushT eval L40S array, and `9400716` LeWM official PushT train-smoke L40S array, all QOS embers.
l40s_early_result: `9400714_0-3` completed 0:0; row0 walker-walk seed0 eval R 42.247 train R 51.202; row1 walker-run seed0 eval R 42.179 train R 23.809; row2 cheetah-run seed0 eval R 5.871 train R 6.516; row3 hopper-hop seed0 eval/train 0.0. Remaining NEWT L40S rows plus LeWM L40S eval/train are pending.
h200_fix_wrapper: added `scripts/experiments/image_official/submit_newt_h200_remaining_lewm_h200_fix_20260602.sh` for NEWT H200 rows 7-15 only, plus shimmed LeWM H200 eval/train replacement rows.
h200_submit_blocker: `sbatch --test-only` on gpu-h200 / embers returned `QOSMaxSubmitJobPerUserLimit`, so the H200 wrapper was validated but not submitted in this poll.
next_action: when the submit quota reopens, run `scripts/experiments/image_official/submit_newt_h200_remaining_lewm_h200_fix_20260602.sh`; monitor `9400715` first because it is the direct replacement for the failed LeWM H200 eval root cause.
```

NEWT/LeWM poll before H200 replacement submission:

```text
poll_time: 2026-06-02 19:10-19:13 America/New_York.
newt_l40s_completed: `9400714_0-6` completed 0:0 on gpu-l40s / embers. Row metrics: walker-walk seed0 eval/train `42.247/51.202`; walker-run `42.179/23.809`; cheetah-run `5.871/6.516`; hopper-hop `0.0/0.0`; reacher-easy `34.000/0.0`; pendulum-swingup `0.0/0.0`; cartpole-swingup `183.107/8.803`.
lewm_l40s_pending: `9400715_[0-5]` and `9400716_[0-1]` remain pending Priority, so the repaired LeWM env has not yet been tested under GPU Slurm.
h200_submit_probe: gpu-h200 / embers `sbatch --test-only` succeeded after the inventory check.
candidate: submit `scripts/experiments/image_official/submit_newt_h200_remaining_lewm_h200_fix_20260602.sh` after this record is committed. It is non-duplicative: remaining NEWT H200 rows 7-15 were never accepted before, and LeWM H200 rows use unique `_h200_fix` result names after the pyarrow/datasets repair.
```

NEWT H200 remaining single-row extension:

```text
accepted_before: 9400778_[7] cup-catch seed0 was already accepted on H200 / embers.
accepted_after_a0a0751: 9400797_[8] walker-walk seed1; 9400798_[9] walker-run seed1; 9400799_[10] cheetah-run seed1; 9400800_[11] hopper-hop seed1.
blocked_after_a0a0751: row 12 failed with `QOSMaxSubmitJobPerUserLimit`; rows 12-15 remain useful unsent H200 backup candidates.
status_at_first_check: accepted rows 7-11 are pending Resources.
l40s_overlap_progress: L40S NEWT rows 7, 8, and 9 completed 0:0 with eval/train rewards `0.0/0.0`, `18.338/37.122`, and `18.157/28.181`.
interpretation: NEWT official environment continues to be healthy on L40S. H200 rows are queued as backup coverage, not duplicates to interpret independently until logs complete.
next_action: retry rows 12-15 and LeWM H200 train-smoke singles only when submit slots free; monitor LeWM eval fix rows first.
```

NEWT H200 remaining rows 12-15 accepted:

```text
accepted_after_903642d: 9400814_[12] reacher-easy seed1; 9400815_[13] pendulum-swingup seed1; 9400816_[14] cartpole-swingup seed1; 9400817_[15] cup-catch seed1.
status_at_first_check: all four were pending Resources on gpu-h200 / embers.
coverage_status: NEWT H200 backup rows 7-15 are now fully queued; avoid duplicate H200 submissions for these rows.
lewm_eval_h200_status: repaired LeWM row 9400771_0 started on H200 / embers and initially showed only Slurm prolog output, with no immediate repeat of the pyarrow import failure.
next_action: prioritize monitoring 9400771_0; submit LeWM H200 train-smoke singles only if submit slots reopen and repaired eval does not expose a new root cause.
```

LeWM HDF5 plugin repair:

```text
failed_job: 9400771_0 `lewm_official_pusht_eval_h200_fix_row0_20260602` FAILED 1:0 after 00:01:29.
failure_interpretation: this is a new official environment/data-format issue, not the previous pyarrow issue. The job reached Hydra and dataset construction, then failed because `stable_worldmodel.data.HDF5Dataset` was not exported.
underlying_cause: `stable_worldmodel.data.formats.hdf5` requires `hdf5plugin`; the official env is read-only and did not have `hdf5plugin` installed. The PushT `pixels` dataset uses an unknown HDF5 plugin filter and cannot be read without plugin registration.
canceled_to_prevent_repeat: 9400772_[1]..9400776_[5], 9400715, and 9400716.
repair: repo-local vendor install under `scripts/experiments/image_official/compat/vendor/` using `scripts/experiments/image_official/install_lewm_compat_vendor_20260602.sh`; vendor is intentionally git-ignored.
validation: batch-visible `sitecustomize.py` adds the vendor path, exposes `HDF5Dataset`, and allowed reading `pusht_expert_train.h5` `pixels[0]`.
next_action: resubmit LeWM eval replacement rows after committing the repair and checking quota; use unique result/log names to distinguish from 9400771..9400776.
```

LeWM HDF5-fix H200 replacement submission:

```text
commit_before_submission: 8270506.
script: scripts/experiments/image_official/submit_lewm_hdf5fix_h200_20260602.sh.
submitted_jobs:
  9401543_[0-5%3] lewm_official_pusht_eval_hdf5fix_h200_20260602, gpu-h200 / embers.
  9401544_[0-1%2] lewm_official_pusht_train_hdf5fix_h200_20260602, gpu-h200 / embers.
inputs: official LeWM env, converted PushT object checkpoint, PushT HDF5 dataset, and repo-local hdf5plugin vendor/compat shim all exist.
wandb_mode: disabled.
expected_artifacts:
  logs/slurm/image_official/lewm_official_pusht_eval_hdf5fix_h200_%A_%a.{out,err}
  logs/slurm/image_official/lewm_official_pusht_train_hdf5fix_h200_%A_%a.{out,err}
  official LeWM eval result files with `_h200_hdf5fix_results.txt` suffixes.
first_scheduler_check:
  squeue showed both arrays PENDING Priority with QOS embers and gres/gpu:h200:1.
  sacct showed both arrays PENDING, QOS embers, exit 0:0.
next_action: inspect the first started eval row immediately; if it fails, cancel the sibling rows, record the new root cause, and repair before resubmitting.
```

Continuation poll after HDF5-fix submission:

```text
9401543_[0-5%3] LeWM HDF5-fix eval remains PENDING Priority on gpu-h200 / embers; no logs yet.
9401544_[0-1%2] LeWM HDF5-fix train remains PENDING Priority on gpu-h200 / embers; no logs yet.
9400714_[0-15] NEWT official broad L40S completed 0:0. Eval/train rewards by row were:
  0 42.247/51.202
  1 42.179/23.809
  2 5.871/6.516
  3 0.000/0.000
  4 34.000/0.000
  5 0.000/0.000
  6 183.107/8.803
  7 0.000/0.000
  8 18.338/37.122
  9 18.157/28.181
  10 13.833/5.912
  11 0.142/0.000
  12 10.000/0.000
  13 0.000/0.000
  14 144.140/14.021
  15 985.000/0.000
interpretation: official NEWT L40S is now fully smoke-covered; do not submit duplicate NEWT L40S rows. Wait for LeWM HDF5-fix rows to start.
```
