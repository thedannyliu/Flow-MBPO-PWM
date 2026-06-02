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
