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
| `newt_official_env_setup_20260602` | setup | Create the official NEWT conda environment from `docker/environment.yaml` and install `ale_py==0.10` as specified by the README. | Yes: official repo and environment file exist. | Off. | Slurm logs under `logs/slurm/image_official/`; env under `/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602`. | CPU / no GPU QOS. | No. | Submit. |
| `lewm_official_env_setup_20260602` | setup | Create the official LeWM uv virtualenv and install `stable-worldmodel[train,env]`. | Yes: official repo exists. `uv` is not globally installed, so the wrapper installs the `uv` tool into a project-local prefix before running the official venv command. | Off. | Slurm logs under `logs/slurm/image_official/`; env under `/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602`. | CPU / no GPU QOS. | No. | Submit. |
| `newt_official_import_config_smoke_20260602` | smoke | Import official NEWT modules, parse `walker-walk` config, and list task/model metadata. | Depends on NEWT env setup. Official repo and `tasks.json` exist. | Off. | Slurm logs with task count, model sizes, action dim, parsed config. | CPU / no GPU QOS. | `afterok:newt_official_env_setup_20260602`. | Submit. |
| `newt_official_walker_smoke_a100_20260602` | smoke / exploratory | Run the smallest official NEWT single-task train smoke on DMControl `walker-walk`, W&B disabled, no demos, no video, no checkpoint save. This is an official runner smoke, not a performance row. | Depends on NEWT env setup. Uses DMControl task to avoid ManiSkill asset dependency in the first smoke. | Off (`enable_wandb=false`, `WANDB_MODE=disabled`). | Slurm logs and NEWT local `logs/walker-walk/0/official_walker_smoke_20260602` if the smoke reaches logger setup. | A100 / `embers`. | `afterok:newt_official_env_setup_20260602`. | Submit. |
| `lewm_official_import_config_smoke_20260602` | smoke | Import official LeWM modules, compose the training config, and verify SIGReg/model classes without requiring datasets. | Depends on LeWM env setup. Official repo/configs exist. | Off. | Slurm logs with torch/stable-worldmodel/stable-pretraining imports, train data name, max epochs, SIGReg weight, and class names. | CPU / no GPU QOS. | `afterok:lewm_official_env_setup_20260602`. | Submit. |
| `lewm_official_data_checkpoint_download_20260602` | setup / diagnostic | Identify and download the smallest official LeWM dataset/checkpoint needed for an official train/eval smoke. | Missing locally: no `.h5`/`.lance` LeWM datasets were found under `/storage/project/r-agarg35-0/eliu354`. | Off. | HF repo/file inventory, downloaded dataset/checkpoint paths, or failure reason. | CPU first; GPU not needed. | Depends on LeWM env only if using its HF tooling. | Prepare after env setup; do not submit a train/eval job until inputs are known. |

Submission wrapper:

```text
scripts/experiments/image_official/submit_newt_lewm_official_smokes_20260602.sh
```

Validation before submission:

```text
bash -n scripts/experiments/image_official/submit_newt_lewm_official_smokes_20260602.sh
official NEWT remote HEAD: 1d3fc058b81ddf8d36a5457c29ea407dd374c1b8
official LeWM remote HEAD: 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
local search found no existing NEWT/LeWM clone before cloning
local search found no existing LeWM `.h5` or `.lance` datasets under /storage/project/r-agarg35-0/eliu354
```
