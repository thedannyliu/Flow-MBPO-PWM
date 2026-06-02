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
