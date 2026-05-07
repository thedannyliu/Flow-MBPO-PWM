# PWM Extension Submission Plan - 2026-05-07

## Goal

This batch extends the current MJLab-QS PWM-style pipeline in four ordered stages requested on 2026-05-07:

1. add pure Flow WM (`flow_ref`) to the same policy-extraction/final-eval stage,
2. add PWM-style online finetuning for MLP WM vs Flow endpoint WM with MLP policy,
3. run the offline-pretrain -> policy-extraction 2x2 table with MLP policy vs Flow policy,
4. run the online-finetune 2x2 table.

Each formal stage uses its own W&B project. Smoke tests are submitted first with W&B disabled.

## Code Changes

### `run_offline_pwm_policy_extraction.py`

New support:

- `--wm-method flow_ref` in addition to `mlp_ref` and `flow_endpoint`.
- `--policy-type {mlp,flow}`.
- Flow policy uses `ActorFlowODE` from `src/flow_mbpo_pwm/models/flow_actor.py`.
- Flow policy is optimized with the same PWM-style first-order imagined-return actor objective; it is not FPO/PPO-ratio training.
- Optional PWM-style online finetune loop:
  1. collect real MJLab windows with the current extracted policy,
  2. finetune the state-space WM on those online windows,
  3. continue actor-critic optimization through imagined rollouts,
  4. final-evaluate the resulting policy in real MJLab.

Important limitation:

- This online finetune loop is a state-space MJLab-QS approximation of PWM online finetuning. It uses normalized physical state as the latent state, not the original PWM learned encoder checkpoint format.

### `run_policy_extraction_row.py`

New support:

- passes `policy_type`, Flow policy integrator/substeps, and online finetune settings from manifest rows,
- writes outputs under `wm_method/policy_type/online_profile/compute_profile/seed_*` to avoid 2x2 result overwrites.

## Smoke Test

Smoke manifest:

- `scripts/outputs/mjlab_qs/manifests/smoke_pwm_extensions_20260507.csv`

Smoke rows:

1. `flow_ref + mlp_policy`, offline, no real eval,
2. `mlp_ref + flow_policy`, offline, no real eval,
3. `flow_endpoint + flow_policy`, one tiny online-finetune round, no final real eval.

W&B is disabled for smoke.

Submitted smoke job:

| Stage | GPU | Job ID | Dependency | Notes |
|---|---|---:|---|---|
| smoke | L40S | `5268668` | none | Uses `/storage/ice1/2/9/eliu354/conda_envs/flow-mbpo/bin/python` |

The first smoke attempt failed because Slurm used a Python without torch. A second smoke attempt exposed and fixed an online-finetune indexing bug. The current smoke/formal chain pins the Python interpreter to the `flow-mbpo` conda environment, which has `torch`, `wandb`, `mjlab`, and `tensordict`.

## Formal Stage 1: Pure Flow WM to Final Eval

Purpose:

- Run pure Flow Matching WM (`flow_ref`) to the same offline policy-extraction/final-eval stage as the current MLP/Flow-endpoint runs.

W&B project:

- `flow-mbpo-mjlab-pure-flow-pwm-final-eval`

Manifest:

- `scripts/outputs/mjlab_qs/manifests/pure_flow_pwm_final_eval_20260507.csv`

Submitted jobs:

| GPU | Job ID | Manifest | Dependency |
|---|---:|---|---|
| H100 | `5268673` | `pure_flow_pwm_final_eval_20260507_seed0_h100.csv` | `afterok:5268668` |
| H200 | `5268674` | `pure_flow_pwm_final_eval_20260507_seed1_h200.csv` | `afterok:5268668` |
| L40S | `5268675` | `pure_flow_pwm_final_eval_20260507_seed2_l40s.csv` | `afterok:5268668` |

## Formal Stage 2: Online Finetune, MLP Policy, Two WMs

Purpose:

- Compare online finetuning for `MLP WM + MLP policy` vs `Flow endpoint WM + MLP policy`.
- This isolates the WM side before adding Flow policy.

W&B project:

- `flow-mbpo-mjlab-online-finetune-wm2`

Manifest:

- `scripts/outputs/mjlab_qs/manifests/online_finetune_wm2_mlp_policy_20260507.csv`

Online settings:

- `online_finetune_rounds = 2`
- `online_collect_windows = 512`
- `online_wm_iters = 2000`
- `online_policy_iters = 5000`

Submitted jobs:

| GPU | Job ID | Manifest | Dependency |
|---|---:|---|---|
| H100 | `5268681` | `online_finetune_wm2_mlp_policy_20260507_seed0_h100.csv` | `afterok:5268673:5268674:5268675` |
| H200 | `5268680` | `online_finetune_wm2_mlp_policy_20260507_seed1_h200.csv` | `afterok:5268673:5268674:5268675` |
| L40S | `5268682` | `online_finetune_wm2_mlp_policy_20260507_seed2_l40s.csv` | `afterok:5268673:5268674:5268675` |

## Formal Stage 3: Offline 2x2

Purpose:

- Run offline pretrain -> policy extraction -> final real eval with:
  - MLP WM vs Flow endpoint WM,
  - MLP policy vs Flow policy.

W&B project:

- `flow-mbpo-mjlab-offline-pwm-2x2`

Manifest:

- `scripts/outputs/mjlab_qs/manifests/offline_pwm_2x2_20260507.csv`

Submitted jobs:

| GPU | Job ID | Manifest | Dependency |
|---|---:|---|---|
| H100 | `5268689` | `offline_pwm_2x2_20260507_seed0_h100.csv` | `afterok:5268680:5268681:5268682` |
| H200 | `5268688` | `offline_pwm_2x2_20260507_seed1_h200.csv` | `afterok:5268680:5268681:5268682` |
| L40S | `5268690` | `offline_pwm_2x2_20260507_seed2_l40s.csv` | `afterok:5268680:5268681:5268682` |

## Formal Stage 4: Online 2x2

Purpose:

- Add the same online-finetune loop to the 2x2 comparison.

W&B project:

- `flow-mbpo-mjlab-online-pwm-2x2`

Manifest:

- `scripts/outputs/mjlab_qs/manifests/online_pwm_2x2_20260507.csv`

Submitted jobs:

| GPU | Job ID | Manifest | Dependency |
|---|---:|---|---|
| H100 | `5268693` | `online_pwm_2x2_20260507_seed0_h100.csv` | `afterok:5268688:5268689:5268690` |
| H200 | `5268694` | `online_pwm_2x2_20260507_seed1_h200.csv` | `afterok:5268688:5268689:5268690` |
| L40S | `5268695` | `online_pwm_2x2_20260507_seed2_l40s.csv` | `afterok:5268688:5268689:5268690` |

## Interpretation Rules

- Stage 1 checks whether pure FM dynamics can produce a usable extracted policy, even though its fixed-data rollout loss is much worse than endpoint Flow.
- Stage 2 answers whether online WM finetuning helps Flow endpoint or MLP more when the policy class is held fixed.
- Stage 3 is the first offline 2x2 table.
- Stage 4 is the online-finetune 2x2 table.
- If smoke fails, all dependent formal jobs should remain blocked. Fix smoke first, then resubmit the dependency chain.
