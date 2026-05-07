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

## No-Dependency Resubmission - 2026-05-07

The user requested that the remaining 2x2 experiments should be submitted without waiting for the earlier dependency chain. To avoid output-path collisions with the already-submitted dependency-chain jobs, I created no-dependency manifest copies with new stage/group names:

- `offline_pwm_2x2_nodep_20260507`
- `online_pwm_2x2_nodep_20260507`

The original dependency-chain jobs were not cancelled.

### No-Dependency Offline 2x2

W&B project remains:

- `flow-mbpo-mjlab-offline-pwm-2x2`

Submitted jobs:

| GPU | Job ID | Manifest | Dependency |
|---|---:|---|---|
| H100 | `5270278` | `offline_pwm_2x2_nodep_20260507_seed0_h100.csv` | none |
| H200 | `5270280` | `offline_pwm_2x2_nodep_20260507_seed1_h200.csv` | none |
| L40S | `5270282` | `offline_pwm_2x2_nodep_20260507_seed2_l40s.csv` | none |

Initial scheduler state after submission:

- all pending with reason `None`, not dependency-blocked.

### No-Dependency Online 2x2

W&B project remains:

- `flow-mbpo-mjlab-online-pwm-2x2`

Submitted jobs:

| GPU | Job ID | Manifest | Dependency |
|---|---:|---|---|
| H100 | `5270279` | `online_pwm_2x2_nodep_20260507_seed0_h100.csv` | none |
| H200 | `5270281` | `online_pwm_2x2_nodep_20260507_seed1_h200.csv` | none |
| L40S | `5270283` | `online_pwm_2x2_nodep_20260507_seed2_l40s.csv` | none |

Initial scheduler state after submission:

- all pending with reason `None`, not dependency-blocked.

### Note on Duplicate Experiments

The dependency-chain jobs are still queued under the original stage names. The no-dependency jobs write to different output directories because the manifest `stage` and `wandb_group` fields were changed to the `_nodep_20260507` names. This prevents direct filesystem overwrite with the original dependency-chain jobs.

## Online Result Sanity Check - 2026-05-07

I checked the currently available online-finetune results for:

- stage: `online_finetune_wm2_mlp_policy_20260507`
- W&B project: `flow-mbpo-mjlab-online-finetune-wm2`
- comparison: `MLP WM + MLP policy` vs `Flow endpoint WM + MLP policy`
- online setting: 2 online finetune rounds, 512 collected windows per round, 2000 WM finetune iterations per round, 5000 additional actor-critic iterations per round.

### Mechanical Validity

The completed rows look mechanically valid:

- summaries were written for 5 of 6 expected rows,
- real-environment eval was not skipped,
- each completed row reports 40 eval episodes,
- `eval/resolved_task_id` is `Mjlab-Velocity-Flat-Unitree-G1`, matching the requested G1 task,
- no NaN/Inf values were found in `best_imagined_return`, `eval/return_mean`, `eval/return_std`, or `eval/episode_length_mean`,
- stdout shows the online WM finetune loss decreasing in the running Flow endpoint seed0 row.

Completed rows at check time:

| WM | Policy | Seed | Online Rounds | Eval Return Mean | Eval Return Std | Eval Episode Length Mean | Eval Episodes |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP ref | MLP | 0 | 2 | -4.655990 | 0.517542 | 53.775002 | 40 |
| MLP ref | MLP | 1 | 2 | -4.690122 | 1.340816 | 69.000000 | 40 |
| MLP ref | MLP | 2 | 2 | -5.884936 | 1.009590 | 66.599998 | 40 |
| Flow endpoint | MLP | 1 | 2 | -3.717136 | 0.539649 | 59.000000 | 40 |
| Flow endpoint | MLP | 2 | 2 | -3.632309 | 0.495255 | 59.200001 | 40 |

The remaining row at check time is:

- `Flow endpoint + MLP policy`, seed 0, Slurm array row `5268681_1`, still running.

### Running Row Check

For `5268681_1`, stdout shows online WM finetuning is active and numerically stable so far:

```text
online_wm/loss: 167.4056 -> 0.1924
online_wm/rollout_dyn_mse_H16: 51.1276 -> 0.1873
online_wm/reward_mse: 10.0200 -> 0.0027
```

The subsequent actor-critic updates continue to log finite values for imagined return, actor gradient norm, critic loss, and action norm. There is no sign of NaN/Inf or immediate training crash in the checked log tail.

### Caveats

This online result should be treated as a valid running/completed engineering result, but not yet as a final clean scientific conclusion:

1. The 3-seed Flow endpoint online result is incomplete until seed 0 finishes.
2. W&B reports non-monotonic step warnings during online WM finetuning. This does not invalidate stdout logs or final summaries, but some online WM scalar points may be ignored by W&B. A later cleanup should make W&B step indices globally monotonic for online WM + policy logging.
3. The MJLab environment log still shows active domain-randomization/event terms such as `randomize_terrain`, `foot_friction`, `encoder_bias`, and `base_com`. Therefore these online/eval runs should be labeled as MJLab default online/eval setting, not as a strict fixed-simulator canonical comparison.
4. This online loop remains the state-space MJLab-QS approximation of PWM online finetuning. It collects online windows, finetunes the state-space WM, and continues imagined actor-critic updates, but it is not a byte-identical reproduction of the original PWM learned-encoder training loop.

### Current Interpretation

The currently completed rows suggest Flow endpoint online finetuning is not obviously broken and is numerically stable. The two completed Flow endpoint seeds have better eval return than the three completed MLP seeds, but this should not be over-interpreted until:

- Flow endpoint seed 0 finishes,
- the no-dependency online 2x2 jobs complete,
- the W&B step logging issue is cleaned up for future runs,
- and the default-domain-randomization caveat is explicitly tracked in result tables.

## Baseline Return Evaluation and Capability Levers - 2026-05-07

### Why This Is Needed

The current extracted-policy returns are mostly in the `-3` to `-5` range. Negative return is plausible for this MJLab G1 task because weak policies terminate early and the reward includes many penalty terms. However, the absolute value is not interpretable without reference policies.

To make the results meaningful, I submitted a baseline-return evaluation stage using the same G1 task and a comparable 40-episode evaluation budget. This stage measures where the extracted PWM policies sit relative to:

1. random smooth actions,
2. the medium data-collector checkpoint,
3. the final MJLab-native PPO/expert collector checkpoint,
4. the same expert checkpoint with action noise.

### Submitted Baseline Return Jobs

Stage:

- `g1_policy_baseline_return_20260507`

Manifest:

- `scripts/outputs/mjlab_qs/manifests/g1_policy_baseline_return_20260507.csv`

Rows per seed:

| Baseline | Method | Checkpoint | Episodes | Purpose |
|---|---|---|---:|---|
| random_smooth | random_smooth | none | 40 | lower-bound reference |
| collector_medium_iter15000 | rslrl_ppo_conservative | `model_15000.pt` | 40 | medium collector reference |
| ppo_expert_iter29999 | rslrl_ppo_conservative | `model_29999.pt` | 40 | final native PPO / expert collector reference |
| ppo_expert_noisy_iter29999 | rslrl_ppo_conservative | `model_29999.pt` + action noise 0.05 | 40 | robustness/noisy-expert reference |

Submitted jobs:

| GPU | Job ID | Manifest |
|---|---:|---|
| H100 | `5271830` | `g1_policy_baseline_return_20260507_seed0_h100.csv` |
| H200 | `5271831` | `g1_policy_baseline_return_20260507_seed1_h200.csv` |
| L40S | `5271832` | `g1_policy_baseline_return_20260507_seed2_l40s.csv` |

These jobs use the native collection runner rather than the PWM policy-extraction runner. They write raw episodes plus metadata under:

- `scripts/outputs/mjlab_qs/raw/g1_policy_baseline_return_20260507/`

### Capability Improvement Ablation: Longer Policy Extraction

The most immediate failure mode may be insufficient actor-critic extraction budget rather than only world-model quality. The original 2x2 uses `policy_iters = 15000`. I submitted a controlled longer-extraction ablation that changes only policy extraction length:

- same dataset,
- same WM checkpoints,
- same 2x2 combinations,
- same final real-env eval,
- `policy_iters = 50000`,
- `eval_every = 2500`.

Stage:

- `offline_pwm_2x2_policy50k_20260507`

W&B project:

- `flow-mbpo-mjlab-offline-pwm-2x2-policy50k`

Submitted jobs:

| GPU | Job ID | Manifest |
|---|---:|---|
| H100 | `5271839` | `offline_pwm_2x2_policy50k_20260507_seed0_h100.csv` |
| H200 | `5271840` | `offline_pwm_2x2_policy50k_20260507_seed1_h200.csv` |
| L40S | `5271841` | `offline_pwm_2x2_policy50k_20260507_seed2_l40s.csv` |

### Critical Interpretation Rules

- If extracted policies are worse than random, the policy-extraction loop is likely broken or exploiting bad WM gradients.
- If extracted policies beat random but are far below the PPO/expert collector, then the WM + imagined actor-critic loop is working only weakly.
- If 50k extraction improves return materially over 15k, then the current result is partly budget-limited.
- If 50k does not improve, the bottleneck is more likely WM gradient quality, actor objective/regularization, online data distribution, or the mismatch between our state-space approximation and original PWM.

### Next Capability Levers to Consider

The most promising levers, in order:

1. Behavior-cloning or warm-start the actor from the collector dataset before imagined PWM extraction. Starting the actor from scratch inside the WM is likely inefficient for this G1 task.
2. Increase online finetune replay quality: keep a cumulative online replay buffer instead of only finetuning on the latest collected windows.
3. Run more online finetune rounds only after the baseline-return stage confirms that extracted policies are better than random.
4. Tune actor-side regularization: action L2 penalty, initial logstd/min logstd, and action-noise/entropy behavior for Flow policy.
5. Separate strict fixed-simulator eval from MJLab-default eval. Current online/eval logs show MJLab default randomization/event terms active, so final claims should explicitly separate these two settings.
6. Compare policy extraction with and without Flow policy under longer budgets. The current 15k 2x2 has visible variance, so one short run is not enough to decide policy-side Flow.

