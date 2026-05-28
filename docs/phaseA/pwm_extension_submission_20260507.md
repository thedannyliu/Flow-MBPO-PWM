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

## Correct PWM/Flow Endpoint Continuation - 2026-05-27

The next run should wait for the refreshed G1 QS dataset:

```text
dataset stage = rerun_a25_native_qs_g1stage4_expertboost_20260527
required file = scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.pt
```

After the data audit and H16 window build pass, train the fixed-data WM stage
with W&B enabled:

```bash
python scripts/experiments/mjlab_qs/build_phaseA_train_manifest_from_windows.py \
  --stage rerun_a25_native_qs_g1stage4_expertboost_20260527 \
  --output scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_wm_20260527.csv \
  --tasks velocity_flat_unitree_g1 \
  --methods mlp_ref,flow_endpoint \
  --seeds 0,1,2 \
  --train-iters 50000 \
  --eval-every 2500 \
  --wandb-project flow-mbpo-mjlab-pwm-flow-endpoint-20260527
```

Then submit with `scripts/experiments/mjlab_qs/submit_array.sh --kind train`
using PACE Phoenix `embers`. Once WM checkpoints exist, build the 2x2 policy
manifest:

Submitted WM job:

```text
job = 9193988
manifest = scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_wm_20260527.csv
rows = 6
methods = mlp_ref, flow_endpoint
seeds = 0,1,2
partition = gpu-h100
qos = embers
wandb_project = flow-mbpo-mjlab-pwm-flow-endpoint-20260527
```

WM result:

```text
job = 9193988
state = COMPLETED
qos = embers
mlp_ref test H16 = seed0 0.02444, seed1 0.02612, seed2 0.02487
flow_endpoint test H16 = seed0 0.02502, seed1 0.03381, seed2 0.02469
```

The Flow endpoint fit is mixed on this refreshed dataset: seed2 slightly
improves over MLP, seed0 is close, and seed1 is worse despite a similar best
validation H16. Policy extraction should therefore be interpreted seed-wise,
not only by method average.

```bash
python scripts/experiments/mjlab_qs/build_policy_extraction_manifest_from_wm.py \
  --stage rerun_g1_pwm_flow_policy2x2_20260527 \
  --wm-stage rerun_a25_native_qs_g1stage4_expertboost_20260527 \
  --dataset-stage rerun_a25_native_qs_g1stage4_expertboost_20260527 \
  --output scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_policy2x2_20260527.csv \
  --wm-methods mlp_ref,flow_endpoint \
  --policy-types mlp,flow \
  --seeds 0,1,2 \
  --policy-iters 50000 \
  --eval-every 2500 \
  --wandb-project flow-mbpo-mjlab-pwm-flow-policy2x2-20260527
```

This gives the intended comparison:

- `mlp_ref + mlp policy`: PWM-style baseline.
- `mlp_ref + flow policy`: policy-architecture swap only.
- `flow_endpoint + mlp policy`: model-architecture swap only.
- `flow_endpoint + flow policy`: combined Flow WM and Flow policy.

All rows keep W&B enabled unless `disable_wandb=true` is explicitly written in
the manifest, which should not be used for formal runs.

Submitted 2x2 policy extraction:

```text
job = 9194509
manifest = scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_policy2x2_20260527.csv
rows = 12
wm_methods = mlp_ref, flow_endpoint
policy_types = mlp, flow
seeds = 0,1,2
qos = embers
time_limit = 08:00:00
wandb_project = flow-mbpo-mjlab-pwm-flow-policy2x2-20260527
```

An initial 16h submission was rejected by the `embers` QOS wall-time limit, so
the formal submission keeps `embers` and uses 8h instead of switching QOS.

Initial running rows:

```text
9194509_0 = mlp_ref + mlp policy, seed0, wandb run 8d805foy
9194509_1 = mlp_ref + mlp policy, seed1, wandb run ftedxbby
status = both reached policy iter 2500 with W&B metrics logging
```

### SIGReg Flow Endpoint Ablation

LeWorldModel (arXiv:2603.19312) uses a next-embedding prediction objective plus
SIGReg, a sketched isotropic Gaussian regularizer over latent embeddings. For
the current state-space MJLab runner, there is no learned pixel encoder; the
first compatible ablation therefore applies the SIGReg-style Epps-Pulley
normality statistic to predicted rollout states from the Flow endpoint WM.
This keeps the existing reward and H16 rollout losses unchanged and adds:

```text
loss = base_wm_loss + sigreg_weight * sigreg(predicted_rollout_states)
sigreg_weight = 0.05
projections = 128
knots = 8
bandwidth = 1.0
```

Submitted job:

```text
job = 9194028
manifest = scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_wm_sigreg_20260527.csv
rows = 3
method = flow_endpoint
seeds = 0,1,2
partition = gpu-h100
qos = embers
wandb_project = flow-mbpo-mjlab-pwm-flow-sigreg-20260527
```

Interpret this as an architecture/loss ablation against the baseline
`flow_endpoint` rows from job `9193988`, not as a replacement for the baseline
2x2 PWM comparison.

Initial SIGReg result:

```text
9194028_0 = flow_endpoint + SIGReg, seed0, COMPLETED
wandb run = brp8tz4z
test H16 = 0.02476
baseline flow_endpoint seed0 test H16 = 0.02502
baseline mlp_ref seed0 test H16 = 0.02444
```

Seed0 suggests SIGReg may slightly improve Flow endpoint rollout fit over the
non-SIGReg Flow endpoint row, but it does not exceed the MLP seed0 baseline.
Wait for seeds 1 and 2 before drawing a method-level conclusion.

### QOS Audit

On 2026-05-27, active and recent MJLab GPU jobs were audited with `squeue` and
`sacct`. The current WM jobs use PACE Phoenix `embers` only:

```text
baseline WM = 9193988, qos = embers, account = gts-agarg35
SIGReg WM = 9194028, qos = embers, account = gts-agarg35
collection retry = 9193797, qos = embers, account = gts-agarg35
```

No GPU job since 2026-05-01 was found with a non-`embers` QOS in the checked
Slurm history. Continue to require explicit user approval before any `inferno`
submission.

### Live Monitor And QOS Guard - 2026-05-27

Follow-up audit confirmed that the active GPU jobs are still using `embers`:

```text
9194028_1 = SIGReg WM, flow_endpoint seed1, RUNNING, qos = embers
9194028_2 = SIGReg WM, flow_endpoint seed2, PENDING, qos = embers
9194509_0 = mlp_ref + mlp policy, seed0, RUNNING, qos = embers
9194509_1 = mlp_ref + mlp policy, seed1, RUNNING, qos = embers
9194509_2 = mlp_ref + mlp policy, seed2, RUNNING, qos = embers
9194509_3-11 = remaining 2x2 policy rows, PENDING, qos = embers
```

`sacct -S 2026-05-01` was rechecked for GPU partitions and did not show any
non-`embers` GPU job in the inspected history.

W&B status:

```text
SIGReg seed1 W&B run = fck40uxq
policy row0 W&B run = 8d805foy
policy row1 W&B run = ftedxbby
policy row2 W&B run = i82c7gys
```

At this check, policy extraction had not yet written final
`summary.json`/`eval_summary.json` files. The latest stdout progress was:

```text
9194509_0 reached iter 15000
9194509_1 reached iter 12500
9194509_2 reached iter 5000
```

To prevent accidental charged submissions, the MJLab submitter now rejects
`--qos inferno` unless `ALLOW_INFERNO_QOS=1` is set after explicit user
approval. The single-task online GPU submitters now default to `embers` instead
of the cluster default and use the same `inferno` guard.

Added a read-only live status helper for this rerun:

```bash
python scripts/experiments/mjlab_qs/summarize_pwm_flow_rerun_status.py \
  --wm-manifest scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_wm_sigreg_20260527.csv \
  --wm-job 9194028 \
  --policy-manifest scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_policy2x2_20260527.csv \
  --policy-job 9194509
```

Current helper output summary:

```text
wm row0 flow_endpoint seed0 = done, test H16 0.024757, wandb brp8tz4z
wm row1 flow_endpoint seed1 = partial, best.pt present, wandb fck40uxq
wm row2 flow_endpoint seed2 = missing/pending
policy row0 mlp_ref + mlp seed0 = partial, latest iter 15000, wandb 8d805foy
policy row1 mlp_ref + mlp seed1 = partial, latest iter 15000, wandb ftedxbby
policy row2 mlp_ref + mlp seed2 = partial, latest iter 5000, wandb i82c7gys
policy rows3-11 = missing/pending
```

The helper was extended to read `sacct` when Slurm job IDs are supplied. It now
reports `slurm_state` and `qos` per manifest row, including compact array
ranges such as `9194509_[3-11%3]`. This makes pending rows explicit instead of
labeling them as missing and keeps the `embers` audit visible in the same CSV.

Latest live status after the extension:

```text
SIGReg seed0 = done, COMPLETED, qos embers
SIGReg seed1 = running, RUNNING, qos embers
SIGReg seed2 = pending, PENDING, qos embers
policy rows0-2 = running, RUNNING, qos embers
policy rows3-11 = pending, PENDING, qos embers
```

Follow-up monitor:

```text
active GPU jobs = all qos embers
policy row0 = running, latest iter 17500, imagined_return 2909.669, wandb 8d805foy
policy row1 = running, latest iter 17500, imagined_return 3206.729, wandb ftedxbby
policy row2 = running, latest iter 7500, imagined_return 1293.210, wandb i82c7gys
SIGReg seed1 = running, wandb fck40uxq, best.pt present
SIGReg seed2 = pending
```

The scanned stderr logs for the running SIGReg and policy rows showed W&B
initialization and no traceback, CUDA OOM, or NaN/Inf error lines at this check.

Added a final result exporter for the 2x2 policy stage:

```bash
python scripts/experiments/mjlab_qs/export_policy_2x2_results.py \
  --manifest scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_policy2x2_20260527.csv \
  --rows-output scripts/outputs/mjlab_qs/status/rerun_g1_pwm_flow_policy2x2_rows_latest.csv \
  --aggregate-output scripts/outputs/mjlab_qs/status/rerun_g1_pwm_flow_policy2x2_aggregate_latest.csv \
  --require-complete
```

The exporter writes per-row real-eval metrics and aggregate mean/std by
`wm_method x policy_type`. The `--require-complete` guard intentionally fails
until all 12 rows have `summary.json`, `eval_summary.json`, and
`final_policy_extraction.pt`, so incomplete runs cannot be accidentally reported
as final 2x2 results.

Added a matching WM ablation exporter for baseline-vs-SIGReg comparison:

```bash
python scripts/experiments/mjlab_qs/export_wm_ablation_results.py \
  --manifest scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_wm_20260527.csv \
  --manifest scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_wm_sigreg_20260527.csv \
  --rows-output scripts/outputs/mjlab_qs/status/rerun_g1_pwm_flow_wm_rows_latest.csv \
  --aggregate-output scripts/outputs/mjlab_qs/status/rerun_g1_pwm_flow_wm_aggregate_latest.csv \
  --require-complete
```

Current WM aggregate snapshot without `--require-complete`:

```text
mlp_ref = 3/3 complete, test H16 mean 0.025142
flow_endpoint = 3/3 complete, test H16 mean 0.027840
flow_endpoint_sigreg0.05 = 1/3 complete, test H16 0.024757 for seed0 only
```

The guarded mode currently fails as expected because SIGReg seeds 1 and 2 are
not complete.

The live status helper now includes manifest-level W&B and progress fields:

```text
wandb_project
disable_wandb
expected_iters
progress_fraction
```

This makes W&B coverage auditable before pending rows start. Current formal rows
all report a non-empty W&B project and `disable_wandb=false`. The current policy
progress snapshot is:

```text
policy row0 = 20000 / 50000 = 0.40
policy row1 = 20000 / 50000 = 0.40
policy row2 = 10000 / 50000 = 0.20
```

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed1 = running, qos embers, wandb fck40uxq, best.pt present
SIGReg seed2 = pending, qos embers
policy row0 = running, 20000 / 50000 = 0.40, imagined_return 2934.196, wandb 8d805foy
policy row1 = running, 20000 / 50000 = 0.40, imagined_return 2953.052, wandb ftedxbby
policy row2 = running, 10000 / 50000 = 0.20, imagined_return 1404.533, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

The running policy rows still have no traceback, CUDA OOM, or NaN/Inf lines in
stderr. No row needed resubmission at this check.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed1 = running, best.pt updated at 2026-05-27 06:03:56 -0400, wandb fck40uxq
SIGReg seed2 = pending, qos embers
policy row0 = running, 22500 / 50000 = 0.45, imagined_return 2977.739, wandb 8d805foy
policy row1 = running, 20000 / 50000 = 0.40, imagined_return 2953.052, wandb ftedxbby
policy row2 = running, 12500 / 50000 = 0.25, imagined_return 1494.117, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

The running policy stderr scans remain clean for traceback, CUDA OOM, and
NaN/Inf lines. No row needed resubmission at this check.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed1 = running, checkpoint best iter 32500, val H16 0.027256, wandb fck40uxq
SIGReg seed2 = pending, qos embers
policy row0 = running, 22500 / 50000 = 0.45, imagined_return 2977.739, wandb 8d805foy
policy row1 = running, 20000 / 50000 = 0.40, imagined_return 2953.052, wandb ftedxbby
policy row2 = running, 12500 / 50000 = 0.25, imagined_return 1494.117, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

SIGReg seed1's current checkpoint is an in-flight validation metric, not a final
test result. Wait for `summary.json` before comparing against baseline Flow and
MLP seeds.

The live status helper now supports:

```text
--load-wm-checkpoints
```

When enabled, it reads `best.pt` for WM rows and reports `latest_iter`,
`progress_fraction`, and `val_h16` for in-flight WM jobs. Current checkpoint
status:

```text
SIGReg seed1 = running, best iter 40000 / 50000 = 0.80, val H16 0.026124, wandb fck40uxq
```

This is still an in-flight validation metric, not a final test result.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed1 = running, best iter 42500 / 50000 = 0.85, val H16 0.024473, wandb fck40uxq
SIGReg seed2 = pending, qos embers
policy row0 = running, 22500 / 50000 = 0.45, imagined_return 2977.739, wandb 8d805foy
policy row1 = running, 22500 / 50000 = 0.45, imagined_return 3003.335, wandb ftedxbby
policy row2 = running, 12500 / 50000 = 0.25, imagined_return 1494.117, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

SIGReg seed1 improved its in-flight validation H16 from 0.026124 at iter 40000
to 0.024473 at iter 42500. Wait for final `summary.json` before making a
test-set comparison.

Added a row-filter helper for surgical resubmission if an array element fails
or times out:

```bash
python scripts/experiments/mjlab_qs/filter_manifest_rows.py \
  --manifest scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_policy2x2_20260527.csv \
  --status-csv scripts/outputs/mjlab_qs/status/rerun_g1_pwm_flow_live_status_latest.csv \
  --statuses failed,completed_missing \
  --output scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_policy2x2_resubmit_failed.csv
```

Manual row ranges are also supported with `--rows 0,2,5-7`. At this check there
are no failed or `completed_missing` rows, so the helper correctly refuses to
write an empty resubmission manifest.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed1 = running, best iter 47500 / 50000 = 0.95, val H16 0.024323, wandb fck40uxq
SIGReg seed2 = pending, qos embers
policy row0 = running, 27500 / 50000 = 0.55, imagined_return 3131.722, wandb 8d805foy
policy row1 = running, 25000 / 50000 = 0.50, imagined_return 3069.477, wandb ftedxbby
policy row2 = running, 17500 / 50000 = 0.35, imagined_return 1653.797, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

No final `summary.json` has been written for SIGReg seed1 or the policy rows at
this check, so these remain in-flight metrics only.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed0 = complete, test H16 0.024757, best val H16 0.023479, wandb brp8tz4z
SIGReg seed1 = complete, test H16 0.023867, best val H16 0.023111, wandb fck40uxq
SIGReg seed2 = running, checkpoint present, wandb sjk9m3t7
policy row0 = running, 27500 / 50000 = 0.55, imagined_return 3131.722, wandb 8d805foy
policy row1 = running, 27500 / 50000 = 0.55, imagined_return 3163.316, wandb ftedxbby
policy row2 = running, 17500 / 50000 = 0.35, imagined_return 1653.797, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

The SIGReg seed1 final test result is now available. Against the same seed,
SIGReg Flow endpoint improves over the baseline Flow endpoint WM
(`0.023867` vs `0.033806` test H16) and the MLP reference WM (`0.023867` vs
`0.026118` test H16). The current partial SIGReg aggregate is 2/3 complete:

```text
flow_endpoint_sigreg0.05 = 2/3 complete, test H16 mean 0.024312
flow_endpoint = 3/3 complete, test H16 mean 0.027840
mlp_ref = 3/3 complete, test H16 mean 0.025142
```

Do not report the SIGReg ablation as final until seed2 writes `summary.json`.

The live status helper was updated to merge `squeue` with `sacct` state. This
fixes pending array ranges that appear in `squeue` as `9194509_[3-11]` but in
`sacct` as `9194509_[4-11%3]`, preventing policy row3 from being incorrectly
marked `missing` while it is still pending.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed2 = running, 2500 / 50000 = 0.05, val H16 0.117984, wandb sjk9m3t7
policy row0 = running, 30000 / 50000 = 0.60, imagined_return 2827.888, wandb 8d805foy
policy row1 = running, 27500 / 50000 = 0.55, imagined_return 3163.316, wandb ftedxbby
policy row2 = running, 20000 / 50000 = 0.40, imagined_return 1713.699, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

The running stderr scans remain clean for traceback, CUDA OOM, RuntimeError,
Exception, and NaN/Inf matches. The W&B manifest audit still reports no missing
project and no `disable_wandb=true` rows. Policy `--require-complete` still
fails as expected because no policy row has final `summary.json`,
`eval_summary.json`, and `final_policy_extraction.pt` yet.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed2 = running, 2500 / 50000 = 0.05, val H16 0.117984, wandb sjk9m3t7
policy row0 = running, 30000 / 50000 = 0.60, imagined_return 2827.888, wandb 8d805foy
policy row1 = running, 30000 / 50000 = 0.60, imagined_return 3255.210, wandb ftedxbby
policy row2 = running, 20000 / 50000 = 0.40, imagined_return 1713.699, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

The WM and policy `--require-complete` guards both still fail for expected
incomplete rows: SIGReg seed2 is missing its final summary, and all 12 policy
rows are missing final policy artifacts. No resubmission is needed at this
check.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed2 = running, 5000 / 50000 = 0.10, val H16 0.069703, wandb sjk9m3t7
policy row0 = running, 30000 / 50000 = 0.60, imagined_return 2827.888, wandb 8d805foy
policy row1 = running, 30000 / 50000 = 0.60, imagined_return 3255.210, wandb ftedxbby
policy row2 = running, 20000 / 50000 = 0.40, imagined_return 1713.699, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

SIGReg seed2's in-flight validation H16 improved from `0.117984` at 2500 iters
to `0.069703` at 5000 iters. This is still checkpoint validation only, not a
final test result. Stderr scans remain clean, W&B manifest audit remains clean,
and both completion guards still fail only for expected incomplete artifacts.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed2 = running, 7500 / 50000 = 0.15, val H16 0.054029, wandb sjk9m3t7
policy row0 = running, 32500 / 50000 = 0.65, imagined_return 3005.745, wandb 8d805foy
policy row1 = running, 30000 / 50000 = 0.60, imagined_return 3255.210, wandb ftedxbby
policy row2 = running, 22500 / 50000 = 0.45, imagined_return 1760.466, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

SIGReg seed2 continues to improve its in-flight validation H16 (`0.054029` at
7500 iters). Policy rows 0 and 2 also advanced. No GPU job is outside `embers`,
stderr scans remain clean, and the completion guards still fail only because
expected final artifacts are not ready.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed2 = running, 7500 / 50000 = 0.15, val H16 0.054029, wandb sjk9m3t7
policy row0 = running, 32500 / 50000 = 0.65, imagined_return 3005.745, wandb 8d805foy
policy row1 = running, 32500 / 50000 = 0.65, imagined_return 2909.064, wandb ftedxbby
policy row2 = running, 22500 / 50000 = 0.45, imagined_return 1760.466, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

Policy row1 has advanced to 32500 iters. No new final WM or policy artifact is
ready yet; WM `--require-complete` is still blocked only by SIGReg seed2, and
policy `--require-complete` is still blocked only by expected missing final
policy artifacts.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed2 = running, 10000 / 50000 = 0.20, val H16 0.045658, wandb sjk9m3t7
policy row0 = running, 32500 / 50000 = 0.65, imagined_return 3005.745, wandb 8d805foy
policy row1 = running, 32500 / 50000 = 0.65, imagined_return 2909.064, wandb ftedxbby
policy row2 = running, 22500 / 50000 = 0.45, imagined_return 1760.466, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

SIGReg seed2 improved again on in-flight validation, from `0.054029` at 7500
iters to `0.045658` at 10000 iters. Policy rows did not write a newer checkpoint
at this check. Stderr scans remain clean, W&B manifest audit remains clean, and
the completion guards still fail only for expected incomplete final artifacts.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed2 = running, 12500 / 50000 = 0.25, val H16 0.044124, wandb sjk9m3t7
policy row0 = running, 35000 / 50000 = 0.70, imagined_return 3079.021, wandb 8d805foy
policy row1 = running, 32500 / 50000 = 0.65, imagined_return 2909.064, wandb ftedxbby
policy row2 = running, 25000 / 50000 = 0.50, imagined_return 1771.889, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

After a short follow-up poll, SIGReg seed2 advanced to 12500 iters and policy
rows 0 and 2 wrote newer checkpoints. No GPU job is outside `embers`, stderr
scans remain clean, W&B manifest audit remains clean, and completion guards
still fail only for expected incomplete final artifacts.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed2 = running, 12500 / 50000 = 0.25, val H16 0.044124, wandb sjk9m3t7
policy row0 = running, 35000 / 50000 = 0.70, imagined_return 3079.021, wandb 8d805foy
policy row1 = running, 35000 / 50000 = 0.70, imagined_return 3009.114, wandb ftedxbby
policy row2 = running, 25000 / 50000 = 0.50, imagined_return 1771.889, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

Policy row1 caught up to 35000 iters on the follow-up status refresh. SIGReg
seed2 remains in-flight at 12500 iters; no final WM or policy artifact is ready
yet. QOS, W&B, stderr, and completion-guard checks remain clean.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed2 = running, 15000 / 50000 = 0.30, val H16 0.040792, wandb sjk9m3t7
policy row0 = running, 35000 / 50000 = 0.70, imagined_return 3079.021, wandb 8d805foy
policy row1 = running, 35000 / 50000 = 0.70, imagined_return 3009.114, wandb ftedxbby
policy row2 = running, 25000 / 50000 = 0.50, imagined_return 1771.889, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

SIGReg seed2 continues improving in-flight validation (`0.040792` at 15000
iters). Policy rows did not write newer checkpoints at this check. No final
summary or extraction artifact is ready yet, and all QOS/W&B/stderr/guard checks
remain clean.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed2 = running, 17500 / 50000 = 0.35, val H16 0.035879, wandb sjk9m3t7
policy row0 = running, 37500 / 50000 = 0.75, imagined_return 3171.349, wandb 8d805foy
policy row1 = running, 35000 / 50000 = 0.70, imagined_return 3009.114, wandb ftedxbby
policy row2 = running, 27500 / 50000 = 0.55, imagined_return 1754.402, wandb i82c7gys
policy rows3-11 = pending, qos embers, W&B enabled in manifest
```

After a short follow-up poll, SIGReg seed2 improved to `0.035879` validation H16
at 17500 iters. Policy rows 0 and 2 also advanced. No GPU job is outside
`embers`, stderr scans remain clean, W&B manifest audit remains clean, and both
completion guards still fail only for expected incomplete artifacts.

Follow-up monitor:

```text
active GPU jobs = all qos embers
SIGReg seed2 = complete, test H16 0.024699, best val H16 0.023555, wandb sjk9m3t7
policy rows0-2 = complete, mlp_ref + mlp, qos embers, W&B runs 8d805foy/ftedxbby/i82c7gys
policy row5 = complete, mlp_ref + flow seed2, qos embers, W&B 581hbdol
policy rows3,4,6,7 = preempted before final artifacts, qos embers
policy rows8-11 = pending on original job 9194509, qos embers
resubmit rows3,4,6,7 = job 9203172, pending, qos embers
```

SIGReg WM is now 3/3 complete. The aggregate test H16 is:

```text
flow_endpoint_sigreg0.05 = 0.024441 +/- 0.000498
mlp_ref = 0.025142 +/- 0.000872
flow_endpoint = 0.027840 +/- 0.005169
```

The guarded WM exporter now passes. Policy 2x2 is still incomplete: `mlp_ref +
mlp` is 3/3 complete with eval return mean `-4.521467`, `mlp_ref + flow` is
1/3 complete from seed2 with eval return `-2.935666`, and the Flow-WM policy
cells have no final rows yet. Rows 3, 4, 6, and 7 were cancelled by `embers`
preemption before writing final artifacts, so a surgical resubmit manifest was
created:

```text
scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_policy2x2_resubmit_preempted_20260527.csv
```

Submitted resubmission job `9203172` with `--qos embers`, W&B enabled in the
manifest, `gpu-h100`, `--time 8:00:00`, and max concurrent 2. It is currently
pending because H100 nodes are unavailable/reserved for maintenance, not because
of a QOS or manifest issue.

Follow-up monitor:

```text
original H100 pending rows8-11 = cancelled to avoid duplicate writes
H100 resubmit job 9203172 rows3,4,6,7 = cancelled to avoid duplicate writes
new missing-row manifest = rerun_g1_pwm_flow_policy2x2_missing_h200_20260527.csv
new H200 resubmit job = 9203199
job 9203199 qos = embers
job 9203199 state = pending, reason ReqNodeNotAvail / Reserved for maintenance
```

The new H200 manifest contains the eight policy rows still missing final
artifacts: original rows 3, 4, 6, 7, 8, 9, 10, and 11. The manifest keeps W&B
enabled for every row and uses the same output stage, so completed artifacts
will fill the original 2x2 result tree. The old pending H100 arrays were
cancelled before submitting the H200 replacement to prevent H100 and H200 jobs
from writing the same output directories if maintenance clears suddenly. No
`inferno` job was submitted.

Follow-up scheduler hardening:

```text
policy runner guard = per-output .policy_extraction.lock plus completed-artifact skip
H200 replacement job = 9203199, pending, qos embers
A100 fallback job = 9203237, pending, qos embers, array throttle 1
active user GPU jobs = all qos embers
```

Because `9203199` still had no estimated start time, a low-concurrency A100
fallback was submitted with the same missing-row manifest. The runner now holds
a lock in each policy output directory and skips rows that already have
`summary.json`, `eval_summary.json`, and `final_policy_extraction.pt`, so the
H200 and A100 arrays can safely race for resources without overwriting completed
policy artifacts. The fallback keeps W&B enabled through the existing manifest
and does not use `inferno`.

Monitor tooling update:

```bash
python scripts/experiments/mjlab_qs/summarize_pwm_flow_rerun_status.py \
  --policy-manifest scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_policy2x2_missing_h200_20260527.csv \
  --policy-job 9203199 --policy-job 9203237 \
  --output scripts/outputs/mjlab_qs/status/rerun_g1_pwm_flow_policy2x2_replacement_combined_status_latest.csv
```

The status exporter now accepts repeated or comma-separated `--policy-job`
values and writes a `slurm_job` column, so the H200 replacement and A100 fallback
can be monitored as one missing-row pool. The latest combined snapshot still
shows all eight rows pending, W&B enabled in the manifest, and QOS `embers`.

Environment-fix resubmission:

```text
failed H200 replacement job = 9203199, all rows FAILED, qos embers
failed A100 fallback job = 9203237, all rows FAILED, qos embers
failure cause = base Python 3.13 did not have wandb installed
verified env = /storage/home/hcoda1/9/eliu354/r-agarg35-0/envs/pwm/bin/python
verified packages = wandb 0.23.0, torch 2.10.0+cu128
new A100 fallback job = 9210886, pending, qos embers
new H200 replacement job = 9210887, pending, qos embers
```

The failed jobs did allocate GPU nodes briefly, so the failure was not caused by
the scheduler or the machine-room maintenance notice. They failed immediately at
`import wandb` because the submission used base `python`. The corrected
submissions use the `pwm` environment's Python directly while keeping the same
missing-row manifest, W&B-enabled rows, output locks, and `embers` QOS.

Additional `embers` fallback:

```text
A100 env-fixed job = 9210886, pending, qos embers, reason Resources
H200 env-fixed job = 9210887, pending, qos embers, reason Priority
RTX6000 fallback job = 9210910, pending, qos embers, reason Priority
RTX6000 resources = gpu:rtx_6000:1, cpus-per-task 6, mem 128G, time 8h
```

The first RTX6000 submission attempt used 8 CPU cores and was rejected by Slurm's
6:1 CPU:GPU limit for that node class before entering the queue. The accepted
fallback uses 6 CPU cores and the same `pwm` Python, W&B-enabled manifest, and
output-directory locking. The combined monitor now tracks `9210886`, `9210887`,
and `9210910` together.

RTX6000 start:

```text
running row = missing-row manifest row0, original policy row3
cell = mlp_ref world model + flow policy, seed0
job = 9210910_0, qos embers, node atl1-1-02-004-35-0
wandb run = 7dyj672f
first logged iter = 1 / 50000
```

The row started under the corrected `pwm` environment and wrote both the Slurm
prolog and the first JSON training metric. W&B is syncing to
`flow-mbpo-mjlab-pwm-flow-policy2x2-20260527`. Rows 1-7 remain pending behind
the RTX6000 array throttle, while the A100 and H200 fallback arrays remain
pending under `embers`.

Duplicate-start lock behavior:

```text
active row holder = 9210910_0, RTX6000, row0
duplicate start = 9210887_0, H200, row0
result = skipped immediately because the row lock was already held
walltime = 2 seconds, qos embers
```

The policy row runner now uses a non-blocking output-directory lock. If another
fallback array starts a row that is already running, it exits immediately instead
of waiting on the lock while holding a GPU allocation. Completed rows still skip
through the existing final-artifact check.

H200 starts:

```text
missing-row row1 = original policy row4, mlp_ref + flow, seed1
job = 9210887_1, qos embers, wandb run dtwoo1db, first logged iter 1 / 50000
missing-row row2 = original policy row6, flow_endpoint + mlp, seed0
job = 9210887_2, qos embers, wandb run a8narre6, first logged iter 1 / 50000
```

The H200 fallback array began useful non-duplicate work after row0 skipped on
the held RTX6000 lock. The combined monitor now shows rows 0, 1, and 2 running,
all with W&B run IDs and QOS `embers`.

In-flight policy progress:

```text
row0 / original row3 = mlp_ref + flow, seed0, RTX6000, wandb 7dyj672f, iter 15000 / 50000
row1 / original row4 = mlp_ref + flow, seed1, H200, wandb dtwoo1db, iter 25000 / 50000
row2 / original row6 = flow_endpoint + mlp, seed0, H200, wandb a8narre6, iter 27500 / 50000
row3 / original row7 = flow_endpoint + mlp, seed1, A100, wandb 3jn08peh, iter 20000 / 50000
```

The A100 fallback skipped rows 0-2 quickly because those output locks were held
by the RTX6000/H200 jobs, then started useful work on row3. The remaining
missing rows are row4 and rows5-7. The aggregate exporter still reports 4/12
completed policy rows because these four active rows have not written final
artifacts yet.

Policy/QOS status refresh at 2026-05-27 18:57 EDT:

```text
active jobs checked = 9210886, 9210887, 9210910
non-embers GPU jobs = none observed in current squeue
row0 / original row3 = mlp_ref + flow, seed0, RTX6000, qos embers, wandb 7dyj672f, iter 15000 / 50000
row1 / original row4 = mlp_ref + flow, seed1, H200, qos embers, wandb dtwoo1db, iter 30000 / 50000
row2 / original row6 = flow_endpoint + mlp, seed0, H200, qos embers, wandb a8narre6, iter 32500 / 50000
row3 / original row7 = flow_endpoint + mlp, seed1, A100, qos embers, wandb 3jn08peh, iter 22500 / 50000
rows pending behind throttles = missing rows 4-7
policy aggregate = 4 / 12 complete; require-complete still fails with first missing row 3
```

The active replacement/fallback jobs remain under `embers` across H200, A100,
and RTX6000 partitions. No replacement row has finished since the previous
snapshot, so the tracked policy aggregate is unchanged. Continue monitoring
these same arrays before submitting any additional work.

Rollout-video requirement and first submission at 2026-05-27 19:49 EDT:

```text
new requirement = completed policy weights must be checked with readable real-env rollouts, not only scalar eval
renderer = scripts/experiments/mjlab_qs/render_policy_rollout.py
row runner = scripts/experiments/mjlab_qs/run_policy_rollout_row.py
submit kind = policy_rollout
completed-row rollout manifest = scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_policy2x2_done_rollouts_20260527.csv
manifest rows = original policy rows 0, 1, 2, 5, 6
rollout output root = scripts/outputs/mjlab_qs/policy_rollouts/rerun_g1_pwm_flow_policy2x2_20260527/
video artifact = rollout.mp4 plus rollout_summary.csv, rollout_steps.csv, summary.json
W&B job type = policy_rollout_video
Slurm job = 9218121, gpu-a100, qos embers, max-concurrent 1
```

This rollout pass starts with the five policy rows that already have
`final_policy_extraction.pt`: three `mlp_ref + mlp` seeds, `mlp_ref + flow`
seed2, and `flow_endpoint + mlp` seed0. Rows that finish later should be added
to a refreshed rollout manifest and rendered the same way.

Rollout-video completion update at 2026-05-27 19:57 EDT:

```text
initial rollout job = 9218121, qos embers, failed/cancelled after MP4 writer used an invalid imageio v3 plugin name
fix commit = 78224e5, switches MP4 export to imageio.v2 FFMPEG writer
retry rollout job = 9218448, qos embers, completed 5/5 rows
incremental rollout job = 9218749, qos embers, completed newly finished original policy row4
rollout mp4 count = 6
rollout summary count = 6
non-embers GPU jobs observed = none
```

Completed rollout summaries:

```text
mlp_ref + mlp seed0 = return -4.0669, mean length 68.3, 205 frames
mlp_ref + mlp seed1 = return -5.0430, mean length 56.7, 170 frames
mlp_ref + mlp seed2 = return -4.6002, mean length 74.0, 222 frames
mlp_ref + flow seed1 = return -4.3979, mean length 65.3, 196 frames
mlp_ref + flow seed2 = return -2.9713, mean length 62.0, 186 frames
flow_endpoint + mlp seed0 = return -3.6572, mean length 55.0, 165 frames
```

The scalar policy aggregate is now 6/12 complete: `mlp_ref + mlp` is 3/3,
`mlp_ref + flow` is 2/3, `flow_endpoint + mlp` is 1/3, and
`flow_endpoint + flow` is 0/3. In-flight policy rows remain under `embers`:
`mlp_ref + flow` seed0, `flow_endpoint + mlp` seeds1-2, and
`flow_endpoint + flow` seed0; flow-policy seeds1-2 are still pending.

Collector baseline comparison and true-best fix at 2026-05-27 20:19 EDT:

```text
formal G1 expert data source = native RSL-RL PPO conservative collector
stage-probe top source = seed1 iter15000, return 80.28, episode length 996.68
formal audit expert bin = return 94.54, episode length 999.0, fall rate 0.0
formal audit expert_noisy expert bin = return 95.80, episode length 999.0, fall rate 0.0
current extracted policy rollouts = return roughly -5.04 to -2.97, episode length roughly 55 to 74
```

The current extracted PWM policies are therefore not close to the data-collector
quality. The scalar differences among extracted policies should be read as
"less bad" rather than as good locomotion. This makes the current branch a
diagnostic/model-exploitation ablation, not a final claim that offline PWM
extraction beats the source policy.

Bug fix:

```text
old behavior = best_policy_extraction.pt stored the final actor plus best metric payload
new behavior = clone actor/critic state when imagined_return improves and store that true snapshot
new checkpoint marker = is_true_best_snapshot: true
rollout runner = renders final and best variants, but skips legacy non-snapshot best files
```

Rows already completed or already running before this code change cannot recover
their true best actor from the existing artifact. Newly started rows will produce
a real best checkpoint; older rows need a rerun if final-vs-best comparison is
required for the same seed.

BC-only baseline start at 2026-05-27 20:53 EDT:

```text
motivation = verify action semantics, observation split, normalization, env match, and dataset quality before further WM exploitation
collector target scale = expert collector return roughly 80-95, episode length roughly 999
current extracted-policy scale = return roughly -5 to -3, episode length roughly 55-74
manifest = scripts/outputs/mjlab_qs/manifests/rerun_g1_bc_only_mlp10k_20260527.csv
stage = rerun_g1_bc_only_mlp10k_20260527
rows = mlp_ref placeholder WM + MLP policy, seeds 0/1/2
bc_warmstart_iters = 10000
policy_iters = 0
W&B project = flow-mbpo-mjlab-bc-baseline-20260527
```

The BC-only run intentionally does not do imagined-return policy optimization.
It still uses the policy-extraction runner so it shares actor definitions,
normalization, real-env eval, W&B logging, and rollout-video tooling with the
PWM/Flow experiments. The runner now writes `best_imagined_return = null` for
BC-only rows rather than serializing an invalid infinite value.

BC-only baseline completion at 2026-05-27 21:03 EDT:

```text
policy job = 9221898, gpu-a100, qos embers, completed 3/3
rollout job = 9222278, gpu-a100, qos embers, completed BC rollouts 3/3 plus new policy row7 rollout
W&B project = flow-mbpo-mjlab-bc-baseline-20260527
BC seed0 eval = return 4.5857, episode length 108.93
BC seed1 eval = return 3.4789, episode length 93.20
BC seed2 eval = return 3.9408, episode length 98.38
BC seed0 rollout = return 5.4168, episode length 122.0
BC seed1 rollout = return 4.5028, episode length 111.3
BC seed2 rollout = return 10.9708, episode length 185.7
```

The BC-only baseline is better than the failed frozen-WM extracted policies in
episode length, but it is still far below the expert collector scale
(`return ~= 80-95`, `episode length ~= 999`). This suggests the dataset/action
interface is not completely broken, but plain 10k-step BC is not enough to
recover collector-quality locomotion. Future BC/IL work should test stronger
training, expert-only sampling, longer training, and policy architecture before
using BC as the starting point for PWM-style WM improvement.

Policy 2x2 update at the same checkpoint:

```text
scalar aggregate = 7 / 12 complete
mlp_ref + mlp = 3/3, return mean -4.52
mlp_ref + flow = 2/3, return mean -3.59
flow_endpoint + mlp = 2/3, return mean -4.70
flow_endpoint + flow = 0/3
new rollout added = flow_endpoint + mlp seed1, return -5.8816, episode length 63.3
```

Expert-filtered BC protocol update at 2026-05-27 21:12 EDT:

```text
motivation = test whether high-quality QS windows alone can recover collector-like behavior before further PWM exploitation
code change = policy extraction now supports separate BC and PWM sampling quality filters
new CLI = --bc-quality-filter and --policy-quality-filter
manifest builder fields = bc_quality_filter, policy_quality_filter, optional compute_profile
expert-filtered manifest = scripts/outputs/mjlab_qs/manifests/rerun_g1_bc_expert_mlp50k_20260528.csv
stage = rerun_g1_bc_expert_mlp50k_20260528
rows = MLP policy, seeds 0/1/2
bc_warmstart_iters = 50000
policy_iters = 0
bc_quality_filter = expert,expert_noisy
W&B project = flow-mbpo-mjlab-bc-expert-20260528
Slurm job = 9223151, gpu-a100, qos embers, max-concurrent 1
```

The previous 10k BC baseline used all train windows and reached only
single-digit returns. This new run is a stricter dataset/action-semantics test:
if expert-only BC remains far below the collector, the next debugging target is
BC capacity/training protocol or observation/action alignment rather than
world-model policy improvement. If it improves substantially, use the same
filter as the BC warm start for conservative PWM/Flow ablations.

Expert-filtered BC completion update at 2026-05-27 21:26 EDT:

```text
policy job = 9223151, gpu-a100, qos embers, completed 3/3
W&B runs = seed0 lb2x51ov, seed1 6aqpoqs5, seed2 znsudeac
train windows = 282887
BC train windows = 250559
BC quality counts = expert 200178, expert_noisy 50381
seed0 eval = return 46.1703, episode length 604.58
seed1 eval = return 36.6089, episode length 490.75
seed2 eval = return 46.6763, episode length 613.15
aggregate eval = return mean 43.1518, return std 4.6312
aggregate length = mean 569.49, std 55.79
rollout job = 9223566, gpu-a100, qos embers, final checkpoints only
```

Expert-filtered BC is a major improvement over the all-window 10k BC baseline,
but it still does not recover collector-quality behavior. The current evidence
supports using expert-filtered BC as the next warm-start baseline, while also
treating BC capacity/training schedule as an unresolved issue before making
claims about PWM/Flow improvement.

Expert-filtered BC rollout completion at 2026-05-27 21:29 EDT:

```text
rollout job = 9223566, gpu-a100, qos embers, completed 3/3
rollout W&B runs = seed0 jebhwezc, seed1 ucbm020h, seed2 xfyb30lu
rollout mp4 count = 3
seed0 rollout = return 28.2471, episode length 300.0, frames 900
seed1 rollout = return 14.3453, episode length 207.0, frames 621
seed2 rollout = return 14.6558, episode length 207.67, frames 623
aggregate rollout = return mean 19.0827, return std 6.4814
aggregate rollout length = mean 238.22, std 43.68
```

The rollout videos confirm the same qualitative conclusion as the scalar evals:
expert-filtered BC is much better than the failed frozen-WM extracted policies,
but it is still unstable and not close to the collector baseline. Treat this as
the minimum BC warm-start baseline, not as a solved imitation policy.

Conservative BC-warmstart PWM start at 2026-05-27 21:36 EDT:

```text
code change = PWM actor optimization supports --policy-bc-reg
regularizer = MSE between actor actions and dataset policy_action on sampled real windows
manifest = scripts/outputs/mjlab_qs/manifests/rerun_g1_bcwarm_pwm_bcreg1_mlpwm_vs_flowwm_seed0_20260528.csv
stage = rerun_g1_bcwarm_pwm_bcreg1_mlpwm_vs_flowwm_seed0_20260528
rows = mlp_ref + mlp seed0, flow_endpoint + mlp seed0
bc_warmstart_iters = 50000
policy_iters = 10000
policy_bc_reg = 1.0
bc_quality_filter = expert,expert_noisy
policy_quality_filter = expert,expert_noisy
W&B project = flow-mbpo-mjlab-bcwarm-pwm-bcreg-20260528
Slurm job = 9223782, gpu-a100, qos embers, max-concurrent 1
```

This is the first PWM-style policy-improvement run that starts from the stronger
expert-filtered BC protocol and keeps the actor tied to real dataset actions
during imagined-return optimization. It is intentionally limited to seed0 and
MLP policy while comparing MLP vs flow-endpoint world models; expand to more
seeds and flow policies only after rollout videos show non-degenerate behavior.

Warmstart regularizer verification at 2026-05-27 21:38 EDT:

```text
row0 W&B run = 8nb4nx3w
row0 first PWM update = iter 1
train/policy_bc_reg_loss = 0.0012539
train/policy_bc_reg_weight = 1.0
train/imagined_return = -35.9184
```

This confirms that the conservative BC term is active in the actor update after
the 50k-step expert-filtered BC warm start. Completion still requires real-env
eval and rollout videos for both rows.

Conservative BC-warmstart PWM completion at 2026-05-28 02:21 EDT:

```text
policy job = 9223782, gpu-a100, qos embers, completed 2/2
mlp_ref + mlp seed0 W&B = 8nb4nx3w
flow_endpoint + mlp seed0 W&B = oel3gq67
mlp_ref + mlp eval = return -4.0412, episode length 60.78
flow_endpoint + mlp eval = return -2.3336, episode length 44.47
mlp_ref best imagined return = 2270.6699 at iter 8000
flow_endpoint best imagined return = 2018.7007 at iter 7000
rollout job = 9233777, gpu-a100, qos embers, pending
```

This run did not preserve the expert-filtered BC behavior after 10k imagined
policy updates, despite the BC regularizer being active. The result is another
model-exploitation failure: imagined return increases, but real-env return
collapses back near the failed frozen-WM extraction regime. The next conservative
PWM attempt should reduce policy-update strength substantially before expanding
seeds: shorter policy optimization, larger BC/action-deviation weight, lower
actor LR, or explicit early stopping on real-env eval.

2x2 policy refresh at 2026-05-28 02:21 EDT:

```text
flow_endpoint + mlp = 3/3, eval return mean -4.5951, episode length mean 58.94
flow_endpoint + flow = 2/3, eval return mean -3.9597, episode length mean 65.94
mlp_ref + flow = 2/3, eval return mean -3.5919, episode length mean 61.39
mlp_ref + mlp = 3/3, eval return mean -4.5215, episode length mean 63.62
new rollout job = 9233776, gpu-a100, qos embers, pending
new rollout rows = mlp_ref+flow seed0, flow_endpoint+mlp seed2, flow_endpoint+flow seed0, flow_endpoint+flow seed1
remaining 2x2 training row = flow_endpoint+flow seed2, job 9210910_7, qos embers
```

The scalar 2x2 refresh still shows all learned policies far below the collector
and expert-filtered BC baselines. The new rollout job is required before making
any qualitative comparison among these variants.

Best-checkpoint rollout status at 2026-05-28 02:28 EDT:

```text
true-best snapshots currently available = flow_endpoint+flow seed1, conservative mlp_ref+mlp seed0, conservative flow_endpoint+mlp seed0
legacy best checkpoints = most older 2x2 rows; these are not valid actor snapshots and are skipped by the rollout runner
A100 rollout jobs = 9233776 for new 2x2 rows, 9233777 for conservative BC-warmstart rows, both qos embers and pending
L40S fallback rollout jobs = 9234013 for new 2x2 rows, 9234014 for conservative BC-warmstart rows, both qos embers and pending
```

Until one of these rollout arrays completes, "best" only means best imagined
return inside the world model. It does not prove better real-environment
behavior. Current final-checkpoint evidence still says the PWM variants collapse
far below expert-filtered BC and collector behavior.

Best-vs-final rollout evidence at 2026-05-28 02:49 EDT:

```text
conservative rollout job = 9234137, gpu-rtx6000, qos embers, completed 2/2
2x2 rollout job = 9234138, gpu-rtx6000, qos embers, completed 3/4, row0 failed
2x2 row0 failure = MuJoCo Warp CUDA module load failure on RTX6000
pending fallback rollouts = 9233776 gpu-a100 embers, 9234013 gpu-l40s embers
redundant conservative fallback rollouts still pending = 9233777 gpu-a100 embers, 9234014 gpu-l40s embers
remaining 2x2 training row = flow_endpoint+flow seed2, job 9210910_7, gpu-rtx6000, qos embers, running
```

Completed true-best rollout comparisons:

| Run | Final rollout return | Final length | Best rollout return | Best length | Interpretation |
|---|---:|---:|---:|---:|---|
| conservative `mlp_ref+mlp` seed0 | -3.7989 | 67.67 | -2.1859 | 56.00 | best improves return but remains collapsed |
| conservative `flow_endpoint+mlp` seed0 | -2.0479 | 47.00 | -3.0904 | 68.67 | best is worse by return |
| 2x2 `flow_endpoint+flow` seed1 | -3.8629 | 56.33 | -4.3508 | 55.33 | best is worse |

Additional completed final rollouts from the 2x2 refresh:

| Run | Final rollout return | Final length | Frames |
|---|---:|---:|---:|
| `flow_endpoint+mlp` seed2 | -4.7771 | 62.67 | 188 |
| `flow_endpoint+flow` seed0 | -3.6777 | 148.67 | 446 |

The best-checkpoint evidence does not change the conclusion. In the few rows
with true best-actor snapshots and rendered videos, selecting the best imagined
return checkpoint is mixed or worse in real MJLab. It does not recover the
expert-filtered BC baseline, and it is nowhere near the collector-quality data
policy. Future PWM attempts should treat best-imagined return as a diagnostic
only and should add stronger real-data anchoring or real-env early stopping
before expanding the 2x2.

Queue cleanup at 2026-05-28 02:52 EDT:

```text
cancelled redundant conservative fallback jobs = 9233777 gpu-a100 embers, 9234014 gpu-l40s embers
kept 2x2 fallback jobs = 9233776 gpu-a100 embers, 9234013 gpu-l40s embers
kept running 2x2 training row = 9210910_7 gpu-rtx6000 embers, flow_endpoint+flow seed2
```

The conservative final/best rollout artifacts already exist from RTX6000 job
9234137, so the cancelled fallback jobs would not add required evidence. The
2x2 fallback jobs are still needed because RTX6000 row0 failed before rendering
`mlp_ref+flow` seed0.

2x2 row0 failure audit at 2026-05-28 02:56 EDT:

```text
row = mlp_ref+flow seed0
primary completed checkpoint path = scripts/outputs/mjlab_qs/policy_extraction/rerun_g1_pwm_flow_policy2x2_20260527/velocity_flat_unitree_g1/mlp_ref/flow/offline/policy50k/seed_0/
checkpoint files = final_policy_extraction.pt, best_policy_extraction.pt
missing files = summary.json, eval_summary.json
training log = logs/slurm/mjlab_qs/policy_extract/mjqs_policy_extract_9210910_0.out
error log = logs/slurm/mjlab_qs/policy_extract/mjqs_policy_extract_9210910_0.err
failure point = after iter 50000 checkpoint save, during real-env eval setup
failure reason = MuJoCo Warp CUDA kernel build/load failure on RTX6000
fallback status = A100 job 9233776 and L40S job 9234013 remain pending, both qos embers
```

Do not rerun the row through `run_policy_extraction_row.py` just to recover the
summary, because that runner does not resume from the saved actor and would
retrain/overwrite the existing checkpoint directory. The correct recovery path
is to render/evaluate the saved checkpoints through the rollout runner on a GPU
class where MuJoCo Warp initializes cleanly.

Row7 rollout preparation at 2026-05-28 02:59 EDT:

```text
row = flow_endpoint+flow seed2
policy job = 9210910_7, gpu-rtx6000, qos embers, still running
latest observed progress = 47500/50000 policy iterations
output directory = scripts/outputs/mjlab_qs/policy_extraction/rerun_g1_pwm_flow_policy2x2_20260527/velocity_flat_unitree_g1/flow_endpoint/flow/offline/policy50k/seed_2/
current files = .policy_extraction.lock only
prepared rollout manifest = scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_policy2x2_flow_endpoint_flow_seed2_rollout_20260528.csv
```

Do not submit the rollout manifest until row7 writes `final_policy_extraction.pt`
and, if available, a true `best_policy_extraction.pt`. Once complete, submit the
prepared rollout manifest with embers, for example:

```bash
bash scripts/experiments/mjlab_qs/submit_array.sh \
  --kind policy_rollout \
  --manifest scripts/outputs/mjlab_qs/manifests/rerun_g1_pwm_flow_policy2x2_flow_endpoint_flow_seed2_rollout_20260528.csv \
  --gpu-type A100 \
  --partition gpu-a100 \
  --qos embers \
  --time 02:00:00 \
  --max-concurrent 1 \
  --python-bin /storage/home/hcoda1/9/eliu354/r-agarg35-0/envs/pwm/bin/python
```

2x2 row0 fallback rollout completion at 2026-05-28 03:02 EDT:

```text
rollout job = 9233776_0, gpu-a100, qos embers, completed
row = mlp_ref+flow seed0
checkpoint kind = final
rollout return = -3.0762
rollout episode length = 73.0
frames = 219
video = scripts/outputs/mjlab_qs/policy_rollouts/rerun_g1_pwm_flow_policy2x2_20260527/velocity_flat_unitree_g1/mlp_ref/flow/offline/policy50k/seed_0/rollout.mp4
W&B run = 0gcy0xu9
best checkpoint status = legacy/non-snapshot, skipped by rollout runner
```

This fills the main evidence gap left by the RTX6000 MuJoCo/Warp eval failure.
The result remains a collapsed policy and is consistent with the broader 2x2
rollout pattern. After row0 completed, the remaining A100 fallback row and the
L40S duplicate fallback were cancelled because their target rollouts were
already complete or redundant:

```text
cancelled = 9233776_3, 9234013
remaining active 2x2 policy job = 9210910_7, gpu-rtx6000, qos embers
```

2x2 row7 completion and rollout at 2026-05-28 03:03 EDT:

```text
policy job = 9210910_7, gpu-rtx6000, qos embers, completed
row = flow_endpoint+flow seed2
W&B policy run = btxwrgsy
final eval = return -3.0830, episode length 48.35
best imagined return = 3891.8916 at iter 45000
best checkpoint = true best-actor snapshot
rollout job = 9234836_0, gpu-a100, qos embers, completed
final rollout W&B = 7cbfvkco
best rollout W&B = p2z4lyxt
final rollout = return -3.0837, episode length 51.0, frames 153
best rollout = return -3.3635, episode length 55.67, frames 167
```

Row7 again shows that best imagined return does not select a better real policy:
the true-best snapshot is worse than the final actor by rollout return.

Updated 2x2 aggregate after row7 completion:

| WM | Policy | Completed | Eval return mean | Eval length mean | Final rollout return mean | Final rollout length mean |
|---|---:|---:|---:|---:|---:|---:|
| `flow_endpoint` | `flow` | 3/3 | -3.6674 | 60.08 | -3.5414 | 85.33 |
| `flow_endpoint` | `mlp` | 3/3 | -4.5951 | 58.94 | -4.7720 | 60.33 |
| `mlp_ref` | `flow` | 2/3 eval, 3/3 rollout | -3.5919 | 61.39 | -3.4818 | 66.78 |
| `mlp_ref` | `mlp` | 3/3 | -4.5215 | 63.62 | -4.5700 | 66.33 |

The flow policy variants are less bad than the MLP policy variants in this
frozen-WM extraction table, but all policies still fail quickly and remain far
below both the expert-filtered BC baseline and the collector. The result is
diagnostic only; it does not support a policy-improvement claim.

Active GPU queue after row7 rollout completion:

```text
no active project GPU jobs observed
non-embers GPU jobs observed = none
```

Conservative BC-warmstart ablation prepared at 2026-05-28 03:08 EDT:

```text
stage = rerun_g1_bcwarm_pwm_bcreg10_short2k_lr1e4_mlpwm_vs_flowwm_seed0_20260528
manifest = scripts/outputs/mjlab_qs/manifests/rerun_g1_bcwarm_pwm_bcreg10_short2k_lr1e4_mlpwm_vs_flowwm_seed0_20260528.csv
rows = mlp_ref+mlp seed0, flow_endpoint+mlp seed0
bc_warmstart_iters = 50000
policy_iters = 2000
actor_lr = 1e-4
critic_lr = 5e-4
bc_lr = 5e-4
policy_bc_reg = 10.0
bc_quality_filter = expert,expert_noisy
policy_quality_filter = expert,expert_noisy
W&B project = flow-mbpo-mjlab-bcwarm-pwm-conservative-20260528
```

This is the next conservative recovery test after the BC-reg-1, 10k-policy run
collapsed. It keeps the same expert-filtered 50k BC warm start but reduces the
imagined policy-update budget by 5x, lowers actor LR by 5x, and increases the
BC action anchor by 10x. Success is not higher imagined return; success requires
preserving or improving real-env behavior relative to the expert-filtered BC
baseline and producing final/best rollout videos.

Conservative BC-warmstart ablation submission at 2026-05-28 03:10 EDT:

```text
Slurm job = 9234991
partition = gpu-a100
qos = embers
array rows = 0-1
max concurrent = 1
time limit = 03:00:00
state at submission check = PENDING
non-embers GPU jobs observed = none
```

The run uses committed code `c8c4a1b` and W&B is enabled in the manifest. After
both rows complete, render final and true-best rollout MP4s before interpreting
the scalar evals.

Conservative ablation H200 fallback submission at 2026-05-28 03:16 EDT:

```text
primary job = 9234991, gpu-a100, qos embers, still pending
fallback job = 9235104, gpu-h200, qos embers, pending
manifest = scripts/outputs/mjlab_qs/manifests/rerun_g1_bcwarm_pwm_bcreg10_short2k_lr1e4_mlpwm_vs_flowwm_seed0_20260528.csv
reason = A100 estimated start was unavailable; H200 fallback uses the same output paths and row locks
non-embers GPU jobs observed = none
```

The duplicate submission is safe because `run_policy_extraction_row.py` locks
each row output directory and skips complete rows. If one array starts first,
the other should skip rows that are already running or complete.

Conservative ablation launch update at 2026-05-28 03:11 EDT:

```text
additional fallback job = 9235185, gpu-h100, qos embers, pending
active row = 9235104_0, gpu-h200, qos embers, running
row = mlp_ref+mlp seed0
W&B run = jz5juvkc
startup check = BC warmstart active, bc/action_mse decreased from 0.3116 at iter 1 to 0.00131 at iter 6000
non-embers GPU jobs observed = none
```

Conservative ablation row0 completion and follow-up at 2026-05-28 03:23 EDT:

```text
completed policy row = mlp_ref+mlp seed0
policy job = 9235104_0, gpu-h200, qos embers, completed
W&B policy run = jz5juvkc
final eval = return -1.4710, episode length 54.33
best imagined return = 634.3125 at iter 2000
best checkpoint = true best-actor snapshot
rollout job = 9235522, gpu-h200, qos embers, pending
rollout manifest = scripts/outputs/mjlab_qs/manifests/rerun_g1_bcwarm_pwm_bcreg10_short2k_lr1e4_mlp_ref_seed0_rollout_20260528.csv
```

The scalar eval is less negative than the previous BC-reg-1, 10k-policy run,
but this cannot be interpreted until final/best rollout MP4s are available.

W&B step-order fix at 2026-05-28 03:24 EDT:

```text
commit = c6c74a5
issue = BC warmstart logs reached step 50000, then policy logs restarted at step 1 and were ignored by W&B
fix = offset policy/online W&B steps by bc_warmstart_iters
scope = row0 was already running and still has partial policy-history warnings; later rows use the fix
```

Flow-endpoint row standalone submission at 2026-05-28 03:24 EDT:

```text
original row1 state = 9235104_[1%1], gpu-h200, qos embers, pending with JobArrayTaskLimit
standalone manifest = scripts/outputs/mjlab_qs/manifests/rerun_g1_bcwarm_pwm_bcreg10_short2k_lr1e4_flow_endpoint_seed0_policy_20260528.csv
standalone job = 9235536, gpu-h200, qos embers
reason = avoid waiting on stale array task-limit scheduling; row locks still protect the shared output path
```
