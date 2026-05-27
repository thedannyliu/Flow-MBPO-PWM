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
