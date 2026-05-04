# MJLab-QS Neutral Collector Retraining Pipeline - 2026-04-30

## Problem

The corrected MJLab-QS quality probe showed that the previous collector-labeled `expert` data is not empirically expert-quality:

- Old A2.5 data: both Go1 and G1 had zero empirical expert episodes.
- Corrected L40S quality probe: raw NaNs were fixed, but Go1 still had no empirical expert episodes and G1 had only `5 / 100` empirical expert episodes.

Therefore, formal `D_QS_core` recollection and A2.5/A3 world-model training are blocked until stronger neutral collectors exist.

## Goal

Train neutral MLP/PWM collector candidates that can generate quality-stratified MJLab offline data. These collectors are not part of the Flow-vs-MLP comparison. They are data-generation policies only.

A collector is eligible for formal QS data collection only if it passes empirical gates measured from actual rollouts:

```text
fall_rate <= 0.10
episode_length_mean >= 800
return_mean >= random_return_mean + 1.0
empirical expert episodes >= 50 in a 100-episode probe
raw reward/action NaN count = 0
```

## Training Design

Tasks:

```text
velocity_flat_unitree_go1
velocity_flat_unitree_g1
```

Collector family:

```text
mlpwm_mlppolicy only
```

Profiles:

```text
pwmorig_long
  alg = pwm_5M_baseline_pwmorig
  max_epochs = 50,000
  purpose = original PWM-aligned MLP baseline with longer training

baseline_final_rewrms_long
  alg = pwm_5M_baseline_final
  max_epochs = 50,000
  purpose = MLP baseline with reward RMS enabled

large48m_long
  alg = pwm_48M
  max_epochs = 50,000
  purpose = larger MLP world-model capacity while still neutral MLP policy/WM
```

Seeds:

```text
0, 1, 2
```

Total collector training jobs:

```text
2 tasks x 3 profiles x 3 seeds = 18 jobs
```

All runs use strict MJLab task resolution and attempt to keep canonical collection conditions aligned with the QS protocol:

```text
+env.config.mjlab_env_kwargs.domain_randomization=false
alg.save_interval = 1000 for 5M profiles
alg.save_interval = 2500 for 48M profile
```

Hydra note: `mjlab_env_kwargs` is an open dictionary in the environment
config, so `domain_randomization` must be inserted with a `+` override. A
plain assignment fails when the key is not already present.

## Manifests

Generated with:

```text
scripts/experiments/mjlab_qs/build_collector_retrain_manifest.py
```

Outputs:

```text
scripts/outputs/mjlab_qs/manifests/collector_retrain_v1.csv
scripts/outputs/mjlab_qs/manifests/collector_retrain_v1_h100.csv
scripts/outputs/mjlab_qs/manifests/collector_retrain_v1_h200.csv
scripts/outputs/mjlab_qs/manifests/collector_retrain_v1_l40s.csv
```

The shards distribute one seed per GPU class:

```text
H100: seed 0 rows
H200: seed 1 rows
L40S: seed 2 rows
```

## Post-Training Pipeline

After collector training finishes:

1. Use `eval_summary.json` from each run to rank candidates by episode length and return.
2. Build a quality-probe collection manifest with:

```text
scripts/experiments/mjlab_qs/build_collector_quality_probe_from_runs.py
```

3. Collect random reference + top collector checkpoint rollouts.
4. Run:

```text
scripts/experiments/mjlab_qs/audit_mjlab_qs_quality.py
```

5. Only if the audit passes, build the formal QS dataset and submit A2.5/A3 WM feasibility training.

## Current Execution Status

Collector retraining was submitted as Slurm arrays on PACE ICE:

```text
H100: collector_retrain_v1_h100.csv
H200: collector_retrain_v1_h200.csv
L40S: collector_retrain_v1_l40s.csv
```

Formal QS collection and A2.5/A3 WM training remain blocked until these collectors pass the empirical quality gate.

## Submission Record

The first submission attempt used `36:00:00`, then `24:00:00`; PACE ICE rejected both as exceeding the active QoS/partition limit. The jobs were submitted with `08:00:00` chunks instead. The single-task online runner resumes from `latest_checkpoint.pt` / `final_policy.pt`, so the same manifests can be resubmitted to continue incomplete 50k-epoch collectors.

Submitted Slurm arrays:

```text
5147405  sto_mjlab_qs_collector_retrain_v1_H100  manifest=collector_retrain_v1_h100.csv
5147404  sto_mjlab_qs_collector_retrain_v1_H200  manifest=collector_retrain_v1_h200.csv
5147406  sto_mjlab_qs_collector_retrain_v1_L40S  manifest=collector_retrain_v1_l40s.csv
```

Current state at submission time: pending due to PACE GPU maintenance reservation:

```text
ReqNodeNotAvail, Reserved for maintenance
```

No formal QS data recollection or A2.5/A3 world-model training was submitted. Those remain blocked until retrained collectors pass the empirical quality gate.

## Failure Diagnosis And Retry Patch - 2026-05-01

The first executed collector retraining arrays failed before training started.
The failure was not a GPU allocation or CUDA problem. All H100, H200, and L40S
rows exited during Hydra config composition with:

```text
Could not override 'env.config.mjlab_env_kwargs.domain_randomization'.
To append to your config use +env.config.mjlab_env_kwargs.domain_randomization=false
Key 'domain_randomization' is not in struct
```

Patch:

```text
env.config.mjlab_env_kwargs.domain_randomization=false
```

was replaced with:

```text
+env.config.mjlab_env_kwargs.domain_randomization=false
```

This keeps canonical QS collection aligned with domain randomization disabled,
but uses the correct Hydra syntax for adding the key to `mjlab_env_kwargs`.

Retry arrays submitted after regenerating the manifests:

```text
5148957  sto_mjlab_qs_collector_retrain_v1_H100  manifest=collector_retrain_v1_h100.csv
5148959  sto_mjlab_qs_collector_retrain_v1_H200  manifest=collector_retrain_v1_h200.csv
5148958  sto_mjlab_qs_collector_retrain_v1_L40S  manifest=collector_retrain_v1_l40s.csv
```

Initial retry status: H100 and L40S rows started successfully and reached W&B
initialization. The earlier Hydra override failure is no longer present. H200
rows were still pending for GPU resources at the first retry check.

## Collector Retraining Status - 2026-05-03

The May 1 retry arrays finished without the previous Hydra failure:

```text
Go1 rows:
  9 / 9 completed with final_policy.pt and eval_summary.json.

G1 rows:
  9 / 9 reached latest_checkpoint.pt but timed out near the 8 hour limit
  before final_policy/eval generation.
```

Go1 eval summaries are available for all three profiles and all three seeds.
The strongest Go1 candidate so far is:

```text
velocity_flat_unitree_go1 / baseline_final_rewrms_long / seed_0
  return_mean = 3.826
  episode_length_mean = 63.713
```

This is still far below the empirical expert gate (`episode_length_mean >= 800`,
low fall rate), so the retrained Go1 collectors are not yet approved for
formal QS expert data.

G1 timeout diagnosis:

```text
G1 jobs were still training at timeout, usually around 76% to 97% of 50k epochs.
latest_checkpoint.pt was written for each G1 row, so the single-task online
runner can resume them.
```

Submitted G1-only resume arrays:

```text
5243858  sto_mjlab_qs_collector_retrain_v1_H100  manifest=collector_retrain_v1_g1_retry_h100.csv
5243857  sto_mjlab_qs_collector_retrain_v1_H200  manifest=collector_retrain_v1_g1_retry_h200.csv
5243856  sto_mjlab_qs_collector_retrain_v1_L40S  manifest=collector_retrain_v1_g1_retry_l40s.csv
```

Initial resume status: L40S rows and one H100 row started and W&B reported
`Resuming run ...`; H200 and remaining H100 rows were pending for priority.
There were no new Hydra or CUDA errors at the first resume check.

## MJLab-Native PPO Collector Branch - 2026-05-03

The PWM-style collector retraining branch has not yet produced empirically
expert-quality Go1 collectors. Go1 completed rows remain far below the formal
QS expert gate, with the best completed candidate currently at:

```text
velocity_flat_unitree_go1 / baseline_final_rewrms_long / seed_0
  return_mean = 3.826
  episode_length_mean = 63.713
```

This is not sufficient for `D_QS_core` expert data. Therefore, a separate
neutral MJLab-native PPO/RSL-RL collector branch was added. This branch is
still only a data-generation policy search; it is not part of the Flow-vs-MLP
world-model comparison.

Two MJLab-native collector methods are submitted:

```text
rslrl_ppo_default
  Uses the MJLab-native RSL-RL/PPO task defaults.
  We only control seed, num_envs, save interval, output path, and logging.

rslrl_ppo_conservative
  Keeps MJLab-native RSL-RL/PPO, but uses lower learning rate / KL / entropy,
  enables actor and critic observation normalization, and removes exogenous
  perturbation/randomization events:
    push_robot, foot_friction, encoder_bias, base_com
  This is intended as a more canonical flat-velocity collector.
```

Both methods are run on:

```text
velocity_flat_unitree_go1
velocity_flat_unitree_g1
seeds = 0, 1, 2
num_envs = 2048 for formal training
```

Smoke testing uses:

```text
stage = mjlab_native_collector_smoke_v1
rows = 2 tasks x 2 methods x seed 0 = 4
num_envs = 64
max_iterations = 2
logger = tensorboard
W&B = disabled by using tensorboard logger
```

Formal training uses:

```text
stage = mjlab_native_collector_v1
rows = 2 tasks x 2 methods x 3 seeds = 12
logger = wandb
wandb_project = flow-mbpo-mjlab-native-collector
```

Formal shards are split by seed / GPU class:

```text
H100: seed 0 rows
H200: seed 1 rows
L40S: seed 2 rows
```

Submission record:

```text
5243907  mjqs_native_collector_L40S  smoke manifest=mjlab_native_collector_smoke_v1.csv
5243912  mjqs_native_collector_H100  formal manifest=mjlab_native_collector_v1_h100.csv  dependency=afterok:5243907
5243914  mjqs_native_collector_H200  formal manifest=mjlab_native_collector_v1_h200.csv  dependency=afterok:5243907
5243913  mjqs_native_collector_L40S  formal manifest=mjlab_native_collector_v1_l40s.csv  dependency=afterok:5243907
```

The dependency is intentional: formal runs cannot start unless all smoke rows
finish successfully. This preserves the required execution order while avoiding
idle time after smoke success.

Implementation note:

```text
scripts/experiments/mjlab_qs/run_mjlab_native_collector.py
```

patches two headless-cluster compatibility issues before importing the MJLab
training stack:

```text
1. MuJoCo enum drift: missing mjENBL_MULTICCD is shimmed to 0.
2. mediapy optional display import: a tiny IPython.display shim is installed
   when IPython is absent from the training environment.
```

The second patch is required because `mjlab.scripts.train` imports MJLab's
video wrapper, which imports `mediapy`; `mediapy` imports `IPython.display`
even when video display is not used.

Quality gate is unchanged. These MJLab-native collectors still need empirical
rollout audits before they can be used for formal QS expert data:

```text
fall_rate <= 0.10
episode_length_mean >= 800
return_mean >= random_return_mean + 1.0
empirical expert episodes >= 50 in a 100-episode probe
raw reward/action NaN count = 0
```

Formal QS dataset collection and A2.5/A3 world-model feasibility remain blocked
until one or more collectors pass this gate.

## MJLab-Native PPO Collector Status - 2026-05-04

All MJLab-native smoke and formal collector rows completed.

Smoke:

```text
5243907  L40S  mjlab_native_collector_smoke_v1
  rows = 4 / 4 completed
  last checkpoint = model_1.pt for all rows
  logger = tensorboard only
```

Formal:

```text
5243912  H100  seed 0 shard
5243914  H200  seed 1 shard
5243913  L40S  seed 2 shard
```

Formal completion check:

```text
velocity_flat_unitree_go1 / rslrl_ppo_default       seeds 0,1,2  last=model_9999.pt   complete
velocity_flat_unitree_go1 / rslrl_ppo_conservative  seeds 0,1,2  last=model_9999.pt   complete
velocity_flat_unitree_g1  / rslrl_ppo_default       seeds 0,1,2  last=model_29999.pt  complete
velocity_flat_unitree_g1  / rslrl_ppo_conservative  seeds 0,1,2  last=model_29999.pt  complete
```

No native collector log contains Python tracebacks, timeout messages, CUDA
errors, or module import failures. The earlier headless dependency issue was
handled by the runner shim before submission.

Training summaries exported from local W&B summaries:

```text
Go1 / rslrl_ppo_conservative:
  seed 0: train_mean_reward=89.107, train_mean_episode_length=984.57
  seed 1: train_mean_reward=89.269, train_mean_episode_length=991.28
  seed 2: train_mean_reward=88.877, train_mean_episode_length=990.28

Go1 / rslrl_ppo_default:
  seed 0: train_mean_reward=71.513, train_mean_episode_length=969.52
  seed 1: train_mean_reward=70.048, train_mean_episode_length=941.89
  seed 2: train_mean_reward=69.316, train_mean_episode_length=952.23

G1 / rslrl_ppo_conservative:
  seed 0: train_mean_reward=69.550, train_mean_episode_length=991.59
  seed 1: train_mean_reward=69.701, train_mean_episode_length=1000.00
  seed 2: train_mean_reward=70.744, train_mean_episode_length=996.15

G1 / rslrl_ppo_default:
  seed 0: train_mean_reward=44.457, train_mean_episode_length=998.88
  seed 1: train_mean_reward=45.453, train_mean_episode_length=990.69
  seed 2: train_mean_reward=46.169, train_mean_episode_length=988.79
```

Interpretation:

```text
1. The MJLab-native branch is much stronger than the previous PWM-style
   collector branch on the available training summaries.
2. Go1 conservative is currently the strongest Go1 collector candidate.
3. G1 conservative has higher training reward than G1 default, but default has
   lower x/y velocity tracking error in the final summary. Both should be
   quality-probed before selecting the final collector.
4. These are training summaries, not formal QS rollout audits. A collector is
   not approved for D_QS_core until it passes the empirical rollout quality
   probe with raw NaN checks, fall-rate checks, return, and episode length.
```

Status CSV:

```text
scripts/outputs/mjlab_qs/native_collectors/mjlab_native_collector_v1_status_20260504.csv
```

Next required action:

```text
Build and run a quality-probe manifest for the completed MJLab-native
collectors. Probe at least the final checkpoint for both methods and all seeds,
then rank by empirical return / fall rate / episode length. Only passing
checkpoints can be used as expert or medium QS data collectors.
```

## Native Quality-Probe Collection Submission - 2026-05-04

The old raw episode collector only supports PWM checkpoints. MJLab-native PPO
checkpoints are RSL-RL checkpoints and require the native MJLab observation
TensorDict path. A new native raw-episode collector was added:

```text
scripts/experiments/mjlab_qs/collect_mjlab_qs_native_episodes.py
scripts/experiments/mjlab_qs/build_native_collector_probe_manifest.py
scripts/experiments/mjlab_qs/run_native_collection_row.py
```

The native collector writes the same raw episode schema expected by the existing
QS audit and window builder:

```text
env_obs
phys_obs
model_obs
command
policy_action
env_action
reward
transition_valid
termination
truncation
done
```

Important implementation choice:

```text
Terminal transitions are dropped in native collection.
```

Reason: native MJLab auto-resets terminated environments before returning the
next observation. Since the native API does not expose a reliable
`obs_before_reset`, storing the terminal transition would risk reset-state
contamination. Dropping the final terminal transition preserves clean
pre-terminal dynamics windows. Episode quality still records whether an episode
terminated or timed out.

Quality-probe manifests:

```text
scripts/outputs/mjlab_qs/manifests/mjlab_native_quality_probe_smoke_v1.csv
scripts/outputs/mjlab_qs/manifests/mjlab_native_quality_probe_v1.csv
```

Smoke:

```text
5251034  L40S  mjqs_native_collection_L40S
  stage = mjlab_native_quality_probe_smoke_v1
  rows = 4
  coverage = 2 tasks x (random_smooth + conservative seed 0)
  episodes = 2 per row
```

Formal probe:

```text
5251035  L40S  mjqs_native_collection_L40S
  stage = mjlab_native_quality_probe_v1
  rows = 14
  coverage = 2 tasks x (random_smooth + 2 methods x 3 seeds)
  episodes = 100 per row
  dependency = afterok:5251034_*
```

After the formal probe completes, run:

```text
python scripts/experiments/mjlab_qs/audit_mjlab_qs_quality.py \
  --raw scripts/outputs/mjlab_qs/raw/mjlab_native_quality_probe_v1 \
  --csv-output scripts/outputs/mjlab_qs/audits/mjlab_native_quality_probe_v1.csv \
  --json-output scripts/outputs/mjlab_qs/audits/mjlab_native_quality_probe_v1.json \
  --md-output scripts/outputs/mjlab_qs/audits/mjlab_native_quality_probe_v1.md
```

Also export a per-shard native ranking for selecting the formal QS collector:

```text
python scripts/experiments/mjlab_qs/export_native_quality_probe_ranking.py \
  --raw scripts/outputs/mjlab_qs/raw/mjlab_native_quality_probe_v1 \
  --output scripts/outputs/mjlab_qs/audits/mjlab_native_quality_probe_v1_ranking.csv
```

If both Go1 and G1 have expert-gate-passing native checkpoints, build formal
`D_QS_core` collection rows with:

```text
python scripts/experiments/mjlab_qs/build_native_qs_collection_manifest.py \
  --ranking scripts/outputs/mjlab_qs/audits/mjlab_native_quality_probe_v1_ranking.csv \
  --output scripts/outputs/mjlab_qs/manifests/a25_native_qs_collection.csv \
  --stage a25_native_qs
```

This produces the A2.5 QS bucket mix:

```text
random_smooth: 63 episodes/task
weak:          125 episodes/task
medium:        219 episodes/task
expert:        157 episodes/task
expert_noisy:   63 episodes/task
```

Only checkpoints passing this audit can be used to build formal `D_QS_core`.

Current cluster note:

```text
Only one unrelated sidecar job remains in queue:
  5243879  swm_fit_flow_scene_expert_0  PENDING  DependencyNeverSatisfied

It is not part of the MJLab-native collector branch and is not blocking the QS
collector pipeline, but it should be canceled or resubmitted separately if that
sidecar is still needed.
```
