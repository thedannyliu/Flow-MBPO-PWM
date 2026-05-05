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

## Native Quality-Probe Audit - 2026-05-04

The formal native quality probe completed with 14/14 raw shards and no Slurm
Traceback/Error/Exception messages.

Artifacts:

```text
scripts/outputs/mjlab_qs/raw/mjlab_native_quality_probe_v1
scripts/outputs/mjlab_qs/audits/mjlab_native_quality_probe_v1.csv
scripts/outputs/mjlab_qs/audits/mjlab_native_quality_probe_v1.json
scripts/outputs/mjlab_qs/audits/mjlab_native_quality_probe_v1.md
scripts/outputs/mjlab_qs/audits/mjlab_native_quality_probe_v1_ranking.csv
```

Empirical quality result:

```text
G1:
  rslrl_ppo_conservative seed 0 passed the expert gate.
  return_mean ~= 81.35
  episode_length_mean ~= 984.02
  fall_rate ~= 0.03

Go1:
  no checkpoint passed the expert gate.
  best candidate = rslrl_ppo_conservative seed 0
  return_mean ~= 70.00
  episode_length_mean ~= 856.20
  fall_rate ~= 0.30
```

Decision:

```text
Do not build canonical two-task D_QS_core from this probe.
The G1 collector is usable as expert, but the Go1 expert quality is not yet
acceptable. Go1 must be retrained or otherwise strengthened before it can enter
formal A2.5/A3 data collection.
```

Reason:

```text
The restart protocol requires empirical quality bins. A collector label is not
enough. Expert data must have low fall rate, long episode length, and return
clearly above random. The Go1 probe violates the fall-rate requirement.
```

## Go1 Native Long Collector Retraining - 2026-05-04

The first native Go1 collectors were trained for 10k iterations, while G1 used
30k iterations. Since Go1 failed the empirical expert gate, the next controlled
step is to retrain Go1 native collectors for 30k iterations with the same two
neutral MJLab-native methods:

```text
rslrl_ppo_default
rslrl_ppo_conservative
```

Manifest:

```text
scripts/outputs/mjlab_qs/manifests/mjlab_native_collector_go1_long_v2.csv
```

Split manifests for GPU diversity:

```text
scripts/outputs/mjlab_qs/manifests/mjlab_native_collector_go1_long_v2_h100.csv
scripts/outputs/mjlab_qs/manifests/mjlab_native_collector_go1_long_v2_h200.csv
scripts/outputs/mjlab_qs/manifests/mjlab_native_collector_go1_long_v2_l40s.csv
```

Configuration:

```text
task = Mjlab-Velocity-Flat-Unitree-Go1
methods = rslrl_ppo_default, rslrl_ppo_conservative
seeds = 0, 1, 2
num_envs = 2048
max_iterations = 30000
save_interval = 500
wandb_project = flow-mbpo-mjlab-native-collector
wandb_group = mjlab_native_collector_go1_long_v2
```

Acceptance before formal QS collection:

```text
At least one Go1 checkpoint must pass empirical expert gate under a 100-episode
probe:
  low fall rate
  long episode length
  return clearly above random
  no NaN action/reward
  acceptable action clip fraction
```

Submission:

```text
5251165  H100  native_collector  manifest=mjlab_native_collector_go1_long_v2_h100.csv
5251166  H200  native_collector  manifest=mjlab_native_collector_go1_long_v2_h200.csv
5251167  L40S  native_collector  manifest=mjlab_native_collector_go1_long_v2_l40s.csv
```

Post-training action:

```text
1. Build a new Go1 native quality-probe manifest from
   mjlab_native_collector_go1_long_v2.
2. Collect 100-episode empirical probes for all Go1 long checkpoints.
3. Audit by return / fall rate / episode length / NaN / clip fraction.
4. If Go1 passes expert gate, combine it with the existing passing G1 expert
   checkpoint to build A2.5 D_QS_core.
5. Only then build windows and launch WM feasibility / PWM pipeline comparison.
```

## Native QS Window-to-Training Manifest Builder - 2026-05-04

Added a training-manifest builder that starts from already validated native QS
window datasets instead of the older PWM-checkpoint hardcoded manifest builder:

```text
scripts/experiments/mjlab_qs/build_phaseA_train_manifest_from_windows.py
```

Usage after formal native QS windows exist:

```text
python scripts/experiments/mjlab_qs/build_phaseA_train_manifest_from_windows.py \
  --stage a25_native_qs \
  --output scripts/outputs/mjlab_qs/manifests/a25_native_qs_train.csv \
  --methods mlp_ref,flow_ref,residual_flow_frozen_mlp \
  --seeds 0,1,2 \
  --train-iters 50000 \
  --eval-every 5000
```

This preserves the A2.5 execution rule:

```text
Only formal training rows are sent to W&B.
Smoke / collection / audit / window-building steps are local or Slurm-only and
are not logged to W&B as model-comparison results.
```

H100/H200 startup was still pending on priority while L40S rows were running.
To avoid blocking the collector-quality gate on H100/H200 availability, an
independent L40S backup stage was submitted for the rows not covered by the
original L40S split. The backup uses separate output directories, so it cannot
corrupt pending H100/H200 outputs.

```text
5251177  L40S  native_collector  manifest=mjlab_native_collector_go1_long_v2_l40s_backup.csv
stage = mjlab_native_collector_go1_long_v2_l40sbackup
coverage = Go1 default seeds 0/1/2 + Go1 conservative seed 0
```

## Automated A2.5 Native QS Pipeline Driver - 2026-05-04

Added a gated driver that can run after the Go1 long L40S collector jobs finish:

```text
scripts/experiments/mjlab_qs/run_a25_native_qs_after_go1_long.sh
```

Behavior:

```text
1. Build a combined Go1-long quality-probe manifest from the primary long stage
   and the L40S backup stage.
2. Run 100-episode Go1 empirical probes sequentially on one GPU.
3. Audit and rank Go1 long checkpoints. The probe audit is allowed to fail
   because it contains multiple candidate checkpoints; selection is based on
   per-shard expert_gate_pass in the ranking file.
4. Combine the new Go1 ranking with the previous G1 ranking.
5. Build formal a25_native_qs collection manifest only if both tasks have an
   expert-gate-passing checkpoint.
6. Collect formal a25_native_qs raw episodes.
7. Run strict formal dataset audit.
8. Build H=16 windows with the required bucket/window gates.
9. Build A2.5 training manifests and submit formal WM training arrays.
```

This driver preserves the hard gate: if Go1 still fails empirical expert
quality, the formal dataset and WM training will not be launched.

Pipeline submission:

```text
5251184  L40S  mjqs_a25_native_qs_pipeline
  dependency = afterok:5251167:5251177
  script = scripts/experiments/mjlab_qs/run_a25_native_qs_after_go1_long.sh
```

This job intentionally depends only on the L40S Go1-long stages because they
cover all required Go1 long candidates without output-directory conflicts. The
pending H100/H200 duplicate collectors are not canceled, but they are not on the
critical path for A2.5 dataset construction.

Runtime adjustment:

```text
L40S checkpoint rate was about 500 native PPO iterations per 9 minutes.
At that rate, 30k iterations can exceed the original 10h Slurm limit.
Attempted to extend Go1-long collector jobs to 24h and the gated pipeline job
to 18h to avoid timeout-induced partial collectors.
```

Pipeline correction:

```text
5251184 was canceled because a Slurm update left it on a CPU partition while
still requesting a GPU.
5251191 was resubmitted with the correct ice-gpu / L40S / 12h configuration.

dependency = afterok:5251167:5251177
```

## G1-Only Diagnostic Dataset Branch - 2026-05-04

Go1 remains blocked after the long native retrain: no Go1 checkpoint passes the
empirical expert gate. G1 is usable and has a passing native conservative expert
collector:

```text
G1 expert = native_rslrl_ppo_conservative_seed0
return_mean ~= 81.35
episode_length_mean ~= 984.02
fall_rate ~= 0.03
```

Decision:

```text
Start a G1-only diagnostic branch. This is not the canonical two-task A2.5
result, but it is a valid single-task diagnostic dataset for checking the WM /
PWM training pipeline while Go1 collector quality remains unresolved.
```

Implementation update:

```text
scripts/experiments/mjlab_qs/build_native_qs_collection_manifest.py
```

now supports:

```text
--tasks velocity_flat_unitree_g1
```

G1-only collection manifest:

```text
scripts/outputs/mjlab_qs/manifests/a25_native_qs_g1only_collection.csv
```

G1-only post-collection driver:

```text
scripts/experiments/mjlab_qs/run_g1only_after_collection.sh
```

Submission:

```text
5254303  L40S  native_collection  manifest=a25_native_qs_g1only_collection.csv
5254311  L40S  g1only_after_collection  dependency=afterok:5254303
```

The G1-only branch will:

```text
1. Strictly audit the G1-only raw dataset.
2. Build H=16 windows for velocity_flat_unitree_g1.
3. Use a reduced diagnostic window gate of 500 valid train windows per bucket,
   because random_smooth G1 episodes are intentionally short and this branch is
   not the canonical two-task result.
4. Build formal A2.5 training rows for mlp_ref, flow_ref, and
   residual_flow_frozen_mlp with seeds 0,1,2.
5. Submit formal W&B training arrays split across H100, H200, and L40S.
```

## G1 Training-Stage Weak/Medium Correction - 2026-05-04

The first G1-only dataset used `checkpoint_blend_random` for weak and medium.
That is useful for pipeline smoke, but it does not match the intended QS data
logic. The correct QS source for weak and medium should be earlier checkpoints
from the same neutral collector training curve.

Decision:

```text
Stop relying on expert/action-random blending for weak and medium.
Probe intermediate checkpoints from the G1 conservative seed-0 native collector,
then select weak/medium/expert by empirical return, episode length, and fall
rate.
```

Implementation:

```text
scripts/experiments/mjlab_qs/build_native_checkpoint_stage_probe_manifest.py
scripts/experiments/mjlab_qs/export_native_stage_quality_ranking.py
```

Probe manifest:

```text
scripts/outputs/mjlab_qs/manifests/mjlab_native_g1_stage_probe_v1.csv
```

Probe checkpoints:

```text
0, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500,
10000, 15000, 20000, 25000, 29999
```

Submission:

```text
5254406  L40S  native_collection  manifest=mjlab_native_g1_stage_probe_v1.csv
```

The previous G1-only formal WM training jobs based on blended weak/medium data
were canceled before completion. They should be treated as invalidated pipeline
sanity jobs, not as formal data-quality results.

Added a stage-ranking QS manifest builder:

```text
scripts/experiments/mjlab_qs/build_native_qs_collection_manifest_from_stage_ranking.py
```

This builder selects:

```text
weak   = best empirical weak checkpoint
medium = best empirical medium checkpoint
expert = best empirical expert checkpoint
```

and then builds a formal QS collection manifest where weak/medium are actual
training-stage policies, not expert-random action blends.

## G1 Stage-Probe Status - 2026-05-04

Seed-0 checkpoint-stage probe completed and showed that the G1 conservative
training curve jumps from failed/random to expert very quickly:

```text
iter250:   random_or_failed
iter500:   expert
iter15000: medium
iter29999: expert
```

This means seed 0 alone does not provide a clean weak checkpoint. To avoid
fabricating weak data, additional stage probes were submitted for conservative
seed 1 and seed 2:

```text
5254563  L40S  native_collection  manifest=mjlab_native_g1_stage_probe_seed1_v1.csv
5254564  L40S  native_collection  manifest=mjlab_native_g1_stage_probe_seed2_v1.csv
```

If no empirical weak checkpoint exists across seeds 0/1/2, the corrected G1 QS
dataset should either omit weak as a formal bucket or use a separate neutral
collector/replay-buffer source. It should not use expert-random action blending
as a substitute for a weak policy.

## G1 Four-Bucket Dataset Decision - 2026-05-04

The dataset is now changed to four buckets:

```text
random_smooth
medium
expert
expert_noisy
```

Reason:

```text
The G1 conservative seed-0 learning curve does not provide a clean empirical
weak checkpoint. It jumps from random/failed to expert very quickly. Forcing a
weak bucket would either fabricate data or use expert-random action blending,
which does not match the intended QS data semantics.
```

Selected checkpoint-stage sources:

```text
medium = rslrl_ppo_conservative seed 0 iter 15000
  return_mean ~= 67.35
  episode_length_mean ~= 836.33
  fall_rate ~= 0.27

expert = rslrl_ppo_conservative seed 0 iter 29999
  return_mean ~= 81.41
  episode_length_mean ~= 992.08
  fall_rate ~= 0.02
```

Corrected manifest:

```text
scripts/outputs/mjlab_qs/manifests/a25_native_qs_g1stage4_collection.csv
```

This branch should supersede the previous blended G1-only dataset for formal
single-task diagnostics.

Submission status:

```text
5254572  L40S  native_collection  stage=a25_native_qs_g1stage4
5254573  L40S  after-collection pipeline, dependency=afterok:5254572
```

Because the L40S job was still pending on priority, two non-overlapping shadow
stages were also submitted. They write to separate raw/window/result
directories and must be reported as shadow duplicates unless they are the first
complete run adopted for the G1-only diagnostic:

```text
5254582  H100  native_collection  stage=a25_native_qs_g1stage4_h100shadow
5254584  H100  after-collection pipeline, dependency=afterok:5254582

5254583  H200  native_collection  stage=a25_native_qs_g1stage4_h200shadow
5254585  H200  after-collection pipeline, dependency=afterok:5254583
```

Adoption rule:

```text
Use the first stage that fully completes:
1. raw episode collection,
2. quality audit,
3. H=16 window build with the minimum valid-window gate,
4. A2.5 formal WM training for mlp_ref, flow_ref, residual_flow_frozen_mlp
   with seeds 0/1/2.

Do not merge metrics across the L40S/H100/H200 duplicate stages. They are
fallback execution lanes for the same four-bucket G1-only diagnostic.
```

## G1 Four-Bucket A2.5 Execution Status - 2026-05-04 16:05

The L40S lane completed first and is the current adopted G1-only diagnostic
stage:

```text
5254572  L40S  native_collection  COMPLETED
5254573  L40S  after-collection pipeline  COMPLETED
```

Artifacts:

```text
raw:
  scripts/outputs/mjlab_qs/raw/a25_native_qs_g1stage4/

audit:
  scripts/outputs/mjlab_qs/audits/a25_native_qs_g1stage4.{csv,json,md}

windows:
  scripts/outputs/mjlab_qs/windows/a25_native_qs_g1stage4/velocity_flat_unitree_g1/d_qs_core_h16.pt
  scripts/outputs/mjlab_qs/windows/a25_native_qs_g1stage4/velocity_flat_unitree_g1/d_qs_core_h16_normalization.json
  scripts/outputs/mjlab_qs/windows/a25_native_qs_g1stage4/velocity_flat_unitree_g1/d_qs_core_h16_report.md

training manifests:
  scripts/outputs/mjlab_qs/manifests/a25_native_qs_g1stage4_train.csv
  scripts/outputs/mjlab_qs/manifests/a25_native_qs_g1stage4_train_h100.csv
  scripts/outputs/mjlab_qs/manifests/a25_native_qs_g1stage4_train_h200.csv
  scripts/outputs/mjlab_qs/manifests/a25_native_qs_g1stage4_train_l40s.csv
```

Quality audit:

```text
status: PASS
episodes: 502
nan_reward: 0
nan_action: 0
```

Window build:

```text
status: PASS
num_windows: 100415
train valid windows:
  random_smooth: 702
  medium: 37883
  expert: 30625
  expert_noisy: 12300
```

The random_smooth bucket is intentionally small because random locomotion
terminates quickly; this G1-only diagnostic uses the relaxed A2.5 gate of at
least 500 valid train windows per bucket.

Formal A2.5 WM training arrays submitted from the adopted L40S stage:

```text
5254700  H100  mlp_ref seeds 0/1/2  PENDING
5254701  H200  flow_ref seeds 0/1/2  PENDING
5254702  L40S  residual_flow_frozen_mlp seeds 0/1/2  PENDING
```

The H200 shadow collection finished, but its after-collection pipeline failed
the quality audit because the duplicate rollout had only 44 empirical expert
episodes, below the 50-episode audit gate:

```text
5254583  H200  native_collection  COMPLETED
5254585  H200  after-collection pipeline  FAILED
reason: empirical expert episodes 44 < 50
```

This does not invalidate the adopted L40S lane, which passed the audit and
window gates. It does show that the current empirical expert threshold is
sensitive to rollout stochasticity at this small A2.5 scale.

## G1 Four-Bucket A2.5 Training Status - 2026-05-05

Equal-update 50k A2.5 status:

```text
5254700  H100  mlp_ref seeds 0/1/2  COMPLETED
5254701  H200  flow_ref seeds 0/1  COMPLETED
5254701  H200  flow_ref seed 2  RUNNING
5254702  L40S  residual_flow_frozen_mlp seeds 0/1/2  COMPLETED
```

Partial 50k results before `flow_ref` seed 2 finishes:

```text
mlp_ref, n=3:
  test_rollout_dyn_mse_H16_mean ~= 0.0372
  final_train_rollout_dyn_mse_H16 ~= 0.0127-0.0140
  test_rollout_error_ratio_e16_e1_mean ~= 1.98

flow_ref, n=2:
  test_rollout_dyn_mse_H16_mean ~= 0.1538
  final_train_rollout_dyn_mse_H16 ~= 0.1507-0.1631
  test_rollout_error_ratio_e16_e1_mean ~= 6.93
  test_rollout_dyn_mse_H16 is currently ~= 4.13x the mlp_ref mean

residual_flow_frozen_mlp, n=3:
  test_rollout_dyn_mse_H16_mean ~= 0.0538
  final_train_rollout_dyn_mse_H16 ~= 0.0156-0.0171
  test_rollout_error_ratio_e16_e1_mean ~= 4.67
  test_rollout_dyn_mse_H16 is currently ~= 1.45x the mlp_ref mean
```

Interpretation:

```text
1. Pure flow_ref has not matched MLP train rollout loss at 50k updates.
2. Pure flow_ref reward loss is in the same range as MLP, so the dominant gap
   remains latent rollout dynamics.
3. residual_flow_frozen_mlp is much closer to MLP on train/val/test rollout
   loss, but seed 0 has a substantially worse e16/e1 rollout-error ratio.
```

The final A2.5 summary CSV must not be exported until `flow_ref` seed 2 writes
its `summary.json`.

Flow train-loss-match sidecar:

```text
manifest:
  scripts/outputs/mjlab_qs/manifests/a25_native_qs_g1stage4_flow_train_match_300k.csv

split manifests:
  scripts/outputs/mjlab_qs/manifests/a25_native_qs_g1stage4_flow_train_match_300k_h100.csv
  scripts/outputs/mjlab_qs/manifests/a25_native_qs_g1stage4_flow_train_match_300k_h200.csv
  scripts/outputs/mjlab_qs/manifests/a25_native_qs_g1stage4_flow_train_match_300k_l40s.csv

jobs:
  5257317  H100  train_match seed 0  direct submission
  5257318  H200  train_match seed 1  direct submission
  5257319  L40S  train_match seed 2  direct submission

previous dependency jobs:
  5257272/5257273/5257274 were canceled before running after the decision to
  submit all train-match diagnostics immediately.
```

This train-loss-match sidecar is an existence diagnostic, not a compute-fair
comparison. Each row retrains an MLP for 50k updates to define the MLP train
rollout-loss target, then trains Flow for up to 300k updates or until the Flow
train H16 rollout loss reaches the MLP target within 5%.

Residual seed-0 rerun sidecar:

```text
manifest:
  scripts/outputs/mjlab_qs/manifests/a25_native_qs_g1stage4_residual_seed0_rerun.csv

job:
  5257321  H100  residual_flow_frozen_mlp seed 0 rerun  direct submission
```

Purpose:

```text
The equal-update residual_flow_frozen_mlp seed 0 has substantially worse rollout
stability than seeds 1/2. This sidecar checks whether that instability is a
random optimization outcome or a reproducible failure mode.
```
