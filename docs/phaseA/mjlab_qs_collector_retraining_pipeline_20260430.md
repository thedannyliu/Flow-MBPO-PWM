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
