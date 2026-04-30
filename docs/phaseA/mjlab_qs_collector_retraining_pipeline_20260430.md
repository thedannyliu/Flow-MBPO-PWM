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
env.config.mjlab_env_kwargs.domain_randomization=false
alg.save_interval = 1000 for 5M profiles
alg.save_interval = 2500 for 48M profile
```

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
