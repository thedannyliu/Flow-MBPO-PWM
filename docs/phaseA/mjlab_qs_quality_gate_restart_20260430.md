# MJLab-QS Quality Gate Restart - 2026-04-30

## Purpose

This restart blocks any new A2.5/A3 world-model training until the MJLab-QS offline dataset passes an empirical quality gate. The previous A2.5 dataset followed the intended episode-count ratios, but the bucket names were collector labels, not verified behavior-quality labels.

The new rule is:

- `quality_bin` from the collector is source metadata only.
- Formal QS eligibility is determined from empirical episode return, fall rate, and episode length.
- A dataset cannot be called canonical `D_QS_core` unless each task has a real empirical expert bucket.

## Changes Implemented

### 1. Raw Episode NaN Fix

`scripts/experiments/mjlab_qs/collect_mjlab_qs_episodes.py` no longer writes NaN placeholders at row 0.

Row 0 is the initial observation anchor:

- `policy_action[0] = 0`
- `env_action[0] = 0`
- `reward[0] = 0`
- `transition_valid[0] = False`

Actual transitions remain rows `1..T`. Window builders already use transition rows `1..T`, so this preserves alignment while removing raw-shard NaNs.

### 2. Empirical Quality Audit

Added:

```text
scripts/experiments/mjlab_qs/audit_mjlab_qs_quality.py
```

The audit computes per-task/per-bucket:

- episode return mean/std/min/max
- episode length mean/min/max
- fall rate
- action clipping fraction
- reward/action NaN counts
- empirical quality bin

The expert gate currently requires:

```text
fall_rate <= 0.10
episode_length_mean >= 800
return_mean >= random_return_mean + 1.0
empirical expert episodes >= 50
no reward/action NaNs in raw shards
```

These are intentionally conservative. They reflect the restart goal: do not use weak/falling policies as expert data.

### 3. Existing A2.5 Dataset Audit

The previous A2.5 raw shards were audited at:

```text
scripts/outputs/mjlab_qs/quality_audits/a25_existing_empirical_quality.md
scripts/outputs/mjlab_qs/quality_audits/a25_existing_empirical_quality.csv
scripts/outputs/mjlab_qs/quality_audits/a25_existing_empirical_quality.json
```

Result: `FAIL`.

Main failures:

- `Mjlab-Velocity-Flat-Unitree-G1`: 0 empirical expert episodes.
- `Mjlab-Velocity-Flat-Unitree-Go1`: 0 empirical expert episodes.
- Collector-labeled expert fall rate was `1.000` for both tasks.
- Collector-labeled expert episode length was only about `46` for G1 and `31` for Go1.
- Old raw shards contained NaN placeholders in row 0.

Conclusion: the old A2.5 results are useful only as preliminary pipeline/debug results, not as formal QS feasibility results.

## Corrected Replacement Jobs

Submitted corrected quality-probe collection jobs only. These are not training jobs and do not write to W&B.

Manifests:

```text
scripts/outputs/mjlab_qs/manifests/quality_probe_h100.csv
scripts/outputs/mjlab_qs/manifests/quality_probe_h200.csv
scripts/outputs/mjlab_qs/manifests/quality_probe_l40s.csv
```

Submitted Slurm arrays:

```text
5125557  mjqs_collection_H100
5125559  mjqs_collection_H200
5125558  mjqs_collection_L40S
```

Each manifest collects a small probe pool:

- `velocity_flat_unitree_go1`: random_smooth + best available Go1 checkpoint candidate.
- `velocity_flat_unitree_g1`: random_smooth + best available G1 checkpoint candidate.

The goal is to verify:

1. corrected raw episode schema has no NaNs;
2. selected checkpoint candidates can pass empirical expert thresholds;
3. only then build/recollect a formal `D_QS_core` dataset.

## Execution Rule Going Forward

Do not submit formal A2.5/A3 WM training until all are true:

1. quality-probe collection succeeds;
2. empirical quality audit passes for every task;
3. raw shards have zero reward/action NaNs;
4. expert bucket has low fall rate, long episode length, and return clearly above random;
5. window builder passes minimum valid-window gates.

If Go1 still cannot produce an empirical expert, then Go1 cannot enter canonical `D_QS_core` yet. The next step would be to train or locate a stronger neutral collector before formal QS collection.
