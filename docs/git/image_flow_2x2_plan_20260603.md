# Image World-Model Flow 2x2 Plan - 2026-06-03

## Scope

This records the first flow-architecture probes for the official NEWT and
LeWorldModel task pipelines.

## NEWT 2x2

NEWT uses the official `external_repos/newt/tdmpc2/train.py` pipeline. The
runtime patch is loaded via `sitecustomize` and leaves the external checkout
unchanged.

Axes:

- `wm_arch`: `mlp` vs `flow`, replacing the NEWT latent dynamics and reward
  heads with residual-flow MLPs.
- `policy_arch`: `mlp` vs `flow`, replacing the Gaussian policy prior head with
  a residual-flow MLP.

Task:

- `walker-run`
- seeds `0, 1`
- `5000` steps
- metrics source: Slurm stdout final `eval` and `train` rows plus architecture
  markers printed by the patch.

Submission script:

- `scripts/experiments/image_official/submit_newt_flow_2x2_20260603.sh`

## LeWorldModel 2x2

LeWorldModel does not have a standalone actor policy in the official PushT
pipeline; control is CEM planning through a world model. Therefore this 2x2 is
defined over two world-model architecture sites:

- `predictor_arch`: `mlp` vs `flow`, replacing `module.ARPredictor` with a
  residual-flow predictor.
- `action_encoder_arch`: `mlp` vs `flow`, replacing `module.Embedder` with a
  residual-flow action embedder.

Task:

- PushT expert dataset
- seeds `0, 1`
- one epoch, `8` train batches, `2` validation batches
- metrics source: Lightning/stable-pretraining stdout tables for fit and
  validation losses.

Submission script:

- `scripts/experiments/image_official/submit_lewm_flow_2x2_train_20260603.sh`

## Metrics Export

Use:

```bash
python scripts/experiments/image_official/export_flow_2x2_metrics.py \
  --kind newt \
  --manifest <output-dir>/manifest.csv \
  --run-label <run-label> \
  --job-id <array-job-id> \
  --output <output-dir>/metrics.csv
```

For LeWM, change `--kind lewm`.

## Submissions

Submitted on 2026-06-03 with `embers` QOS.

| Pipeline | Label | GPU | Slurm job | Array | Notes |
| --- | --- | --- | --- | --- | --- |
| NEWT | `newt_flow_2x2_h200_20260603` | H200 | `9415373` | `0-7%4` | primary |
| LeWM | `lewm_flow_2x2_train_h200_20260603` | H200 | `9415374` | `0-7%4` | primary |
| NEWT | `newt_flow_2x2_h100_backup_20260603` | H100 | `9415378` | `0-7%4` | backup, distinct output root |
| LeWM | `lewm_flow_2x2_train_h100_backup_20260603` | H100 | `9415379` | `0-7%4` | backup, distinct output root |
| NEWT | `newt_flow_2x2_a100_backup_20260603` | A100 | `9415411` | `0-7%4` | backup, distinct output root |
| LeWM | `lewm_flow_2x2_train_a100_backup_20260603` | A100 | `9415412` | `0-7%4` | backup, distinct output root |

Initial status check:

- `9415373_0` running on H200.
- `9415378_0` running on H100.
- `9415374` and `9415379` pending on priority at first check.
- `9415411` and `9415412` submitted as A100 backups after LeWM remained pending
  and NEWT rows were short enough to duplicate cheaply.

Completion status:

- NEWT H200 primary `9415373_[0-7]` completed with exit `0:0`.
- LeWM H200 primary `9415374_[0-7]` completed with exit `0:0`.
- Remaining NEWT backup rows in `9415378` and `9415411` were cancelled after
  H200 primary completed.
- Remaining LeWM backup rows in `9415379` and `9415412` were cancelled after
  H200 primary completed.

## Initial Results

NEWT H200 metrics:

- CSV: `scripts/outputs/image_official/newt_flow_2x2_h200_20260603/metrics.csv`
- Metric: final `train R` at 5k steps, averaged over seeds 0 and 1.

| `wm_arch` | `policy_arch` | n | final train return mean | final train success mean |
| --- | --- | ---: | ---: | ---: |
| `mlp` | `mlp` | 2 | `16.754` | `0.017` |
| `mlp` | `flow` | 2 | `26.446` | `0.027` |
| `flow` | `mlp` | 2 | `19.320` | `0.019` |
| `flow` | `flow` | 2 | `12.331` | `0.013` |

Interpretation: in this short walker-run smoke, replacing the policy prior with
the flow head improved the 5k train metric when the WM stayed MLP. Replacing the
WM dynamics/reward with the current residual-flow heads did not improve the
short-run metric.

LeWM H200 train metrics:

- CSV:
  `scripts/outputs/image_official/lewm_flow_2x2_train_h200_20260603/metrics.csv`
- Metric: one epoch, 8 train batches, 2 validation batches, averaged over seeds
  0 and 1.

| `predictor_arch` | `action_encoder_arch` | n | fit loss mean | fit pred loss mean | val loss mean | val pred loss mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `mlp` | `mlp` | 2 | `0.346582` | `0.070738` | `0.340797` | `0.041543` |
| `mlp` | `flow` | 2 | `0.345693` | `0.068504` | `0.337587` | `0.038300` |
| `flow` | `mlp` | 2 | `0.347257` | `0.072154` | `0.337663` | `0.039430` |
| `flow` | `flow` | 2 | `0.343569` | `0.069079` | `0.335017` | `0.036737` |

Interpretation: in this tiny LeWM train probe, flow+flow has the best validation
loss and validation prediction loss, but the margins are small and this is not
yet an evaluation-success result.
