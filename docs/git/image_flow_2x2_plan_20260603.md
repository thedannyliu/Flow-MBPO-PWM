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
