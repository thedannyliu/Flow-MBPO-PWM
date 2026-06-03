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

