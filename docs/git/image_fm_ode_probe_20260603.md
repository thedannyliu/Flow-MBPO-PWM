# NEWT And LeWM Flow-Matching ODE Probe - 2026-06-03

## Purpose

This probe separates the earlier residual-flow architecture tests from a stricter flow-matching / ODE-integration pipeline.

- Prior residual-flow tests used gated residual MLP blocks and did not train a velocity field.
- This probe uses a time-conditioned velocity field, ODE endpoint integration, and an explicit flow-matching loss where the training loop exposes the required target embeddings.

## Implemented Pipelines

### NEWT Dynamics

Files:

- `scripts/experiments/image_official/flow_variants/newt_fm_ode_patch.py`
- `scripts/experiments/image_official/newt_fm_ode_site/sitecustomize.py`
- `scripts/experiments/image_official/submit_newt_fm_ode_dynamics_20260603.sh`

Scope:

- Replaces the upstream NEWT one-step latent dynamics head with `FlowMatchingODEDynamics`.
- Integrates the latent endpoint with Euler or Heun; submitted runs use Heun with 4 substeps.
- Adds flow-matching loss on random linear interpolants plus endpoint consistency loss.
- Leaves policy architecture unchanged, so this is a dynamics-only fm_ode versus MLP comparison.

Submitted comparison:

- Task: `walker-run`
- Seeds: `0, 1`
- Steps: `5000`
- Model size: `B`
- Eval episodes: `3`
- W&B: disabled

### LeWorldModel Predictor

Files:

- `scripts/experiments/image_official/flow_variants/lewm_flow_matching.py`
- `scripts/experiments/image_official/run_lewm_fm_ode_train.py`
- `scripts/experiments/image_official/submit_lewm_fm_ode_train_20260603.sh`

Scope:

- Adds `ODEARPredictor`, a time-conditioned autoregressive velocity predictor.
- Adds `FlowMatchingJEPA.predict()`, which integrates the predictor velocity field to produce the next embedding endpoint.
- Adds `FlowMatchingJEPA.flow_matching_loss()`, trained in the custom wrapper alongside endpoint prediction and SIGReg.
- MLP baseline uses the same wrapper with the original `jepa.JEPA` and `module.ARPredictor`; flow-matching loss is zero for that baseline.

Submitted comparison:

- Dataset/task: PushT expert HDF5, `pusht_expert_train.h5`
- Seeds: `0, 1`
- Epochs: `1`
- Train batches: `8`
- Val batches: `2`
- Batch size: `8`
- ODE: Heun with 4 substeps for fm_ode rows
- W&B: disabled

## Local Validation

Committed setup SHA:

- `0dfd05b Add image flow-matching ODE probes`

Validation commands completed before submission:

```bash
bash -n scripts/experiments/image_official/submit_newt_fm_ode_dynamics_20260603.sh
bash -n scripts/experiments/image_official/submit_lewm_fm_ode_train_20260603.sh
python -m py_compile \
  scripts/experiments/image_official/run_lewm_fm_ode_train.py \
  scripts/experiments/image_official/flow_variants/lewm_flow_matching.py \
  scripts/experiments/image_official/flow_variants/newt_fm_ode_patch.py \
  scripts/experiments/image_official/export_flow_2x2_metrics.py \
  scripts/experiments/image_official/newt_fm_ode_site/sitecustomize.py
```

Additional checks:

- NEWT `sitecustomize` patch check returned `True True` for patched `WorldModel` and `TDMPC2`.
- LeWM ODE predictor shape check returned output shape `torch.Size([4, 3, 8])` and finite flow-matching loss.
- LeWM Hydra override check resolved `FlowMatchingJEPA`, `ODEARPredictor`, `ode_substeps=2`, `loss.flow_matching.weight=1.0`, and `num_steps=4`.
- NEWT `TensorDict` return-format compatibility check succeeded.

## Slurm Submissions

All jobs use QOS `embers`, W&B disabled, and distinct output roots.

| Job ID | Track | GPU | Label | State at first poll | Manifest |
| --- | --- | --- | --- | --- | --- |
| `9415949` | NEWT fm_ode dynamics | H200 | `newt_fm_ode_dynamics_h200_20260603` | Pending, priority | `scripts/outputs/image_official/newt_fm_ode_dynamics_h200_20260603/manifest.csv` |
| `9415950` | NEWT fm_ode dynamics | H100 backup | `newt_fm_ode_dynamics_h100_backup_20260603` | Rows 0/1 running, rows 2/3 pending | `scripts/outputs/image_official/newt_fm_ode_dynamics_h100_backup_20260603/manifest.csv` |
| `9415956` | NEWT fm_ode dynamics | A100 backup | `newt_fm_ode_dynamics_a100_backup_20260603` | Pending, priority | `scripts/outputs/image_official/newt_fm_ode_dynamics_a100_backup_20260603/manifest.csv` |
| `9415951` | LeWM fm_ode predictor | H200 | `lewm_fm_ode_train_h200_20260603` | Pending, priority | `scripts/outputs/image_official/lewm_fm_ode_train_h200_20260603/manifest.csv` |
| `9415952` | LeWM fm_ode predictor | H100 backup | `lewm_fm_ode_train_h100_backup_20260603` | Pending, priority | `scripts/outputs/image_official/lewm_fm_ode_train_h100_backup_20260603/manifest.csv` |
| `9415955` | LeWM fm_ode predictor | A100 backup | `lewm_fm_ode_train_a100_backup_20260603` | Row 0 running, rows 1/3 pending | `scripts/outputs/image_official/lewm_fm_ode_train_a100_backup_20260603/manifest.csv` |

Early log check:

- NEWT H100 rows `9415950_0` and `9415950_1` reached Python startup with only expected `pynvml` and Gym deprecation warnings.
- LeWM A100 row `9415955_0` reached Slurm prolog with no stderr at the first log poll.

## LeWM Fix 1

The first LeWM arrays exposed a submitter bug:

- Affected jobs: `9415951`, `9415952`, `9415955`.
- Failure: MLP baseline rows received `+model.ode_substeps` and `+model.ode_integrator`.
- Root cause: those overrides are constructor args for `FlowMatchingJEPA`, but the baseline target is upstream `jepa.JEPA`, whose constructor does not accept them.
- Observed failures: `9415955_0` and `9415952_0` failed with `TypeError: JEPA.__init__() got an unexpected keyword argument 'ode_substeps'`.
- Action: cancelled the affected LeWM arrays with `scancel 9415951 9415952 9415955`.
- Fix: `scripts/experiments/image_official/submit_lewm_fm_ode_train_20260603.sh` now passes ODE model overrides only for `predictor_arch=fm_ode`.

NEWT jobs were not affected by this LeWM submitter bug.

Fix1 replacement submissions:

| Job ID | Track | GPU | Label | State at first poll | Manifest |
| --- | --- | --- | --- | --- | --- |
| `9416042` | LeWM fm_ode predictor fix1 | H100 backup | `lewm_fm_ode_train_h100_fix1_20260603` | Pending, priority | `scripts/outputs/image_official/lewm_fm_ode_train_h100_fix1_20260603/manifest.csv` |
| `9416043` | LeWM fm_ode predictor fix1 | A100 backup | `lewm_fm_ode_train_a100_fix1_20260603` | Pending, priority | `scripts/outputs/image_official/lewm_fm_ode_train_a100_fix1_20260603/manifest.csv` |
| `9416044` | LeWM fm_ode predictor fix1 | H200 | `lewm_fm_ode_train_h200_fix1_20260603` | Pending, priority | `scripts/outputs/image_official/lewm_fm_ode_train_h200_fix1_20260603/manifest.csv` |

NEWT early completed rows:

- `9415950_0` H100 MLP seed 0 completed with final train return `58.669` and success `0.059`.
- `9415950_1` H100 fm_ode seed 0 completed with final train return `23.504` and success `0.024`; log confirms `FlowMatchingODEDynamics(latent_dim=512, condition_dim=16, substeps=4, integrator=heun)`.

## Metrics Export

Use the updated exporter after jobs finish:

```bash
python scripts/experiments/image_official/export_flow_2x2_metrics.py \
  --kind newt \
  --manifest scripts/outputs/image_official/newt_fm_ode_dynamics_h100_backup_20260603/manifest.csv \
  --run-label newt_fm_ode_dynamics_h100_backup_20260603 \
  --job-id 9415950 \
  --output scripts/outputs/image_official/newt_fm_ode_dynamics_h100_backup_20260603/metrics.csv

python scripts/experiments/image_official/export_flow_2x2_metrics.py \
  --kind lewm \
  --manifest scripts/outputs/image_official/lewm_fm_ode_train_a100_backup_20260603/manifest.csv \
  --run-label lewm_fm_ode_train_a100_backup_20260603 \
  --job-id 9415955 \
  --output scripts/outputs/image_official/lewm_fm_ode_train_a100_backup_20260603/metrics.csv
```

Primary result gates:

- NEWT: compare MLP dynamics versus fm_ode dynamics under matched seed/task/steps by final train return and any eval lines printed by the official trainer.
- LeWM: compare MLP predictor versus fm_ode predictor under matched seed/batches by validation prediction loss, flow-matching loss, SIGReg loss, and total loss.

## H200 Results Exported

NEWT H200 primary metrics:

- CSV: `scripts/outputs/image_official/newt_fm_ode_dynamics_h200_20260603/metrics.csv`
- `mlp` seed 0: final train return `19.541`, success `0.020`.
- `fm_ode` seed 0: final train return `32.611`, success `0.033`.
- `mlp` seed 1: final train return `22.362`, success `0.022`.
- `fm_ode` seed 1: final train return `9.903`, success `0.010`.
- Mean final train return: MLP `20.952`, fm_ode `21.257`.
- Interpretation: the strict ODE dynamics path trains end-to-end, but the 2-seed H200 smoke is high variance and does not yet show a reliable improvement over the upstream MLP dynamics.

LeWM H200 fix1 metrics:

- CSV: `scripts/outputs/image_official/lewm_fm_ode_train_h200_fix1_20260603/metrics.csv`
- `mlp` seed 0: validation pred loss `0.037589`, total validation loss `0.335766`.
- `fm_ode` seed 0: validation pred loss `0.047028`, flow-matching validation loss `0.999392`, total validation loss `1.346345`.
- `mlp` seed 1: validation pred loss `0.036997`, total validation loss `0.334957`.
- `fm_ode` seed 1: validation pred loss `0.040036`, flow-matching validation loss `0.999412`, total validation loss `1.337664`.
- Mean validation pred loss: MLP `0.037293`, fm_ode `0.043532`.
- Interpretation: the LeWM flow-matching ODE predictor also trains end-to-end, but this smoke is worse than the MLP predictor on endpoint prediction loss and much worse on total loss because the uncalibrated flow-matching term is about `1.0`.

H100 backup metrics:

- NEWT CSV: `scripts/outputs/image_official/newt_fm_ode_dynamics_h100_backup_20260603/metrics.csv`
- NEWT MLP final train returns: seed 0 `58.669`, seed 1 `14.516`, mean `36.593`.
- NEWT fm_ode final train returns: seed 0 `23.504`, seed 1 `9.660`, mean `16.582`.
- LeWM CSV: `scripts/outputs/image_official/lewm_fm_ode_train_h100_fix1_20260603/metrics.csv`
- LeWM MLP validation pred losses: seed 0 `0.038915`, seed 1 `0.038387`, mean `0.038651`.
- LeWM fm_ode validation pred losses: seed 0 `0.040414`, seed 1 `0.041816`, mean `0.041115`.
- H100 interpretation: the LeWM negative endpoint-prediction trend repeats; NEWT H100 is also negative for fm_ode dynamics. Combined with H200, this argues for treating the current flow-matching ODE implementation as runnable but not yet beneficial.

## Notes

This is still a smoke/diagnostic comparison, not a formal performance claim. The goal is to determine whether a real flow-matching ODE dynamics or predictor path trains and whether early metrics look better or worse than the official MLP baseline under matched small budgets.
