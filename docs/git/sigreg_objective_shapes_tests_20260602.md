# SIGReg Objective, Tensor Shapes, And Tests

Date: 2026-06-02

Purpose: satisfy the active plan requirement to document the SIGReg objective,
expected tensor shapes, and tests before adding new SIGReg experiment jobs.

## Source Inspection

Official LeWM source inspected in a temporary clone:

```text
repo: https://github.com/lucas-maes/le-wm
commit: 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
files:
  /tmp/le-wm-inspect/module.py
  /tmp/le-wm-inspect/train.py
  /tmp/le-wm-inspect/config/train/lewm.yaml
```

LeWM training uses:

```text
emb = output["emb"]                    # (B, T, D)
pred_loss = (pred_emb - tgt_emb)^2 mean
sigreg_loss = SIGReg(emb.transpose(0, 1))
loss = pred_loss + lambda * sigreg_loss
default lambda = 0.09
default knots = 17
default num_proj = 1024
```

The LeWM `SIGReg` module expects:

```text
proj shape: (T, B, D)
T: sequence/time dimension
B: batch dimension used for the empirical characteristic function
D: latent dimension
```

It samples normalized random projection directions `A` with shape `(D, P)`,
projects latents to `(T, B, P)`, evaluates cosine/sine characteristic values at
Gaussian quadrature knots in `[0, 3]`, compares them to the standard-normal
characteristic function `exp(-t^2 / 2)`, weights the squared error by a Gaussian
window and trapezoid weights, multiplies by `B`, and averages over time and
projection directions.

## Mapping To This Repository

State-based MJLab world-model feasibility code already has predicted rollout
states in LeWM-compatible shape:

```text
scripts/experiments/mjlab_qs/run_phaseA_wm_feasibility.py
rollout_predicted_states(model, z, a) -> torch.stack(states, dim=0)
shape: (H + 1, B, state_dim)
```

For current state-based models, the "latent" is the normalized physical state
or model latent used by the runner. The first supported SIGReg input should be:

```text
embeddings: torch.Tensor with shape (T, B, D)
T >= 1
B >= 2 for a meaningful empirical characteristic function
D >= 1
```

Do not silently accept `(B, T, D)` in the core utility. Callers that hold
batch-major tensors should transpose explicitly so shape mistakes are visible.

## Local Utility Contract

The local utility should provide:

```text
SIGRegLoss(num_knots=17, num_projections=1024, t_max=3.0, bandwidth=1.0)
sigreg_loss(embeddings, ...)
add_sigreg_loss(base_loss, embeddings, weight, regularizer)
latent_gaussian_stats(embeddings)
```

The weighted helper must be a no-op when `weight == 0`, returning the base loss
unchanged plus a zero SIGReg scalar. This matters for exact no-SIGReg ablations.

## Required Tests

CPU tests before any new SIGReg GPU job:

```text
finite loss:
  random (T, B, D) embeddings produce a finite nonnegative scalar

gradients:
  loss.backward() produces finite gradients on embeddings

zero/constant anti-collapse:
  zero or constant embeddings produce a nonzero penalty

zero-weight no-op:
  add_sigreg_loss(..., weight=0) leaves the base loss and gradients unchanged

latent variance/isotropy:
  latent_gaussian_stats reports finite variance and off-diagonal covariance metrics
```

## Claim Boundary

SIGReg can improve latent statistics or prediction loss without improving real
rollout. A SIGReg row remains diagnostic until it includes the same real eval,
episode length, fall rate, final/best checkpoints, and video/W&B evidence as the
non-SIGReg baseline.

## Verification

2026-06-02 continuation check after commit `405d8f3`:

```text
command: pytest -q tests/test_sigreg.py
result: 5 passed in 27.61s
coverage: finite loss, finite gradients, constant-latent anti-collapse penalty,
zero-weight no-op, and latent variance/isotropy diagnostics.
```
