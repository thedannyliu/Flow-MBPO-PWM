#!/usr/bin/env bash
set -euo pipefail

# Smoke test for mjlab integration through the main Hydra training path.
# Usage:
#   WANDB_PROJECT=flow-mbpo-mjlab-smoke scripts/mjlab/smoke_train.sh

SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
ENV_NAME="${ENV_NAME:-mjlab_walker}"
ALG_NAME="${ALG_NAME:-pwm_5M_baseline_final}"
WANDB_PROJECT="${WANDB_PROJECT:-flow-mbpo-mjlab-smoke}"
WANDB_GROUP="${WANDB_GROUP:-mjlab-smoke}"
WANDB_NAME="${WANDB_NAME:-smoke_${ENV_NAME}_${ALG_NAME}_s${SEED}}"
WANDB_NOTES="${WANDB_NOTES:-mjlab smoke run with adapter diagnostics + profiling}"

echo "Running mjlab smoke training"
echo "  env=${ENV_NAME}"
echo "  alg=${ALG_NAME}"
echo "  seed=${SEED}"
echo "  num_envs=${NUM_ENVS}"
echo "  max_epochs=${MAX_EPOCHS}"
echo "  wandb_project=${WANDB_PROJECT}"

python scripts/train_online.py \
  env="${ENV_NAME}" \
  alg="${ALG_NAME}" \
  general.seed="${SEED}" \
  alg.max_epochs="${MAX_EPOCHS}" \
  env.config.num_envs="${NUM_ENVS}" \
  general.run_wandb=true \
  ++wandb.project="${WANDB_PROJECT}" \
  ++wandb.group="${WANDB_GROUP}" \
  ++wandb.name="${WANDB_NAME}" \
  ++wandb.notes="${WANDB_NOTES}"
