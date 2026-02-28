#!/bin/bash
#SBATCH --job-name=mjlab_smoke
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --qos=inferno
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --chdir=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
#SBATCH --output=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/logs/slurm/mjlab/smoke_%j.out
#SBATCH --error=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/logs/slurm/mjlab/smoke_%j.err

set -euo pipefail

# Prefer submit dir when available to avoid path-resolution edge cases on Slurm nodes.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts/mjlab" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/logs/slurm/mjlab"
mkdir -p "${LOG_DIR}"

echo "==========================================="
echo "MJLAB SMOKE RUN (SLURM)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "PWD: $(pwd)"
echo "Start Time: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Log Dir: ${LOG_DIR}"
echo "==========================================="

source ~/.bashrc
conda activate pwm
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# Smoke defaults (override via sbatch --export=ALL,VAR=...)
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-120}"   # >100 triggers eval self-check in train_dflex.py
ENV_NAME="${ENV_NAME:-mjlab_walker}"
ALG_NAME="${ALG_NAME:-pwm_5M_baseline_final}"
WANDB_PROJECT="${WANDB_PROJECT:-flow-mbpo-mjlab-smoke}"
WANDB_GROUP="${WANDB_GROUP:-mjlab-smoke}"
WANDB_NAME="${WANDB_NAME:-smoke_${ENV_NAME}_${ALG_NAME}_s${SEED}}"
WANDB_NOTES="${WANDB_NOTES:-PACE slurm smoke run for mjlab integration}"

echo "Config:"
echo "  ENV_NAME=${ENV_NAME}"
echo "  ALG_NAME=${ALG_NAME}"
echo "  SEED=${SEED}"
echo "  NUM_ENVS=${NUM_ENVS}"
echo "  MAX_EPOCHS=${MAX_EPOCHS}"
echo "  WANDB_PROJECT=${WANDB_PROJECT}"

echo "[1/3] Adapter semantics precheck..."
python scripts/mjlab/smoke_adapter_semantics.py

echo "[2/3] Launch training smoke..."
SEED="${SEED}" \
NUM_ENVS="${NUM_ENVS}" \
MAX_EPOCHS="${MAX_EPOCHS}" \
ENV_NAME="${ENV_NAME}" \
ALG_NAME="${ALG_NAME}" \
WANDB_PROJECT="${WANDB_PROJECT}" \
WANDB_GROUP="${WANDB_GROUP}" \
WANDB_NAME="${WANDB_NAME}" \
WANDB_NOTES="${WANDB_NOTES}" \
scripts/mjlab/smoke_train.sh

echo "[3/3] Post-run artifact checks..."
LATEST_RUN_DIR="$(find outputs -mindepth 2 -maxdepth 2 -type d | sort | tail -n 1 || true)"
if [[ -z "${LATEST_RUN_DIR}" ]]; then
  echo "ERROR: Could not find Hydra output directory under outputs/*/*"
  exit 2
fi

echo "Latest Hydra run dir: ${LATEST_RUN_DIR}"
if [[ ! -f "${LATEST_RUN_DIR}/logs/final_policy.pt" ]]; then
  echo "ERROR: Missing final checkpoint: ${LATEST_RUN_DIR}/logs/final_policy.pt"
  exit 3
fi

if [[ ! -f "${LATEST_RUN_DIR}/logs/best_policy.pt" ]]; then
  echo "WARNING: best_policy.pt not found (final_policy exists)."
else
  echo "Found: ${LATEST_RUN_DIR}/logs/best_policy.pt"
fi

echo "Found: ${LATEST_RUN_DIR}/logs/final_policy.pt"
echo "Smoke run SUCCESS at $(date)"
echo "==========================================="
