#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ENV_CONFIG=""
OUTPUT=""
METADATA_OUTPUT=""
GPU_TYPE="H100"
TIME_LIMIT="02:00:00"
MEMORY="64G"
CPUS=8
ACCOUNT=""
PARTITION_OVERRIDE=""
PYTHON_BIN="python"
CONDA_ENV="${CONDA_ENV_NAME:-}"
NUM_ENVS=16
TARGET_EPISODES=48
EPISODE_LENGTH=128
WINDOW_LENGTH=8
WINDOW_STRIDE=1
MAX_WINDOWS=256
ACTION_MODE="random_uniform"
TEACHER_ALG_CONFIG=""
TEACHER_CHECKPOINT=""
TEACHER_DETERMINISTIC=0
MIXED_TEACHER_PROB="0.5"
SEED=0
WANDB_PROJECT="flow-mbpo-phase1-dataset-variants"
WANDB_GROUP="phase1_dataset_collection"
WANDB_NAME=""
WANDB_TAGS="phase1,dataset_collection,mjlab,world_model"
DISABLE_WANDB=0
JOB_NAME="phase1_dataset"

usage() {
  cat <<'EOF'
Usage:
  submit_phase1_dataset_job.sh --env-config <path> --output <dataset.pt> [options]

Required:
  --env-config PATH
  --output PATH

Options:
  --metadata-output PATH
  --gpu-type {H100|H200|A100|L40S}
  --time HH:MM:SS
  --mem SIZE
  --cpus N
  --account NAME
  --partition NAME
  --python-bin PATH
  --conda-env NAME
  --num-envs N
  --target-episodes N
  --episode-length N
  --window-length N
  --window-stride N
  --max-windows N
  --action-mode {random_uniform|zero|teacher_policy|mixed_episode}
  --teacher-alg-config PATH
  --teacher-checkpoint PATH
  --teacher-deterministic
  --mixed-teacher-prob FLOAT
  --seed N
  --wandb-project NAME
  --wandb-group NAME
  --wandb-name NAME
  --wandb-tags CSV
  --disable-wandb
  --job-name NAME
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-config) ENV_CONFIG="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --metadata-output) METADATA_OUTPUT="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --partition) PARTITION_OVERRIDE="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --target-episodes) TARGET_EPISODES="$2"; shift 2 ;;
    --episode-length) EPISODE_LENGTH="$2"; shift 2 ;;
    --window-length) WINDOW_LENGTH="$2"; shift 2 ;;
    --window-stride) WINDOW_STRIDE="$2"; shift 2 ;;
    --max-windows) MAX_WINDOWS="$2"; shift 2 ;;
    --action-mode) ACTION_MODE="$2"; shift 2 ;;
    --teacher-alg-config) TEACHER_ALG_CONFIG="$2"; shift 2 ;;
    --teacher-checkpoint) TEACHER_CHECKPOINT="$2"; shift 2 ;;
    --teacher-deterministic) TEACHER_DETERMINISTIC=1; shift 1 ;;
    --mixed-teacher-prob) MIXED_TEACHER_PROB="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --wandb-name) WANDB_NAME="$2"; shift 2 ;;
    --wandb-tags) WANDB_TAGS="$2"; shift 2 ;;
    --disable-wandb) DISABLE_WANDB=1; shift 1 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${ENV_CONFIG}" || -z "${OUTPUT}" ]]; then
  echo "Error: --env-config and --output are required."
  exit 1
fi

if [[ -z "${METADATA_OUTPUT}" ]]; then
  METADATA_OUTPUT="${OUTPUT%.pt}.json"
fi

case "${GPU_TYPE}" in
  H100) PARTITION="ice-gpu"; GRES="gpu:h100:1"; DEFAULT_ACCOUNT="coc" ;;
  H200) PARTITION="ice-gpu"; GRES="gpu:h200:1"; DEFAULT_ACCOUNT="coc" ;;
  A100) PARTITION="ice-gpu"; GRES="gpu:a100:1"; DEFAULT_ACCOUNT="coc" ;;
  L40S) PARTITION="ice-gpu"; GRES="gpu:l40s:1"; DEFAULT_ACCOUNT="coc" ;;
  *) echo "Unsupported gpu type: ${GPU_TYPE}"; exit 1 ;;
esac

if [[ -z "${ACCOUNT}" ]]; then
  ACCOUNT="${DEFAULT_ACCOUNT}"
fi
if [[ -n "${PARTITION_OVERRIDE}" ]]; then
  PARTITION="${PARTITION_OVERRIDE}"
fi

mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${METADATA_OUTPUT}")"

LOG_DIR="${PROJECT_ROOT}/logs/slurm/world_model_phase1/${JOB_NAME}"
mkdir -p "${LOG_DIR}"

WRAP_CMD="cd ${PROJECT_ROOT}"
if [[ -n "${CONDA_ENV}" ]]; then
  WRAP_CMD+=" && source ~/.bashrc && conda activate ${CONDA_ENV}"
fi
WRAP_CMD+=" && export PYTHONPATH=${PROJECT_ROOT}/src:\$PYTHONPATH"
WRAP_CMD+=" && ${PYTHON_BIN} scripts/experiments/world_model_phase1/collect_mjlab_phase1_dataset.py"
WRAP_CMD+=" --env-config ${ENV_CONFIG}"
WRAP_CMD+=" --output ${OUTPUT}"
WRAP_CMD+=" --metadata-output ${METADATA_OUTPUT}"
WRAP_CMD+=" --device cuda:0"
WRAP_CMD+=" --seed ${SEED}"
WRAP_CMD+=" --num-envs ${NUM_ENVS}"
WRAP_CMD+=" --target-episodes ${TARGET_EPISODES}"
WRAP_CMD+=" --episode-length ${EPISODE_LENGTH}"
WRAP_CMD+=" --window-length ${WINDOW_LENGTH}"
WRAP_CMD+=" --window-stride ${WINDOW_STRIDE}"
WRAP_CMD+=" --max-windows ${MAX_WINDOWS}"
WRAP_CMD+=" --action-mode ${ACTION_MODE}"
if [[ -n "${TEACHER_ALG_CONFIG}" ]]; then
  WRAP_CMD+=" --teacher-alg-config ${TEACHER_ALG_CONFIG}"
fi
if [[ -n "${TEACHER_CHECKPOINT}" ]]; then
  WRAP_CMD+=" --teacher-checkpoint ${TEACHER_CHECKPOINT}"
fi
if [[ "${TEACHER_DETERMINISTIC}" -eq 1 ]]; then
  WRAP_CMD+=" --teacher-deterministic"
fi
WRAP_CMD+=" --mixed-teacher-prob ${MIXED_TEACHER_PROB}"
WRAP_CMD+=" --wandb-project ${WANDB_PROJECT}"
WRAP_CMD+=" --wandb-group ${WANDB_GROUP}"
if [[ -n "${WANDB_NAME}" ]]; then
  WRAP_CMD+=" --wandb-name ${WANDB_NAME}"
fi
WRAP_CMD+=" --wandb-tags ${WANDB_TAGS}"
if [[ "${DISABLE_WANDB}" -eq 1 ]]; then
  WRAP_CMD+=" --disable-wandb"
fi

sbatch \
  --job-name="${JOB_NAME}_${GPU_TYPE}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --gres="${GRES}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS}" \
  --mem="${MEMORY}" \
  --time="${TIME_LIMIT}" \
  --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
  --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
  --wrap="${WRAP_CMD}"
