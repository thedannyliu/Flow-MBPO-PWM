#!/usr/bin/env bash
set -euo pipefail

KIND=""
MANIFEST=""
GPU_TYPE="H100"
PARTITION="ice-gpu"
QOS="embers"
ACCOUNT="gts-agarg35"
MAX_CONCURRENT=2
TIME_LIMIT="04:00:00"
MEMORY="128G"
CPUS=8
PYTHON_BIN="python"
CONDA_ENV="${CONDA_ENV_NAME:-}"
DEPENDENCY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind) KIND="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

if [[ -z "$KIND" || -z "$MANIFEST" ]]; then
  echo "--kind and --manifest are required" >&2
  exit 1
fi
if [[ "${QOS,,}" == "inferno" && "${ALLOW_INFERNO_QOS:-0}" != "1" ]]; then
  echo "Error: inferno QOS requires explicit user approval. Use embers for GPU jobs." >&2
  exit 1
fi

case "${GPU_TYPE}" in
  H100) GRES="gpu:h100:1" ;;
  H200) GRES="gpu:h200:1" ;;
  A100) GRES="gpu:a100:1" ;;
  L40S) GRES="gpu:l40s:1" ;;
  RTX6000|RTX_6000) GRES="gpu:rtx_6000:1" ;;
  PRO6000|RTXPRO6000|RTX_PRO_6000|RTX_PRO_6000_BLACKWELL) GRES="gpu:rtx_pro_6000_blackwell:1" ;;
  *) echo "unsupported gpu ${GPU_TYPE}" >&2; exit 1 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NUM_ROWS="$("${PYTHON_BIN}" - <<'PY' "${MANIFEST}"
import csv, sys
with open(sys.argv[1], newline='', encoding='utf-8') as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
)"
ARRAY="0-$((NUM_ROWS - 1))%${MAX_CONCURRENT}"
LOG_DIR="${PROJECT_ROOT}/logs/slurm/mjlab_qs/${KIND}"
mkdir -p "${LOG_DIR}"

RUNNER="scripts/experiments/mjlab_qs/run_collection_row.py"
if [[ "$KIND" == "train" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_training_row.py"
elif [[ "$KIND" == "train_match" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_train_match_row.py"
elif [[ "$KIND" == "policy_extract" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_policy_extraction_row.py"
elif [[ "$KIND" == "policy_rollout" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_policy_rollout_row.py"
elif [[ "$KIND" == "original_pwm_adapter" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_original_pwm_adapter_row.py"
elif [[ "$KIND" == "native_collector" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_mjlab_native_collector_row.py"
elif [[ "$KIND" == "native_collection" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_native_collection_row.py"
fi

WRAP="cd ${PROJECT_ROOT}"
if [[ -n "${CONDA_ENV}" ]]; then
  WRAP+=" && source ~/.bashrc && conda activate ${CONDA_ENV}"
fi
WRAP+=" && export PYTHONPATH=${PROJECT_ROOT}/src:\$PYTHONPATH"
WRAP+=" && export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless"
WRAP+=" && export WANDB_DIR=${PROJECT_ROOT}/scripts/outputs/mjlab_qs/wandb"
WRAP+=" && mkdir -p ${PROJECT_ROOT}/scripts/outputs/mjlab_qs/wandb"
WRAP+=" && ${PYTHON_BIN} ${RUNNER} --manifest ${MANIFEST} --row-index \$SLURM_ARRAY_TASK_ID --python-bin ${PYTHON_BIN}"

SBATCH_ARGS=(
  --job-name="mjqs_${KIND}_${GPU_TYPE}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --gres="${GRES}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS}" \
  --mem="${MEMORY}" \
  --time="${TIME_LIMIT}" \
  --array="${ARRAY}" \
  --output="${LOG_DIR}/mjqs_${KIND}_%A_%a.out" \
  --error="${LOG_DIR}/mjqs_${KIND}_%A_%a.err" \
  --wrap="${WRAP}"
)
if [[ -n "${DEPENDENCY}" ]]; then
  SBATCH_ARGS+=(--dependency="${DEPENDENCY}")
fi

sbatch "${SBATCH_ARGS[@]}"
