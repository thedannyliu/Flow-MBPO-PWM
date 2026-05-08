#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MANIFEST=""
GPU_TYPE="H100"
MAX_CONCURRENT=2
TIME_LIMIT="04:00:00"
MEMORY="128G"
CPUS=8
ACCOUNT=""
PARTITION_OVERRIDE=""
PYTHON_BIN="python"
CONDA_ENV="${CONDA_ENV_NAME:-}"
DEPENDENCY=""

usage() {
  cat <<'EOF'
Usage:
  submit_phase1_overfit_array.sh --manifest <path> [options]

Options:
  --gpu-type {H100|H200|A100|L40S|RTX6000|RTX_6000|A40|V100}
  --max-concurrent N
  --time HH:MM:SS
  --mem SIZE
  --cpus N
  --account NAME
  --partition NAME
  --python-bin PATH
  --conda-env NAME
  --dependency SLURM_DEPENDENCY
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --partition) PARTITION_OVERRIDE="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${MANIFEST}" ]]; then
  echo "Error: --manifest is required."
  exit 1
fi

case "${GPU_TYPE}" in
  H100) PARTITION="ice-gpu"; GRES="gpu:h100:1"; DEFAULT_ACCOUNT="coc" ;;
  H200) PARTITION="ice-gpu"; GRES="gpu:h200:1"; DEFAULT_ACCOUNT="coc" ;;
  A100) PARTITION="ice-gpu"; GRES="gpu:a100:1"; DEFAULT_ACCOUNT="coc" ;;
  L40S) PARTITION="ice-gpu"; GRES="gpu:l40s:1"; DEFAULT_ACCOUNT="coc" ;;
  RTX6000|RTX_6000) PARTITION="ice-gpu"; GRES="gpu:rtx_6000:1"; DEFAULT_ACCOUNT="coc" ;;
  A40) PARTITION="ice-gpu"; GRES="gpu:a40:1"; DEFAULT_ACCOUNT="coc" ;;
  V100) PARTITION="ice-gpu"; GRES="gpu:v100:1"; DEFAULT_ACCOUNT="coc" ;;
  *) echo "Unsupported gpu type: ${GPU_TYPE}"; exit 1 ;;
esac

if [[ -z "${ACCOUNT}" ]]; then
  ACCOUNT="${DEFAULT_ACCOUNT}"
fi
if [[ -n "${PARTITION_OVERRIDE}" ]]; then
  PARTITION="${PARTITION_OVERRIDE}"
fi

NUM_ROWS="$("${PYTHON_BIN}" - <<'PY' "${MANIFEST}"
import csv, sys
with open(sys.argv[1], newline='', encoding='utf-8') as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
)"

STAGE_NAME="$("${PYTHON_BIN}" - <<'PY' "${MANIFEST}"
import csv, sys
with open(sys.argv[1], newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print(rows[0]['stage'])
PY
)"

ARRAY_RANGE="0-$((NUM_ROWS - 1))%${MAX_CONCURRENT}"
LOG_DIR="${PROJECT_ROOT}/logs/slurm/world_model_phase1/${STAGE_NAME}"
mkdir -p "${LOG_DIR}"

WRAP_CMD="cd ${PROJECT_ROOT}"
if [[ -n "${CONDA_ENV}" ]]; then
  WRAP_CMD+=" && source ~/.bashrc && conda activate ${CONDA_ENV}"
fi
WRAP_CMD+=" && export PYTHONPATH=${PROJECT_ROOT}/src:\$PYTHONPATH"
WRAP_CMD+=" && ${PYTHON_BIN} scripts/experiments/world_model_phase1/run_phase1_overfit_job.py --manifest ${MANIFEST} --row-index \$SLURM_ARRAY_TASK_ID --python-bin ${PYTHON_BIN} --project-root ${PROJECT_ROOT}"

SBATCH_EXTRA=()
if [[ -n "${DEPENDENCY}" ]]; then
  SBATCH_EXTRA+=(--dependency="${DEPENDENCY}")
fi

sbatch \
  --job-name="wm1_${STAGE_NAME}_${GPU_TYPE}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --gres="${GRES}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS}" \
  --mem="${MEMORY}" \
  --time="${TIME_LIMIT}" \
  --array="${ARRAY_RANGE}" \
  --output="${LOG_DIR}/wm1_${STAGE_NAME}_%A_%a.out" \
  --error="${LOG_DIR}/wm1_${STAGE_NAME}_%A_%a.err" \
  "${SBATCH_EXTRA[@]}" \
  --wrap="${WRAP_CMD}"
