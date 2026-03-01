#!/usr/bin/env bash
# Submit one manifest row with profiler-enabled execution.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MANIFEST=""
ROW_INDEX=""
GPU_TYPE="H100"
TIME_LIMIT="04:00:00"
MEMORY="128G"
CPUS=16
ACCOUNT=""
PARTITION_OVERRIDE=""
QOS="${SLURM_QOS_OVERRIDE:-}"
PYTHON_BIN="python"
CONDA_ENV="${CONDA_ENV_NAME:-}"

usage() {
  cat <<'EOF'
Usage:
  submit_profile_single.sh --manifest <path> --row-index <int> [options]

Required:
  --manifest PATH
  --row-index INT

Options:
  --gpu-type {H100|H200|L40S}  (default: H100)
  --time HH:MM:SS              (default: 04:00:00)
  --mem SIZE                   (default: 128G)
  --cpus N                     (default: 16)
  --account NAME               Account override
  --partition NAME             Slurm partition override
  --qos NAME                   Optional Slurm QoS (default: cluster default)
  --python-bin PATH            Python binary (default: python)
  --conda-env NAME             Optional conda env to activate before launch
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --row-index) ROW_INDEX="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --partition) PARTITION_OVERRIDE="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${MANIFEST}" || -z "${ROW_INDEX}" ]]; then
  usage
  exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Error: manifest not found: ${MANIFEST}"
  exit 1
fi

case "${GPU_TYPE}" in
  H100)
    PARTITION="ice-gpu"
    GRES="gpu:h100:1"
    DEFAULT_ACCOUNT="coc"
    ;;
  H200)
    PARTITION="ice-gpu"
    GRES="gpu:h200:1"
    DEFAULT_ACCOUNT="coc"
    ;;
  L40S)
    PARTITION="ice-gpu"
    GRES="gpu:l40s:1"
    DEFAULT_ACCOUNT="coc"
    ;;
  *)
    echo "Error: unsupported GPU type ${GPU_TYPE}"
    exit 1
    ;;
esac

if [[ -z "${ACCOUNT}" ]]; then
  ACCOUNT="${DEFAULT_ACCOUNT}"
fi
if [[ -n "${PARTITION_OVERRIDE}" ]]; then
  PARTITION="${PARTITION_OVERRIDE}"
fi

PROFILE_LOG_DIR="${PROJECT_ROOT}/logs/slurm/single_task_online/profile"
mkdir -p "${PROFILE_LOG_DIR}"

ACTIVATE_SNIPPET=""
if [[ -n "${CONDA_ENV}" ]]; then
  ACTIVATE_SNIPPET="conda activate ${CONDA_ENV}"
fi

SBATCH_WRAP=$(cat <<EOF
cd ${PROJECT_ROOT}
source ~/.bashrc
${ACTIVATE_SNIPPET}
export PYTHONPATH=${PROJECT_ROOT}/src:\$PYTHONPATH
export ENABLE_ROLLOUT_VIDEO=0

if command -v nsys >/dev/null 2>&1; then
  mkdir -p logs/profiles
  nsys profile --trace=cuda,nvtx,osrt --sample=none \\
    --output logs/profiles/nsys_row${ROW_INDEX}_\${SLURM_JOB_ID} \\
    ${PYTHON_BIN} scripts/experiments/single_task_online/run_manifest_job.py \\
      --manifest ${MANIFEST} --row-index ${ROW_INDEX} --project-root ${PROJECT_ROOT} --python-bin ${PYTHON_BIN}
else
  ${PYTHON_BIN} scripts/experiments/single_task_online/run_manifest_job.py \\
    --manifest ${MANIFEST} --row-index ${ROW_INDEX} --project-root ${PROJECT_ROOT} --python-bin ${PYTHON_BIN}
fi
EOF
)

SBATCH_CMD=(
  sbatch
  --job-name="sto_profile_${ROW_INDEX}"
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --gres="${GRES}"
  --nodes=1
  --ntasks=1
  --cpus-per-task="${CPUS}"
  --mem="${MEMORY}"
  --time="${TIME_LIMIT}"
  --output="${PROFILE_LOG_DIR}/profile_${ROW_INDEX}_%j.out"
  --error="${PROFILE_LOG_DIR}/profile_${ROW_INDEX}_%j.err"
  --wrap="${SBATCH_WRAP}"
)
if [[ -n "${QOS}" ]]; then
  SBATCH_CMD+=(--qos="${QOS}")
fi

"${SBATCH_CMD[@]}"
