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
PYTHON_BIN="python"

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
  --python-bin PATH            Python binary (default: python)
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
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
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
    PARTITION="gpu-h100"
    GRES="gpu:H100:1"
    DEFAULT_ACCOUNT="gts-agarg35"
    ;;
  H200)
    PARTITION="gpu-h200"
    GRES="gpu:h200:1"
    DEFAULT_ACCOUNT="gts-agarg35"
    ;;
  L40S)
    PARTITION="gpu-l40s"
    GRES="gpu:l40s:1"
    DEFAULT_ACCOUNT="gts-agarg35-ideas_l40s"
    ;;
  *)
    echo "Error: unsupported GPU type ${GPU_TYPE}"
    exit 1
    ;;
esac

if [[ -z "${ACCOUNT}" ]]; then
  ACCOUNT="${DEFAULT_ACCOUNT}"
fi

PROFILE_LOG_DIR="${PROJECT_ROOT}/logs/slurm/single_task_online/profile"
mkdir -p "${PROFILE_LOG_DIR}"

SBATCH_WRAP=$(cat <<EOF
cd ${PROJECT_ROOT}
source ~/.bashrc
conda activate pwm
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

sbatch \
  --job-name="sto_profile_${ROW_INDEX}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos=inferno \
  --gres="${GRES}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS}" \
  --mem="${MEMORY}" \
  --time="${TIME_LIMIT}" \
  --output="${PROFILE_LOG_DIR}/profile_${ROW_INDEX}_%j.out" \
  --error="${PROFILE_LOG_DIR}/profile_${ROW_INDEX}_%j.err" \
  --wrap="${SBATCH_WRAP}"
