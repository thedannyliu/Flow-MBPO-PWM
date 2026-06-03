#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
CPU_PARTITION="${CPU_PARTITION:-cpu-small}"
CPU_QOS="${CPU_QOS:-embers}"
GPU_PARTITION="${GPU_PARTITION:-gpu-a100}"
GPU_GRES="${GPU_GRES:-gpu:a100:1}"
GPU_QOS="${GPU_QOS:-embers}"
LOG_DIR="${ROOT}/logs/slurm/image_official"

NEWT_ROOT="${NEWT_ROOT:-/storage/project/r-agarg35-0/eliu354/external_repos/newt}"
NEWT_ENV="${NEWT_ENV:-/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602}"
NEWT_MARKER="${NEWT_MARKER:-${NEWT_ENV}/.newt_official_setup_ok_20260602}"
DATA_ROOT="${DATA_ROOT:-/storage/project/r-agarg35-0/eliu354/external_data}"

if { [[ "${CPU_QOS,,}" == "inferno" ]] || [[ "${GPU_QOS,,}" == "inferno" ]]; } \
  && [[ "${ALLOW_INFERNO_QOS:-0}" != "1" ]]; then
  echo "Error: inferno QOS requires explicit user approval. Use embers." >&2
  exit 1
fi

if [[ ! -f "${NEWT_MARKER}" ]]; then
  echo "Error: NEWT official setup marker is missing: ${NEWT_MARKER}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${DATA_ROOT}/newt_demos"

newt_import_job="$(
  sbatch --parsable \
    --job-name="newt_official_import_config_swig_fix1_20260602" \
    --account="${ACCOUNT}" \
    --partition="${CPU_PARTITION}" \
    --qos="${CPU_QOS}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="24G" \
    --time="00:30:00" \
    --output="${LOG_DIR}/newt_official_import_config_swig_fix1_%j.out" \
    --error="${LOG_DIR}/newt_official_import_config_swig_fix1_%j.err" \
    --wrap="cd ${NEWT_ROOT}/tdmpc2 && \
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 WANDB_MODE=disabled && \
${NEWT_ENV}/bin/python - <<'PY'
import json
from pathlib import Path
from config import Config
from common import MODEL_SIZE, TASK_SET
tasks = json.loads(Path('../tasks.json').read_text())
cfg = Config(task='walker-walk', model_size='B', steps=1000, enable_wandb=False, compile=False, save_video=False, save_agent=False, data_dir='${DATA_ROOT}/newt_demos')
print('newt_import_config_ok')
print('task_count', len(tasks))
print('model_sizes', sorted(MODEL_SIZE))
print('walker_action_dim', tasks['walker-walk']['action_dim'])
print('config_task', cfg.task, 'obs', cfg.obs, 'model_size', cfg.model_size, 'num_envs', cfg.num_envs)
print('walker_in_task_set', 'walker-walk' in TASK_SET.get('walker-walk', ['walker-walk']))
PY"
)"

newt_train_job="$(
  sbatch --parsable \
    --job-name="newt_official_walker_swig_fix1_a100_20260602" \
    --account="${ACCOUNT}" \
    --partition="${GPU_PARTITION}" \
    --qos="${GPU_QOS}" \
    --gres="${GPU_GRES}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem="96G" \
    --time="02:00:00" \
    --output="${LOG_DIR}/newt_official_walker_swig_fix1_%j.out" \
    --error="${LOG_DIR}/newt_official_walker_swig_fix1_%j.err" \
    --wrap="cd ${NEWT_ROOT}/tdmpc2 && \
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 WANDB_MODE=disabled HYDRA_FULL_ERROR=1 && \
${NEWT_ENV}/bin/python train.py task=walker-walk model_size=B steps=1000 seed=0 enable_wandb=false save_video=false save_agent=false compile=false num_envs=1 batch_size=16 buffer_size=10000 eval_episodes=1 exp_name=official_walker_smoke_swig_fix1_20260602 data_dir=${DATA_ROOT}/newt_demos"
)"

cat <<EOF
newt_import_job=${newt_import_job}
newt_train_job=${newt_train_job}
EOF
