#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
GPU_QOS="${GPU_QOS:-embers}"
LOG_DIR="${ROOT}/logs/slurm/image_official"
COMPAT_ROOT="${ROOT}/scripts/experiments/image_official/compat"

LEWM_ROOT="${LEWM_ROOT:-/storage/project/r-agarg35-0/eliu354/external_repos/le-wm}"
LEWM_ENV="${LEWM_ENV:-/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602}"
DATA_ROOT="${DATA_ROOT:-/storage/project/r-agarg35-0/eliu354/external_data}"
STABLEWM_HOME="${STABLEWM_HOME:-${DATA_ROOT}/lewm_stablewm}"

if [[ "${GPU_QOS,,}" == "inferno" && "${ALLOW_INFERNO_QOS:-0}" != "1" ]]; then
  echo "Error: inferno QOS requires explicit user approval. Use embers for GPU jobs." >&2
  exit 1
fi

if [[ ! -x "${LEWM_ENV}/bin/python" ]]; then
  echo "Error: LeWM official env python is missing: ${LEWM_ENV}/bin/python" >&2
  exit 1
fi
if [[ ! -s "${STABLEWM_HOME}/pusht_expert_train.h5" ]]; then
  echo "Error: LeWM PushT dataset is missing: ${STABLEWM_HOME}/pusht_expert_train.h5" >&2
  exit 1
fi
if [[ ! -d "${COMPAT_ROOT}/vendor/hdf5plugin" ]]; then
  echo "Error: repo-local hdf5plugin vendor dir is missing: ${COMPAT_ROOT}/vendor/hdf5plugin" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${STABLEWM_HOME}/datasets"
ln -sfn ../pusht_expert_train.h5 "${STABLEWM_HOME}/datasets/pusht_expert_train.h5"

sbatch --parsable \
  --job-name="lewm_official_pusht_train_hfcachefix_fix1_h200_20260602" \
  --account="${ACCOUNT}" \
  --partition="gpu-h200" \
  --qos="${GPU_QOS}" \
  --gres="gpu:h200:1" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem="96G" \
  --time="02:00:00" \
  --array="0-1%2" \
  --output="${LOG_DIR}/lewm_official_pusht_train_hfcachefix_fix1_h200_%A_%a.out" \
  --error="${LOG_DIR}/lewm_official_pusht_train_hfcachefix_fix1_h200_%A_%a.err" \
  --export=ALL,LEWM_ROOT="${LEWM_ROOT}",LEWM_ENV="${LEWM_ENV}",STABLEWM_HOME="${STABLEWM_HOME}",COMPAT_ROOT="${COMPAT_ROOT}" \
  <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
seed="${SLURM_ARRAY_TASK_ID}"
cd "${LEWM_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${COMPAT_ROOT}:${LEWM_ROOT}"
export STABLEWM_HOME="${STABLEWM_HOME}"
export LOCAL_DATASET_DIR="${STABLEWM_HOME}"
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
"${LEWM_ENV}/bin/python" train.py \
  data=pusht \
  data.dataset.name=pusht_expert_train.h5 \
  "seed=${seed}" \
  "subdir=official_train_smoke_h200_hfcachefix_fix1_seed${seed}_20260602" \
  "output_model_name=lewm_train_smoke_h200_hfcachefix_fix1_seed${seed}" \
  trainer.max_epochs=1 \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  trainer.precision=32 \
  +trainer.limit_train_batches=2 \
  +trainer.limit_val_batches=1 \
  num_workers=0 \
  loader.batch_size=8 \
  loader.num_workers=0 \
  loader.persistent_workers=false \
  loader.prefetch_factor=null \
  wandb.enabled=false
SBATCH
