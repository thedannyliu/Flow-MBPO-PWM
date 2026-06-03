#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
GPU_QOS="${GPU_QOS:-embers}"
GPU_TYPE="${GPU_TYPE:-h200}"
RUN_LABEL="${RUN_LABEL:-${GPU_TYPE}_structfix_20260603}"
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
mkdir -p "${STABLEWM_HOME}/datasets"
ln -sfn ../pusht_expert_train.h5 "${STABLEWM_HOME}/datasets/pusht_expert_train.h5"
if [[ ! -e "${STABLEWM_HOME}/datasets/pusht_expert_train.h5" ]]; then
  echo "Error: LeWM dataset compatibility symlink is missing: ${STABLEWM_HOME}/datasets/pusht_expert_train.h5" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

lewm_train_struct_fix_job="$(
  sbatch --parsable \
    --job-name="lewm_official_pusht_train_${RUN_LABEL}" \
    --account="${ACCOUNT}" \
    --partition="gpu-${GPU_TYPE}" \
    --qos="${GPU_QOS}" \
    --gres="gpu:${GPU_TYPE}:1" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="96G" \
    --time="02:00:00" \
    --array="0-1%2" \
    --output="${LOG_DIR}/lewm_official_pusht_train_${RUN_LABEL}_%A_%a.out" \
    --error="${LOG_DIR}/lewm_official_pusht_train_${RUN_LABEL}_%A_%a.err" \
    --export=ALL,LEWM_ROOT="${LEWM_ROOT}",LEWM_ENV="${LEWM_ENV}",STABLEWM_HOME="${STABLEWM_HOME}",COMPAT_ROOT="${COMPAT_ROOT}",RUN_LABEL="${RUN_LABEL}" \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
seed="${SLURM_ARRAY_TASK_ID}"
cd "${LEWM_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${COMPAT_ROOT}:${LEWM_ROOT}"
export STABLEWM_HOME="${STABLEWM_HOME}"
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
"${LEWM_ENV}/bin/python" train.py \
  data=pusht \
  data.dataset.name=pusht_expert_train.h5 \
  "seed=${seed}" \
  "subdir=official_train_smoke_${RUN_LABEL}_seed${seed}" \
  "output_model_name=lewm_train_smoke_${RUN_LABEL}_seed${seed}" \
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
)"

cat <<EOF
lewm_train_struct_fix_job=${lewm_train_struct_fix_job}
EOF
