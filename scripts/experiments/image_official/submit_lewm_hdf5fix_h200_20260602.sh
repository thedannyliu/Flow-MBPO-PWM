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
if [[ ! -s "${STABLEWM_HOME}/pusht/lewm_object.ckpt" ]]; then
  echo "Error: LeWM PushT object checkpoint is missing: ${STABLEWM_HOME}/pusht/lewm_object.ckpt" >&2
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

mkdir -p "${LOG_DIR}"

lewm_eval_h200_hdf5fix_job="$(
  sbatch --parsable \
    --job-name="lewm_official_pusht_eval_hdf5fix_h200_20260602" \
    --account="${ACCOUNT}" \
    --partition="gpu-h200" \
    --qos="${GPU_QOS}" \
    --gres="gpu:h200:1" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="64G" \
    --time="01:00:00" \
    --array="0-5%3" \
    --output="${LOG_DIR}/lewm_official_pusht_eval_hdf5fix_h200_%A_%a.out" \
    --error="${LOG_DIR}/lewm_official_pusht_eval_hdf5fix_h200_%A_%a.err" \
    --export=ALL,LEWM_ROOT="${LEWM_ROOT}",LEWM_ENV="${LEWM_ENV}",STABLEWM_HOME="${STABLEWM_HOME}",COMPAT_ROOT="${COMPAT_ROOT}" \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
case "${SLURM_ARRAY_TASK_ID}" in
  0) policy="pusht/lewm"; seed=0; horizon=2; name="lewm_seed0_cem_e30_h2_h200_hdf5fix" ;;
  1) policy="pusht/lewm"; seed=1; horizon=2; name="lewm_seed1_cem_e30_h2_h200_hdf5fix" ;;
  2) policy="pusht/lewm"; seed=2; horizon=2; name="lewm_seed2_cem_e30_h2_h200_hdf5fix" ;;
  3) policy="pusht/lewm"; seed=0; horizon=5; name="lewm_seed0_cem_e30_h5_h200_hdf5fix" ;;
  4) policy="random"; seed=0; horizon=2; name="random_seed0_e30_h2_h200_hdf5fix" ;;
  5) policy="random"; seed=1; horizon=2; name="random_seed1_e30_h2_h200_hdf5fix" ;;
  *) echo "Unsupported array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac
cd "${LEWM_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${COMPAT_ROOT}:${LEWM_ROOT}"
export STABLEWM_HOME="${STABLEWM_HOME}"
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
"${LEWM_ENV}/bin/python" eval.py \
  --config-name=pusht \
  "cache_dir=${STABLEWM_HOME}" \
  "policy=${policy}" \
  "seed=${seed}" \
  eval.num_eval=4 \
  eval.eval_budget=30 \
  eval.goal_offset_steps=25 \
  "plan_config.horizon=${horizon}" \
  "plan_config.receding_horizon=${horizon}" \
  plan_config.action_block=5 \
  "output.filename=${name}_results.txt"
SBATCH
)"

lewm_train_h200_hdf5fix_job="$(
  sbatch --parsable \
    --job-name="lewm_official_pusht_train_hdf5fix_h200_20260602" \
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
    --output="${LOG_DIR}/lewm_official_pusht_train_hdf5fix_h200_%A_%a.out" \
    --error="${LOG_DIR}/lewm_official_pusht_train_hdf5fix_h200_%A_%a.err" \
    --export=ALL,LEWM_ROOT="${LEWM_ROOT}",LEWM_ENV="${LEWM_ENV}",STABLEWM_HOME="${STABLEWM_HOME}",COMPAT_ROOT="${COMPAT_ROOT}" \
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
  data.dataset.name=pusht_expert_train \
  "seed=${seed}" \
  "subdir=official_train_smoke_h200_hdf5fix_seed${seed}_20260602" \
  "output_model_name=lewm_train_smoke_h200_hdf5fix_seed${seed}" \
  trainer.max_epochs=1 \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  trainer.precision=32 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=1 \
  num_workers=0 \
  loader.batch_size=8 \
  loader.num_workers=0 \
  loader.persistent_workers=false \
  loader.prefetch_factor=null \
  wandb.enabled=false
SBATCH
)"

cat <<EOF
lewm_eval_h200_hdf5fix_job=${lewm_eval_h200_hdf5fix_job}
lewm_train_h200_hdf5fix_job=${lewm_train_h200_hdf5fix_job}
EOF
