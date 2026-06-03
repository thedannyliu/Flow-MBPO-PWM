#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
GPU_QOS="${GPU_QOS:-embers}"
LOG_DIR="${ROOT}/logs/slurm/image_official"
COMPAT_ROOT="${ROOT}/scripts/experiments/image_official/compat"

NEWT_ROOT="${NEWT_ROOT:-/storage/project/r-agarg35-0/eliu354/external_repos/newt}"
NEWT_ENV="${NEWT_ENV:-/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602}"
NEWT_MARKER="${NEWT_MARKER:-${NEWT_ENV}/.newt_official_setup_ok_20260602}"
LEWM_ROOT="${LEWM_ROOT:-/storage/project/r-agarg35-0/eliu354/external_repos/le-wm}"
LEWM_ENV="${LEWM_ENV:-/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602}"
DATA_ROOT="${DATA_ROOT:-/storage/project/r-agarg35-0/eliu354/external_data}"
STABLEWM_HOME="${STABLEWM_HOME:-${DATA_ROOT}/lewm_stablewm}"

if [[ "${GPU_QOS,,}" == "inferno" && "${ALLOW_INFERNO_QOS:-0}" != "1" ]]; then
  echo "Error: inferno QOS requires explicit user approval. Use embers for GPU jobs." >&2
  exit 1
fi

if [[ ! -f "${NEWT_MARKER}" ]]; then
  echo "Error: NEWT official setup marker is missing: ${NEWT_MARKER}" >&2
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

mkdir -p "${LOG_DIR}" "${DATA_ROOT}/newt_demos"

newt_job="$(
  sbatch --parsable \
    --job-name="newt_official_broad_smoke_a100_20260602" \
    --account="${ACCOUNT}" \
    --partition="${NEWT_GPU_PARTITION:-gpu-a100}" \
    --qos="${GPU_QOS}" \
    --gres="${NEWT_GPU_GRES:-gpu:a100:1}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem="96G" \
    --time="02:30:00" \
    --array="0-15%8" \
    --output="${LOG_DIR}/newt_official_broad_smoke_%A_%a.out" \
    --error="${LOG_DIR}/newt_official_broad_smoke_%A_%a.err" \
    --export=ALL,NEWT_ROOT="${NEWT_ROOT}",NEWT_ENV="${NEWT_ENV}",DATA_ROOT="${DATA_ROOT}" \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
tasks=(walker-walk walker-run cheetah-run hopper-hop reacher-easy pendulum-swingup cartpole-swingup cup-catch)
task_count="${#tasks[@]}"
seed="$((SLURM_ARRAY_TASK_ID / task_count))"
task="${tasks[$((SLURM_ARRAY_TASK_ID % task_count))]}"
cd "${NEWT_ROOT}/tdmpc2"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
"${NEWT_ENV}/bin/python" train.py \
  "task=${task}" \
  model_size=B \
  steps=500 \
  "seed=${seed}" \
  enable_wandb=false \
  save_video=false \
  save_agent=false \
  compile=false \
  num_envs=1 \
  batch_size=16 \
  buffer_size=10000 \
  eval_episodes=1 \
  "exp_name=official_broad_${task}_seed${seed}_20260602" \
  "data_dir=${DATA_ROOT}/newt_demos"
SBATCH
)"

lewm_eval_job="$(
  sbatch --parsable \
    --job-name="lewm_official_pusht_eval_h100_20260602" \
    --account="${ACCOUNT}" \
    --partition="${LEWM_EVAL_GPU_PARTITION:-gpu-h100}" \
    --qos="${GPU_QOS}" \
    --gres="${LEWM_EVAL_GPU_GRES:-gpu:h100:1}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem="96G" \
    --time="01:00:00" \
    --array="0-5%6" \
    --output="${LOG_DIR}/lewm_official_pusht_eval_%A_%a.out" \
    --error="${LOG_DIR}/lewm_official_pusht_eval_%A_%a.err" \
    --export=ALL,LEWM_ROOT="${LEWM_ROOT}",LEWM_ENV="${LEWM_ENV}",STABLEWM_HOME="${STABLEWM_HOME}",COMPAT_ROOT="${COMPAT_ROOT}" \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
case "${SLURM_ARRAY_TASK_ID}" in
  0) policy="pusht/lewm"; seed=0; horizon=2; name="lewm_seed0_cem_e30_h2" ;;
  1) policy="pusht/lewm"; seed=1; horizon=2; name="lewm_seed1_cem_e30_h2" ;;
  2) policy="pusht/lewm"; seed=2; horizon=2; name="lewm_seed2_cem_e30_h2" ;;
  3) policy="pusht/lewm"; seed=0; horizon=5; name="lewm_seed0_cem_e30_h5" ;;
  4) policy="random"; seed=0; horizon=2; name="random_seed0_e30_h2" ;;
  5) policy="random"; seed=1; horizon=2; name="random_seed1_e30_h2" ;;
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

lewm_train_job="$(
  sbatch --parsable \
    --job-name="lewm_official_pusht_train_smoke_h100_20260602" \
    --account="${ACCOUNT}" \
    --partition="${LEWM_TRAIN_GPU_PARTITION:-gpu-h100}" \
    --qos="${GPU_QOS}" \
    --gres="${LEWM_TRAIN_GPU_GRES:-gpu:h100:1}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem="128G" \
    --time="02:00:00" \
    --array="0-1%2" \
    --output="${LOG_DIR}/lewm_official_pusht_train_smoke_%A_%a.out" \
    --error="${LOG_DIR}/lewm_official_pusht_train_smoke_%A_%a.err" \
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
  "subdir=official_train_smoke_seed${seed}_20260602" \
  "output_model_name=lewm_train_smoke_seed${seed}" \
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
newt_broad_job=${newt_job}
lewm_eval_job=${lewm_eval_job}
lewm_train_job=${lewm_train_job}
EOF
