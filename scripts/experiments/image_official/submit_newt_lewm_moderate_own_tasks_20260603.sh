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

mkdir -p "${LOG_DIR}" "${DATA_ROOT}/newt_demos"

newt_walker_run_job="$(
  sbatch --parsable \
    --job-name="newt_official_walker_run_moderate_20260603" \
    --account="${ACCOUNT}" \
    --partition="gpu-h200" \
    --qos="${GPU_QOS}" \
    --gres="gpu:h200:1" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="96G" \
    --time="03:00:00" \
    --array="0-1%2" \
    --output="${LOG_DIR}/newt_official_walker_run_moderate_%A_%a.out" \
    --error="${LOG_DIR}/newt_official_walker_run_moderate_%A_%a.err" \
    --export=ALL,NEWT_ROOT="${NEWT_ROOT}",NEWT_ENV="${NEWT_ENV}",DATA_ROOT="${DATA_ROOT}" \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
seed="${SLURM_ARRAY_TASK_ID}"
cd "${NEWT_ROOT}/tdmpc2"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
"${NEWT_ENV}/bin/python" train.py \
  task=walker-run \
  model_size=B \
  steps=5000 \
  "seed=${seed}" \
  enable_wandb=false \
  save_video=false \
  save_agent=true \
  compile=false \
  num_envs=1 \
  batch_size=32 \
  buffer_size=50000 \
  eval_episodes=3 \
  "exp_name=official_walker_run_moderate_seed${seed}_20260603" \
  "data_dir=${DATA_ROOT}/newt_demos"
SBATCH
)"

lewm_pusht_eval_job="$(
  sbatch --parsable \
    --job-name="lewm_official_pusht_moderate_eval_20260603" \
    --account="${ACCOUNT}" \
    --partition="gpu-h200" \
    --qos="${GPU_QOS}" \
    --gres="gpu:h200:1" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="64G" \
    --time="01:30:00" \
    --array="0-2%3" \
    --output="${LOG_DIR}/lewm_official_pusht_moderate_eval_%A_%a.out" \
    --error="${LOG_DIR}/lewm_official_pusht_moderate_eval_%A_%a.err" \
    --export=ALL,LEWM_ROOT="${LEWM_ROOT}",LEWM_ENV="${LEWM_ENV}",STABLEWM_HOME="${STABLEWM_HOME}",COMPAT_ROOT="${COMPAT_ROOT}" \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
case "${SLURM_ARRAY_TASK_ID}" in
  0) policy="pusht/lewm"; seed=0; horizon=2; name="lewm_seed0_cem_e100_h2_moderate_20260603" ;;
  1) policy="pusht/lewm"; seed=1; horizon=2; name="lewm_seed1_cem_e100_h2_moderate_20260603" ;;
  2) policy="pusht/lewm"; seed=0; horizon=5; name="lewm_seed0_cem_e100_h5_moderate_20260603" ;;
  *) echo "bad array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
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
  eval.num_eval=12 \
  eval.eval_budget=100 \
  eval.goal_offset_steps=25 \
  "plan_config.horizon=${horizon}" \
  "plan_config.receding_horizon=${horizon}" \
  plan_config.action_block=5 \
  "output.filename=${name}_results.txt"
SBATCH
)"

echo "newt_walker_run_job=${newt_walker_run_job}"
echo "lewm_pusht_eval_job=${lewm_pusht_eval_job}"
