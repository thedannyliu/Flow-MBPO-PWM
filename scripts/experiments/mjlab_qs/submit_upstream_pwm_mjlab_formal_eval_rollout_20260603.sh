#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
GPU_QOS="${GPU_QOS:-embers}"
PARTITION="${PARTITION:-gpu-h200}"
GPU_GRES="${GPU_GRES:-gpu:h200:1}"
GPU_LABEL="${GPU_LABEL:-h200}"
MODE="${MODE:-eval}"
WANDB_PROJECT="${WANDB_PROJECT:-flow-mbpo-mjlab-full-upstream-pwm}"
WANDB_GROUP="${WANDB_GROUP:-upstream_pwm_mjlab_full_pipeline_20260603}"
WANDB_MODE_VALUE="${WANDB_MODE_VALUE:-online}"
LOG_DIR="${ROOT}/logs/slurm/mjlab_qs/upstream_pwm_full_pipeline"
LOCKED_MJLAB_PYTHON="${LOCKED_MJLAB_PYTHON:-${ROOT}/scripts/experiments/mjlab_qs/locked_mjlab_python.py}"
SUBMIT_GIT_SHA="$(git -C "${ROOT}" rev-parse HEAD)"
SUBMIT_GIT_BRANCH="$(git -C "${ROOT}" rev-parse --abbrev-ref HEAD)"

HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${ROOT}/baselines/PWM/scripts/outputs/2026-06-02/21-35-04}"
POLICY_DIR="${POLICY_DIR:-${HYDRA_RUN_DIR}/logs/upstream_pwm_mjlab_full_longdiag_h200_seed0_20260602}"

if [[ "${GPU_QOS,,}" == "inferno" && "${ALLOW_INFERNO_QOS:-0}" != "1" ]]; then
  echo "Error: inferno QOS requires explicit user approval. Use embers for GPU jobs." >&2
  exit 1
fi

if [[ "${MODE}" != "eval" && "${MODE}" != "rollout" ]]; then
  echo "Error: MODE must be eval or rollout, got '${MODE}'." >&2
  exit 1
fi

if [[ ! -s "${HYDRA_RUN_DIR}/.hydra/config.yaml" ]]; then
  echo "Error: missing Hydra config: ${HYDRA_RUN_DIR}/.hydra/config.yaml" >&2
  exit 1
fi
if [[ ! -s "${POLICY_DIR}/final_policy.pt" ]]; then
  echo "Error: missing final policy: ${POLICY_DIR}/final_policy.pt" >&2
  exit 1
fi
if [[ ! -s "${POLICY_DIR}/best_policy.pt" ]]; then
  echo "Error: missing best policy: ${POLICY_DIR}/best_policy.pt" >&2
  exit 1
fi

if [[ "${MODE}" == "eval" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_formal_eval40_wandb_${GPU_LABEL}_20260603}"
  JOB_NAME="upstream_pwm_mjlab_eval40_wandb_${GPU_LABEL}_20260603"
  TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_rollout10_wandb_${GPU_LABEL}_20260603}"
  JOB_NAME="upstream_pwm_mjlab_roll10_wandb_${GPU_LABEL}_20260603"
  TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

sbatch --parsable \
  --job-name="${JOB_NAME}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${GPU_QOS}" \
  --gres="${GPU_GRES}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS_PER_TASK:-8}" \
  --mem="${MEM:-128G}" \
  --time="${TIME_LIMIT}" \
  --array="0-1%2" \
  --output="${LOG_DIR}/${JOB_NAME}_%A_%a.out" \
  --error="${LOG_DIR}/${JOB_NAME}_%A_%a.err" \
  --export=ALL,ROOT="${ROOT}",LOCKED_MJLAB_PYTHON="${LOCKED_MJLAB_PYTHON}",HYDRA_RUN_DIR="${HYDRA_RUN_DIR}",POLICY_DIR="${POLICY_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}",MODE="${MODE}",WANDB_PROJECT="${WANDB_PROJECT}",WANDB_GROUP="${WANDB_GROUP}",WANDB_MODE_VALUE="${WANDB_MODE_VALUE}",FLOW_MBPO_SUBMIT_GIT_SHA="${SUBMIT_GIT_SHA}",FLOW_MBPO_SUBMIT_GIT_BRANCH="${SUBMIT_GIT_BRANCH}" \
  <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

case "${SLURM_ARRAY_TASK_ID}" in
  0) kind="final"; checkpoint="${POLICY_DIR}/final_policy.pt" ;;
  1) kind="best"; checkpoint="${POLICY_DIR}/best_policy.pt" ;;
  *) echo "Unsupported array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

cd "${ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT}/src:${ROOT}/baselines/PWM/src:${ROOT}/baselines/PWM/external/tdmpc2:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE_VALUE}"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1

if [[ "${MODE}" == "eval" ]]; then
  "${LOCKED_MJLAB_PYTHON}" "${ROOT}/scripts/experiments/mjlab_qs/eval_upstream_pwm_mjlab_checkpoint.py" \
    --hydra-run-dir "${HYDRA_RUN_DIR}" \
    --policy-checkpoint "${checkpoint}" \
    --output-dir "${OUTPUT_ROOT}/${kind}" \
    --checkpoint-kind "${kind}" \
    --device cuda:0 \
    --eval-episodes 40 \
    --eval-num-envs 16 \
    --max-steps 1000 \
    --seed 0 \
    --baseline-return 45.8491 \
    --baseline-length 594.97 \
    --baseline-fall 0.625 \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}_${MODE}" \
    --wandb-name "upstream_pwm_mjlab_${kind}_eval40_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --notes "Formal 40-episode real-env eval for full upstream train_dflex/PWM.train MJLab checkpoint."
else
  "${LOCKED_MJLAB_PYTHON}" "${ROOT}/scripts/experiments/mjlab_qs/render_upstream_pwm_mjlab_checkpoint.py" \
    --hydra-run-dir "${HYDRA_RUN_DIR}" \
    --policy-checkpoint "${checkpoint}" \
    --output-dir "${OUTPUT_ROOT}/${kind}" \
    --checkpoint-kind "${kind}" \
    --device cuda:0 \
    --rollout-episodes 10 \
    --max-steps 1000 \
    --seed 0 \
    --baseline-return 54.1283 \
    --baseline-length 688.40 \
    --baseline-fall 0.400 \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}_${MODE}" \
    --wandb-name "upstream_pwm_mjlab_${kind}_rollout10_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --notes "Formal 10-episode MP4 rollout for full upstream train_dflex/PWM.train MJLab checkpoint."
fi
SBATCH
