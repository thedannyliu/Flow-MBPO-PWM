#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
GPU_QOS="${GPU_QOS:-embers}"
PARTITION="${PARTITION:-gpu-h200}"
GPU_GRES="${GPU_GRES:-gpu:h200:1}"
GPU_LABEL="${GPU_LABEL:-h200}"
LOG_DIR="${ROOT}/logs/slurm/mjlab_qs/upstream_pwm_full_pipeline"
LOCKED_MJLAB_PYTHON="${LOCKED_MJLAB_PYTHON:-${ROOT}/scripts/experiments/mjlab_qs/locked_mjlab_python.py}"

HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${ROOT}/baselines/PWM/scripts/outputs/2026-06-02/21-35-04}"
POLICY_DIR="${POLICY_DIR:-${HYDRA_RUN_DIR}/logs/upstream_pwm_mjlab_full_longdiag_h200_seed0_20260602}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/scripts/outputs/mjlab_qs/upstream_pwm_full_pipeline_real_eval_smoke_20260602}"

if [[ "${GPU_QOS,,}" == "inferno" && "${ALLOW_INFERNO_QOS:-0}" != "1" ]]; then
  echo "Error: inferno QOS requires explicit user approval. Use embers for GPU jobs." >&2
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

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

sbatch --parsable \
  --job-name="upstream_pwm_mjlab_real_eval_smoke_${GPU_LABEL}_20260602" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${GPU_QOS}" \
  --gres="${GPU_GRES}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem="96G" \
  --time="01:00:00" \
  --array="0-1%2" \
  --output="${LOG_DIR}/upstream_pwm_mjlab_real_eval_smoke_${GPU_LABEL}_%A_%a.out" \
  --error="${LOG_DIR}/upstream_pwm_mjlab_real_eval_smoke_${GPU_LABEL}_%A_%a.err" \
  --export=ALL,ROOT="${ROOT}",LOCKED_MJLAB_PYTHON="${LOCKED_MJLAB_PYTHON}",HYDRA_RUN_DIR="${HYDRA_RUN_DIR}",POLICY_DIR="${POLICY_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
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
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1

"${LOCKED_MJLAB_PYTHON}" "${ROOT}/scripts/experiments/mjlab_qs/eval_upstream_pwm_mjlab_checkpoint.py" \
  --hydra-run-dir "${HYDRA_RUN_DIR}" \
  --policy-checkpoint "${checkpoint}" \
  --output-dir "${OUTPUT_ROOT}/${kind}" \
  --checkpoint-kind "${kind}" \
  --device cuda:0 \
  --eval-episodes 8 \
  --eval-num-envs 16 \
  --max-steps 1000 \
  --seed 0 \
  --baseline-return 45.8491 \
  --baseline-length 594.97 \
  --baseline-fall 0.625 \
  --notes "W&B-disabled real-env eval smoke for full upstream PWM MJLab longdiag checkpoint 9401906."
SBATCH
