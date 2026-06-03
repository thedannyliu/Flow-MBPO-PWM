#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
GPU_QOS="${GPU_QOS:-embers}"
GPU_TYPE="${GPU_TYPE:-h200}"
PARTITION="${PARTITION:-gpu-${GPU_TYPE}}"
CONDA_ENV="${CONDA_ENV:-pwm}"
RUN_LABEL="${RUN_LABEL:-${GPU_TYPE}_20260603}"
LOG_DIR="${ROOT}/logs/slurm/mjlab_qs/original_pwm_collapse_probe"

if [[ "${GPU_QOS,,}" == "inferno" && "${ALLOW_INFERNO_QOS:-0}" != "1" ]]; then
  echo "Error: inferno QOS requires explicit user approval. Use embers for GPU jobs." >&2
  exit 1
fi

case "${GPU_TYPE}" in
  h200|H200) GRES="gpu:h200:1" ;;
  h100|H100) GRES="gpu:h100:1" ;;
  a100|A100) GRES="gpu:a100:1" ;;
  l40s|L40S) GRES="gpu:l40s:1" ;;
  *) echo "unsupported GPU_TYPE=${GPU_TYPE}" >&2; exit 1 ;;
esac

DATASET="scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.pt"
METADATA="scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.json"
NORMALIZATION="scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16_normalization.json"
BASE_OUT="${BASE_OUT:-scripts/outputs/mjlab_qs/original_pwm_collapse_probe_${RUN_LABEL}}"
BASE_CKPT="scripts/outputs/mjlab_qs/original_pwm_adapter/original_pwm_dataset_replay_locked_diag_20260603/velocity_flat_unitree_g1/locked_replay_qs_core_h16_normobs_normrew/seed_0"

mkdir -p "${LOG_DIR}"

job="$(
  sbatch --parsable \
    --job-name="mjqs_original_pwm_collapse_probe_${RUN_LABEL}" \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --qos="${GPU_QOS}" \
    --gres="${GRES}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="96G" \
    --time="01:30:00" \
    --array="0-2%3" \
    --output="${LOG_DIR}/original_pwm_collapse_probe_${RUN_LABEL}_%A_%a.out" \
    --error="${LOG_DIR}/original_pwm_collapse_probe_${RUN_LABEL}_%A_%a.err" \
    --export=ALL,ROOT="${ROOT}",CONDA_ENV="${CONDA_ENV}",DATASET="${DATASET}",METADATA="${METADATA}",NORMALIZATION="${NORMALIZATION}",BASE_OUT="${BASE_OUT}",BASE_CKPT="${BASE_CKPT}" \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
case "${SLURM_ARRAY_TASK_ID}" in
  0) kind="pretrained"; ckpt="${BASE_CKPT}/pretrained_original_pwm_adapter.pt" ;;
  1) kind="final"; ckpt="${BASE_CKPT}/final_policy_extraction.pt" ;;
  2) kind="best"; ckpt="${BASE_CKPT}/best_policy_extraction.pt" ;;
  *) echo "bad array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac
cd "${ROOT}"
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi
conda activate "${CONDA_ENV}"
export PYTHONPATH="${ROOT}/src:${ROOT}/baselines/PWM/src:${PYTHONPATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
python scripts/experiments/mjlab_qs/analyze_original_pwm_collapse.py \
  --dataset "${DATASET}" \
  --metadata "${METADATA}" \
  --normalization "${NORMALIZATION}" \
  --checkpoint "${ckpt}" \
  --checkpoint-kind "${kind}" \
  --output-dir "${BASE_OUT}/${kind}" \
  --seed 0 \
  --device cuda:0 \
  --batch-size 256 \
  --max-batches 64 \
  --horizon 16 \
  --obs-mode normalized \
  --reward-mode normalized
SBATCH
)"

echo "original_pwm_collapse_probe_job=${job}"
