#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
GPU_QOS="${GPU_QOS:-embers}"
GPU_TYPE="${GPU_TYPE:-h200}"
RUN_LABEL="${RUN_LABEL:-newt_flow_2x2_${GPU_TYPE}_20260603}"
LOG_DIR="${ROOT}/logs/slurm/image_official"
OUT_DIR="${ROOT}/scripts/outputs/image_official/${RUN_LABEL}"
PATCH_ROOT="${ROOT}/scripts/experiments/image_official"
SITE_ROOT="${PATCH_ROOT}/newt_flow_site"

NEWT_ROOT="${NEWT_ROOT:-/storage/project/r-agarg35-0/eliu354/external_repos/newt}"
NEWT_ENV="${NEWT_ENV:-/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602}"
NEWT_MARKER="${NEWT_MARKER:-${NEWT_ENV}/.newt_official_setup_ok_20260602}"
DATA_ROOT="${DATA_ROOT:-/storage/project/r-agarg35-0/eliu354/external_data}"

if [[ "${GPU_QOS,,}" == "inferno" && "${ALLOW_INFERNO_QOS:-0}" != "1" ]]; then
  echo "Error: inferno QOS requires explicit user approval. Use embers for GPU jobs." >&2
  exit 1
fi
if [[ ! -f "${NEWT_MARKER}" ]]; then
  echo "Error: NEWT official setup marker is missing: ${NEWT_MARKER}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${DATA_ROOT}/newt_demos"

MANIFEST="${OUT_DIR}/manifest.csv"
cat >"${MANIFEST}" <<'CSV'
row,task,wm_arch,policy_arch,seed,steps,model_size
0,walker-run,mlp,mlp,0,5000,B
1,walker-run,mlp,flow,0,5000,B
2,walker-run,flow,mlp,0,5000,B
3,walker-run,flow,flow,0,5000,B
4,walker-run,mlp,mlp,1,5000,B
5,walker-run,mlp,flow,1,5000,B
6,walker-run,flow,mlp,1,5000,B
7,walker-run,flow,flow,1,5000,B
CSV

newt_flow_job="$(
  sbatch --parsable \
    --job-name="newt_flow_2x2_${GPU_TYPE}" \
    --account="${ACCOUNT}" \
    --partition="gpu-${GPU_TYPE}" \
    --qos="${GPU_QOS}" \
    --gres="gpu:${GPU_TYPE}:1" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="96G" \
    --time="03:00:00" \
    --array="0-7%4" \
    --output="${LOG_DIR}/${RUN_LABEL}_%A_%a.out" \
    --error="${LOG_DIR}/${RUN_LABEL}_%A_%a.err" \
    --export=ALL,ROOT="${ROOT}",NEWT_ROOT="${NEWT_ROOT}",NEWT_ENV="${NEWT_ENV}",DATA_ROOT="${DATA_ROOT}",PATCH_ROOT="${PATCH_ROOT}",SITE_ROOT="${SITE_ROOT}",OUT_DIR="${OUT_DIR}",RUN_LABEL="${RUN_LABEL}",MANIFEST="${MANIFEST}" \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
row="${SLURM_ARRAY_TASK_ID}"
line="$(awk -F, -v row="${row}" 'NR > 1 && $1 == row {print $0}' "${MANIFEST}")"
if [[ -z "${line}" ]]; then
  echo "missing manifest row ${row}" >&2
  exit 1
fi
IFS=, read -r _ task wm_arch policy_arch seed steps model_size <<<"${line}"
export NEWT_FLOW_WM=0
export NEWT_FLOW_POLICY=0
[[ "${wm_arch}" == "flow" ]] && export NEWT_FLOW_WM=1
[[ "${policy_arch}" == "flow" ]] && export NEWT_FLOW_POLICY=1
export NEWT_FLOW_STEPS="${NEWT_FLOW_STEPS:-2}"
run_name="${RUN_LABEL}_${task}_wm${wm_arch}_policy${policy_arch}_seed${seed}"
run_out="${OUT_DIR}/${task}/wm_${wm_arch}/policy_${policy_arch}/seed_${seed}"
mkdir -p "${run_out}"
cat >"${run_out}/run.json" <<EOF
{"row": ${row}, "task": "${task}", "wm_arch": "${wm_arch}", "policy_arch": "${policy_arch}", "seed": ${seed}, "steps": ${steps}, "model_size": "${model_size}", "slurm_job_id": "${SLURM_ARRAY_JOB_ID}", "slurm_array_task_id": "${SLURM_ARRAY_TASK_ID}"}
EOF
cd "${NEWT_ROOT}/tdmpc2"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${SITE_ROOT}:${PATCH_ROOT}:${NEWT_ROOT}/tdmpc2:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
"${NEWT_ENV}/bin/python" train.py \
  "task=${task}" \
  "model_size=${model_size}" \
  "steps=${steps}" \
  "seed=${seed}" \
  enable_wandb=false \
  save_video=false \
  save_agent=true \
  compile=false \
  num_envs=1 \
  batch_size=32 \
  buffer_size=50000 \
  eval_episodes=3 \
  "exp_name=${run_name}" \
  "data_dir=${DATA_ROOT}/newt_demos"
touch "${run_out}/completed"
SBATCH
)"

cat <<EOF
newt_flow_job=${newt_flow_job}
manifest=${MANIFEST}
output_dir=${OUT_DIR}
EOF

