#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
GPU_QOS="${GPU_QOS:-embers}"
GPU_TYPE="${GPU_TYPE:-h200}"
RUN_LABEL="${RUN_LABEL:-lewm_flow_2x2_train_${GPU_TYPE}_20260603}"
LOG_DIR="${ROOT}/logs/slurm/image_official"
OUT_DIR="${ROOT}/scripts/outputs/image_official/${RUN_LABEL}"
PATCH_ROOT="${ROOT}/scripts/experiments/image_official"
COMPAT_ROOT="${PATCH_ROOT}/compat"

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

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

MANIFEST="${OUT_DIR}/manifest.csv"
cat >"${MANIFEST}" <<'CSV'
row,task,predictor_arch,action_encoder_arch,seed,max_epochs,limit_train_batches,limit_val_batches
0,pusht,mlp,mlp,0,1,8,2
1,pusht,mlp,flow,0,1,8,2
2,pusht,flow,mlp,0,1,8,2
3,pusht,flow,flow,0,1,8,2
4,pusht,mlp,mlp,1,1,8,2
5,pusht,mlp,flow,1,1,8,2
6,pusht,flow,mlp,1,1,8,2
7,pusht,flow,flow,1,1,8,2
CSV

lewm_flow_train_job="$(
  sbatch --parsable \
    --job-name="lewm_flow_2x2_train_${GPU_TYPE}" \
    --account="${ACCOUNT}" \
    --partition="gpu-${GPU_TYPE}" \
    --qos="${GPU_QOS}" \
    --gres="gpu:${GPU_TYPE}:1" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="96G" \
    --time="02:00:00" \
    --array="0-7%4" \
    --output="${LOG_DIR}/${RUN_LABEL}_%A_%a.out" \
    --error="${LOG_DIR}/${RUN_LABEL}_%A_%a.err" \
    --export=ALL,ROOT="${ROOT}",LEWM_ROOT="${LEWM_ROOT}",LEWM_ENV="${LEWM_ENV}",STABLEWM_HOME="${STABLEWM_HOME}",PATCH_ROOT="${PATCH_ROOT}",COMPAT_ROOT="${COMPAT_ROOT}",OUT_DIR="${OUT_DIR}",RUN_LABEL="${RUN_LABEL}",MANIFEST="${MANIFEST}" \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
row="${SLURM_ARRAY_TASK_ID}"
line="$(awk -F, -v row="${row}" 'NR > 1 && $1 == row {print $0}' "${MANIFEST}")"
if [[ -z "${line}" ]]; then
  echo "missing manifest row ${row}" >&2
  exit 1
fi
IFS=, read -r _ task predictor_arch action_encoder_arch seed max_epochs limit_train_batches limit_val_batches <<<"${line}"
predictor_target="module.ARPredictor"
action_target="module.Embedder"
[[ "${predictor_arch}" == "flow" ]] && predictor_target="flow_variants.lewm_flow_modules.FlowARPredictor"
[[ "${action_encoder_arch}" == "flow" ]] && action_target="flow_variants.lewm_flow_modules.FlowActionEmbedder"
run_name="${RUN_LABEL}_${task}_pred${predictor_arch}_action${action_encoder_arch}_seed${seed}"
model_name="${run_name}"
run_out="${OUT_DIR}/${task}/predictor_${predictor_arch}/action_${action_encoder_arch}/seed_${seed}"
mkdir -p "${run_out}"
cat >"${run_out}/run.json" <<EOF
{"row": ${row}, "task": "${task}", "predictor_arch": "${predictor_arch}", "action_encoder_arch": "${action_encoder_arch}", "seed": ${seed}, "max_epochs": ${max_epochs}, "limit_train_batches": ${limit_train_batches}, "limit_val_batches": ${limit_val_batches}, "output_model_name": "${model_name}", "slurm_job_id": "${SLURM_ARRAY_JOB_ID}", "slurm_array_task_id": "${SLURM_ARRAY_TASK_ID}"}
EOF
cd "${LEWM_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PATCH_ROOT}:${COMPAT_ROOT}:${LEWM_ROOT}:${PYTHONPATH:-}"
export STABLEWM_HOME="${STABLEWM_HOME}"
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
"${LEWM_ENV}/bin/python" train.py \
  data=pusht \
  data.dataset.name=pusht_expert_train.h5 \
  "seed=${seed}" \
  "subdir=${run_name}" \
  "output_model_name=${model_name}" \
  "model.predictor._target_=${predictor_target}" \
  "model.action_encoder._target_=${action_target}" \
  "trainer.max_epochs=${max_epochs}" \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  trainer.precision=32 \
  "+trainer.limit_train_batches=${limit_train_batches}" \
  "+trainer.limit_val_batches=${limit_val_batches}" \
  num_workers=0 \
  loader.batch_size=8 \
  loader.num_workers=0 \
  loader.persistent_workers=false \
  loader.prefetch_factor=null \
  wandb.enabled=false
touch "${run_out}/completed"
SBATCH
)"

cat <<EOF
lewm_flow_train_job=${lewm_flow_train_job}
manifest=${MANIFEST}
output_dir=${OUT_DIR}
EOF

