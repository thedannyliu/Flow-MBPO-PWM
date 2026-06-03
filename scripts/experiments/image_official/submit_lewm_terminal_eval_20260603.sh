#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
GPU_QOS="${GPU_QOS:-embers}"
GPU_TYPE="${GPU_TYPE:-h200}"
RUN_LABEL="${RUN_LABEL:-lewm_terminal_eval_${GPU_TYPE}_20260603}"
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
row,family,variant,seed,policy,eval_num,eval_budget,goal_offset,horizon,receding,action_block,num_samples,n_steps,topk
0,fm_ode,mlp,0,lewm_fm_ode_train_h200_fix1_20260603_pusht_predmlp_seed0/weights_epoch_1.pt,4,20,20,2,2,5,24,8,6
1,fm_ode,fm_ode,0,lewm_fm_ode_train_h200_fix1_20260603_pusht_predfm_ode_seed0/weights_epoch_1.pt,4,20,20,2,2,5,24,8,6
2,fm_ode,mlp,1,lewm_fm_ode_train_h200_fix1_20260603_pusht_predmlp_seed1/weights_epoch_1.pt,4,20,20,2,2,5,24,8,6
3,fm_ode,fm_ode,1,lewm_fm_ode_train_h200_fix1_20260603_pusht_predfm_ode_seed1/weights_epoch_1.pt,4,20,20,2,2,5,24,8,6
4,residual_2x2,predmlp_actionmlp,0,lewm_flow_2x2_train_h200_20260603_pusht_predmlp_actionmlp_seed0/weights_epoch_1.pt,4,20,20,2,2,5,24,8,6
5,residual_2x2,predflow_actionflow,0,lewm_flow_2x2_train_h200_20260603_pusht_predflow_actionflow_seed0/weights_epoch_1.pt,4,20,20,2,2,5,24,8,6
6,residual_2x2,predmlp_actionmlp,1,lewm_flow_2x2_train_h200_20260603_pusht_predmlp_actionmlp_seed1/weights_epoch_1.pt,4,20,20,2,2,5,24,8,6
7,residual_2x2,predflow_actionflow,1,lewm_flow_2x2_train_h200_20260603_pusht_predflow_actionflow_seed1/weights_epoch_1.pt,4,20,20,2,2,5,24,8,6
CSV

lewm_terminal_eval_job="$(
  sbatch --parsable \
    --job-name="lewm_term_eval_${GPU_TYPE}" \
    --account="${ACCOUNT}" \
    --partition="gpu-${GPU_TYPE}" \
    --qos="${GPU_QOS}" \
    --gres="gpu:${GPU_TYPE}:1" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="96G" \
    --time="01:00:00" \
    --array="0-7%2" \
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
IFS=, read -r _ family variant seed policy eval_num eval_budget goal_offset horizon receding action_block num_samples n_steps topk <<<"${line}"
ckpt_path="${STABLEWM_HOME}/checkpoints/${policy}"
if [[ ! -s "${ckpt_path}" ]]; then
  echo "missing checkpoint ${ckpt_path}" >&2
  exit 1
fi
run_out="${OUT_DIR}/${family}/${variant}/seed_${seed}"
mkdir -p "${run_out}"
cat >"${run_out}/run.json" <<EOF
{"row": ${row}, "family": "${family}", "variant": "${variant}", "seed": ${seed}, "policy": "${policy}", "eval_num": ${eval_num}, "eval_budget": ${eval_budget}, "goal_offset": ${goal_offset}, "horizon": ${horizon}, "receding": ${receding}, "action_block": ${action_block}, "num_samples": ${num_samples}, "n_steps": ${n_steps}, "topk": ${topk}, "slurm_job_id": "${SLURM_ARRAY_JOB_ID}", "slurm_array_task_id": "${SLURM_ARRAY_TASK_ID}"}
EOF
cd "${LEWM_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PATCH_ROOT}:${COMPAT_ROOT}:${LEWM_ROOT}:${PYTHONPATH:-}"
export STABLEWM_HOME="${STABLEWM_HOME}"
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
"${LEWM_ENV}/bin/python" eval.py \
  --config-name=pusht \
  "seed=${seed}" \
  "policy=${policy}" \
  eval.num_eval="${eval_num}" \
  eval.eval_budget="${eval_budget}" \
  eval.goal_offset_steps="${goal_offset}" \
  plan_config.horizon="${horizon}" \
  plan_config.receding_horizon="${receding}" \
  plan_config.action_block="${action_block}" \
  solver.num_samples="${num_samples}" \
  solver.n_steps="${n_steps}" \
  solver.topk="${topk}" \
  output.filename="${RUN_LABEL}_${row}_${family}_${variant}_seed${seed}_results.txt" \
  2>&1 | tee "${run_out}/terminal_eval.log"
touch "${run_out}/completed"
SBATCH
)"

cat <<EOF
lewm_terminal_eval_job=${lewm_terminal_eval_job}
manifest=${MANIFEST}
output_dir=${OUT_DIR}
EOF
