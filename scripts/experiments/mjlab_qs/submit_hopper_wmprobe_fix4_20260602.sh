#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="${ROOT}/logs/pwm_original_parity/locked_env_20260601"
ENV_DIR="/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
QOS="${QOS:-embers}"
CPATH_VALUE="/usr/include/c++/11:/usr/include/c++/11/x86_64-redhat-linux:/usr/lib/gcc/x86_64-redhat-linux/11/include:/usr/include"

if [[ "${QOS,,}" == "inferno" && "${ALLOW_INFERNO_QOS:-0}" != "1" ]]; then
  echo "Error: inferno QOS requires explicit user approval. Use embers for GPU jobs." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${ROOT}/eval_results/pwm_phase2_hopper_locked_probe_20260602"

sbatch \
  --job-name="pwm_hopper_locked_wmprobe_h100_fix4" \
  --account="${ACCOUNT}" \
  --partition="gpu-h100" \
  --qos="${QOS}" \
  --gres="gpu:h100:1" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem="128G" \
  --time="00:30:00" \
  --output="${LOG_DIR}/pwm_hopper_locked_wmprobe_h100_fix4_%j.out" \
  --error="${LOG_DIR}/pwm_hopper_locked_wmprobe_h100_fix4_%j.err" \
  --wrap="cd ${ROOT} && \
export ENV_DIR=${ENV_DIR} && \
export PYTHONNOUSERSITE=1 && \
export PATH=${ENV_DIR}/bin:\$PATH && \
export CUDA_HOME=${ENV_DIR} && \
export CUDACXX=${ENV_DIR}/bin/nvcc && \
export LD_LIBRARY_PATH=${ENV_DIR}/lib:\${LD_LIBRARY_PATH:-} && \
export CC=/usr/bin/gcc && \
export CXX=/usr/bin/g++ && \
export CUDAHOSTCXX=/usr/bin/g++ && \
export MAX_JOBS=4 && \
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 && \
export HYDRA_FULL_ERROR=1 && \
unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH GCC_EXEC_PREFIX COMPILER_PATH && \
export CPATH=${CPATH_VALUE} && \
export SANDBOX=/tmp/dflex_hopper_probe_sandbox_\${SLURM_JOB_ID}_fix4 && \
rm -rf \"\$SANDBOX\" && mkdir -p \"\$SANDBOX\" && \
cp -a ${ENV_DIR}/lib/python3.10/site-packages/dflex \"\$SANDBOX/\" && \
rm -rf \"\$SANDBOX/dflex/kernels\" && \
export PYTHONPATH=\"\$SANDBOX:${ROOT}/baselines/PWM/src:${ROOT}/src:\${PYTHONPATH:-}\" && \
${ENV_DIR}/bin/python scripts/diagnostics/pwm_dflex_checkpoint_probe.py --checkpoint baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601/final_policy.pt --env dflex_hopper --checkpoint-mode full --policy actor --device cuda:0 --seed 0 --num-envs 64 --steps 128 --output eval_results/pwm_phase2_hopper_locked_probe_20260602/final_actor_wm_vs_real_fix4.json && \
${ENV_DIR}/bin/python scripts/diagnostics/pwm_dflex_checkpoint_probe.py --checkpoint baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601/best_policy.pt --env dflex_hopper --checkpoint-mode full --policy actor --device cuda:0 --seed 0 --num-envs 64 --steps 128 --output eval_results/pwm_phase2_hopper_locked_probe_20260602/best_actor_wm_vs_real_fix4.json"
