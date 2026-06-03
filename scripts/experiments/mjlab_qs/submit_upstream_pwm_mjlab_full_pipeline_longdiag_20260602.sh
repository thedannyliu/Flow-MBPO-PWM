#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
GPU_QOS="${GPU_QOS:-embers}"
PARTITION="${PARTITION:-gpu-h200}"
GPU_GRES="${GPU_GRES:-gpu:h200:1}"
LOG_DIR="${ROOT}/logs/slurm/mjlab_qs/upstream_pwm_full_pipeline"
LOCKED_MJLAB_PYTHON="${LOCKED_MJLAB_PYTHON:-${ROOT}/scripts/experiments/mjlab_qs/locked_mjlab_python.py}"

if [[ "${GPU_QOS,,}" == "inferno" && "${ALLOW_INFERNO_QOS:-0}" != "1" ]]; then
  echo "Error: inferno QOS requires explicit user approval. Use embers for GPU jobs." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

sbatch --parsable \
  --job-name="upstream_pwm_mjlab_full_longdiag_h200_20260602" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${GPU_QOS}" \
  --gres="${GPU_GRES}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem="128G" \
  --time="02:00:00" \
  --output="${LOG_DIR}/upstream_pwm_mjlab_full_longdiag_h200_%j.out" \
  --error="${LOG_DIR}/upstream_pwm_mjlab_full_longdiag_h200_%j.err" \
  --export=ALL,ROOT="${ROOT}",LOCKED_MJLAB_PYTHON="${LOCKED_MJLAB_PYTHON}" \
  <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

cd "${ROOT}/baselines/PWM/scripts"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT}/src:${ROOT}/baselines/PWM/src:${ROOT}/baselines/PWM/external/tdmpc2:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1

mkdir -p cfg/env
cat > cfg/env/mjlab_velocity_flat_unitree_g1.yaml <<'YAML'
config:
  _target_: flow_mbpo_pwm.envs.mjlab_pwm_adapter.create_mjlab_pwm_env
  task_id: Mjlab-Velocity-Flat-Unitree-G1
  num_envs: 32
  device: ${general.device}
  seed: ${general.seed}
  episode_length: 64
  action_repeat: 2
  no_grad: false

  obs_key: state
  obs_key_candidates: [state, policy, observation, obs]

  strict_terminal_obs: true
  expect_auto_reset: true
  fail_on_missing_terminal_obs: false
  warn_missing_terminal_obs_every: 10

  mjlab_env_kwargs: {}
YAML

"${LOCKED_MJLAB_PYTHON}" train_dflex.py \
  env=mjlab_velocity_flat_unitree_g1 \
  alg=pwm \
  general.run_wandb=false \
  general.seed=0 \
  general.eval_runs=4 \
  general.logdir=logs/upstream_pwm_mjlab_full_longdiag_h200_seed0_20260602 \
  alg.max_epochs=200 \
  alg.horizon=16 \
  alg.save_interval=50 \
  alg.critic_iterations=2 \
  alg.critic_batches=2 \
  alg.wm_iterations=4 \
  alg.wm_batch_size=64 \
  alg.wm_buffer_size=50000 \
  alg.rew_rms=false \
  alg.ret_rms=true \
  alg.detach=true
SBATCH
