#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
CPU_PARTITION="${CPU_PARTITION:-cpu-small}"
CPU_QOS="${CPU_QOS:-embers}"
GPU_PARTITION="${GPU_PARTITION:-gpu-a100}"
GPU_GRES="${GPU_GRES:-gpu:a100:1}"
GPU_QOS="${GPU_QOS:-embers}"
LOG_DIR="${ROOT}/logs/slurm/image_official"

NEWT_ROOT="${NEWT_ROOT:-/storage/project/r-agarg35-0/eliu354/external_repos/newt}"
LEWM_ROOT="${LEWM_ROOT:-/storage/project/r-agarg35-0/eliu354/external_repos/le-wm}"
NEWT_ENV="${NEWT_ENV:-/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602}"
LEWM_ENV="${LEWM_ENV:-/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602}"
NEWT_MARKER="${NEWT_MARKER:-${NEWT_ENV}/.newt_official_setup_ok_20260602}"
UV_PREFIX="${UV_PREFIX:-/storage/project/r-agarg35-0/eliu354/tools/uv_official_20260602}"
DATA_ROOT="${DATA_ROOT:-/storage/project/r-agarg35-0/eliu354/external_data}"

if { [[ "${CPU_QOS,,}" == "inferno" ]] || [[ "${GPU_QOS,,}" == "inferno" ]]; } \
  && [[ "${ALLOW_INFERNO_QOS:-0}" != "1" ]]; then
  echo "Error: inferno QOS requires explicit user approval. Use embers for GPU jobs." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${DATA_ROOT}/newt_demos" "${DATA_ROOT}/lewm_stablewm"

newt_setup_job="$(
  sbatch --parsable \
    --job-name="newt_official_env_setup_20260602" \
    --account="${ACCOUNT}" \
    --partition="${CPU_PARTITION}" \
    --qos="${CPU_QOS}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem="48G" \
    --time="04:00:00" \
    --output="${LOG_DIR}/newt_official_env_setup_%j.out" \
    --error="${LOG_DIR}/newt_official_env_setup_%j.err" \
    --wrap="cd ${NEWT_ROOT} && \
source ~/.bashrc && \
export PYTHONNOUSERSITE=1 && \
module load swig/4.1.1 && \
if [[ -d ${NEWT_ENV} && ! -f ${NEWT_MARKER} ]]; then rm -rf ${NEWT_ENV}; fi && \
if [[ -d ${NEWT_ENV} && ! -x ${NEWT_ENV}/bin/python ]]; then rm -rf ${NEWT_ENV}; fi && \
if [[ -x ${NEWT_ENV}/bin/python ]] && ! ${NEWT_ENV}/bin/python -c 'import encodings' >/dev/null 2>&1; then rm -rf ${NEWT_ENV}; fi && \
if [[ ! -x ${NEWT_ENV}/bin/python ]]; then conda env create -p ${NEWT_ENV} -f docker/environment.yaml; fi && \
${NEWT_ENV}/bin/python -m pip install --no-cache-dir 'ale_py==0.10' && \
${NEWT_ENV}/bin/python - <<'PY' && touch ${NEWT_MARKER}
import torch, torchvision, hydra, gymnasium
print('newt_env_python_ok')
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('torchvision', torchvision.__version__)
print('hydra', hydra.__version__)
print('gymnasium', gymnasium.__version__)
PY"
)"

lewm_setup_job="$(
  sbatch --parsable \
    --job-name="lewm_official_env_setup_20260602" \
    --account="${ACCOUNT}" \
    --partition="${CPU_PARTITION}" \
    --qos="${CPU_QOS}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem="48G" \
    --time="04:00:00" \
    --output="${LOG_DIR}/lewm_official_env_setup_%j.out" \
    --error="${LOG_DIR}/lewm_official_env_setup_%j.err" \
    --wrap="cd ${LEWM_ROOT} && \
mkdir -p ${UV_PREFIX} && \
if [[ ! -x ${UV_PREFIX}/bin/uv ]]; then python -m pip install --prefix ${UV_PREFIX} uv; fi && \
if [[ -d ${LEWM_ENV} && ! -x ${LEWM_ENV}/bin/python ]]; then rm -rf ${LEWM_ENV}; fi && \
if [[ -x ${LEWM_ENV}/bin/python ]] && ! PYTHONNOUSERSITE=1 ${LEWM_ENV}/bin/python -c 'import encodings' >/dev/null 2>&1; then rm -rf ${LEWM_ENV}; fi && \
if [[ ! -x ${LEWM_ENV}/bin/python ]]; then ${UV_PREFIX}/bin/uv venv --python=3.10 ${LEWM_ENV}; fi && \
${UV_PREFIX}/bin/uv pip install --python ${LEWM_ENV}/bin/python 'stable-worldmodel[train,env]' && \
PYTHONNOUSERSITE=1 PYTHONPATH=${LEWM_ROOT} STABLEWM_HOME=${DATA_ROOT}/lewm_stablewm ${LEWM_ENV}/bin/python - <<'PY'
import torch, hydra, stable_worldmodel, stable_pretraining
print('lewm_env_python_ok')
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('hydra', hydra.__version__)
print('stable_worldmodel', stable_worldmodel.__file__)
print('stable_pretraining', stable_pretraining.__file__)
PY"
)"

newt_import_job="$(
  sbatch --parsable \
    --job-name="newt_official_import_config_smoke_20260602" \
    --account="${ACCOUNT}" \
    --partition="${CPU_PARTITION}" \
    --qos="${CPU_QOS}" \
    --dependency="afterok:${newt_setup_job}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="24G" \
    --time="00:30:00" \
    --output="${LOG_DIR}/newt_official_import_config_smoke_%j.out" \
    --error="${LOG_DIR}/newt_official_import_config_smoke_%j.err" \
    --wrap="cd ${NEWT_ROOT}/tdmpc2 && \
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 WANDB_MODE=disabled && \
${NEWT_ENV}/bin/python - <<'PY'
import json
from pathlib import Path
from config import Config, parse_cfg
from common import MODEL_SIZE, TASK_SET
tasks = json.loads(Path('../tasks.json').read_text())
cfg = Config(task='walker-walk', model_size='B', steps=1000, enable_wandb=False, compile=False, save_video=False, save_agent=False, data_dir='${DATA_ROOT}/newt_demos')
parsed = parse_cfg(cfg)
print('newt_import_config_ok')
print('task_count', len(tasks))
print('model_sizes', sorted(MODEL_SIZE))
print('walker_action_dim', tasks['walker-walk']['action_dim'])
print('parsed_task', parsed.task, 'obs', parsed.obs, 'model_size', parsed.model_size, 'num_envs', parsed.num_envs)
PY"
)"

newt_train_job="$(
  sbatch --parsable \
    --job-name="newt_official_walker_smoke_a100_20260602" \
    --account="${ACCOUNT}" \
    --partition="${GPU_PARTITION}" \
    --qos="${GPU_QOS}" \
    --gres="${GPU_GRES}" \
    --dependency="afterok:${newt_setup_job}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem="96G" \
    --time="02:00:00" \
    --output="${LOG_DIR}/newt_official_walker_smoke_%j.out" \
    --error="${LOG_DIR}/newt_official_walker_smoke_%j.err" \
    --wrap="cd ${NEWT_ROOT}/tdmpc2 && \
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 WANDB_MODE=disabled HYDRA_FULL_ERROR=1 && \
${NEWT_ENV}/bin/python train.py task=walker-walk model_size=B steps=1000 seed=0 enable_wandb=false save_video=false save_agent=false compile=false num_envs=1 batch_size=16 buffer_size=10000 eval_episodes=1 exp_name=official_walker_smoke_20260602 data_dir=${DATA_ROOT}/newt_demos"
)"

lewm_import_job="$(
  sbatch --parsable \
    --job-name="lewm_official_import_config_smoke_20260602" \
    --account="${ACCOUNT}" \
    --partition="${CPU_PARTITION}" \
    --qos="${CPU_QOS}" \
    --dependency="afterok:${lewm_setup_job}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="24G" \
    --time="00:30:00" \
    --output="${LOG_DIR}/lewm_official_import_config_smoke_%j.out" \
    --error="${LOG_DIR}/lewm_official_import_config_smoke_%j.err" \
    --wrap="cd ${LEWM_ROOT} && \
export PYTHONNOUSERSITE=1 PYTHONPATH=${LEWM_ROOT} STABLEWM_HOME=${DATA_ROOT}/lewm_stablewm WANDB_MODE=disabled && \
${LEWM_ENV}/bin/python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
import torch, stable_worldmodel, stable_pretraining
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
config_dir = str((Path.cwd() / 'config' / 'train').resolve())
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name='lewm', overrides=['data=pusht', 'trainer.max_epochs=1', 'loader.batch_size=2', 'wandb.enabled=false'])
print('lewm_import_config_ok')
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('train_data', cfg.data.dataset.name)
print('max_epochs', cfg.trainer.max_epochs)
print('sigreg_weight', cfg.loss.sigreg.weight)
print('classes', JEPA.__name__, ARPredictor.__name__, Embedder.__name__, MLP.__name__, SIGReg.__name__)
PY"
)"

cat <<EOF
newt_setup_job=${newt_setup_job}
lewm_setup_job=${lewm_setup_job}
newt_import_job=${newt_import_job}
newt_train_job=${newt_train_job}
lewm_import_job=${lewm_import_job}
EOF
