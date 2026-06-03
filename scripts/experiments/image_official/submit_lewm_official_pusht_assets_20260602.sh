#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACCOUNT="${ACCOUNT:-gts-agarg35}"
CPU_PARTITION="${CPU_PARTITION:-cpu-small}"
CPU_QOS="${CPU_QOS:-embers}"
LOG_DIR="${ROOT}/logs/slurm/image_official"

LEWM_ROOT="${LEWM_ROOT:-/storage/project/r-agarg35-0/eliu354/external_repos/le-wm}"
LEWM_ENV="${LEWM_ENV:-/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602}"
STABLEWM_HOME="${STABLEWM_HOME:-/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm}"

if [[ "${CPU_QOS,,}" == "inferno" && "${ALLOW_INFERNO_QOS:-0}" != "1" ]]; then
  echo "Error: inferno QOS requires explicit user approval. Use embers for CPU jobs." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${STABLEWM_HOME}"

sbatch --parsable \
  --job-name="lewm_official_pusht_assets_fix2_20260602" \
  --account="${ACCOUNT}" \
  --partition="${CPU_PARTITION}" \
  --qos="${CPU_QOS}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem="64G" \
  --time="04:00:00" \
  --output="${LOG_DIR}/lewm_official_pusht_assets_fix2_%j.out" \
  --error="${LOG_DIR}/lewm_official_pusht_assets_fix2_%j.err" \
  --wrap="cd ${LEWM_ROOT} && \
export PYTHONNOUSERSITE=1 PYTHONPATH=${LEWM_ROOT} STABLEWM_HOME=${STABLEWM_HOME} WANDB_MODE=disabled && \
${LEWM_ENV}/bin/python - <<'PY'
import json
import os
import importlib.util
from pathlib import Path

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import zstandard as zstd
from huggingface_hub import hf_hub_download, list_repo_files

from jepa import JEPA
from module import ARPredictor, Embedder, MLP

def clean_hydra_kwargs(value):
    if isinstance(value, dict):
        return {
            key: clean_hydra_kwargs(item)
            for key, item in value.items()
            if key not in {'_target_', '_partial_'}
        }
    return value

def normalize_legacy_vit_keys(state_dict):
    replacements = {
        '.attention.attention.query.': '.attention.q_proj.',
        '.attention.attention.key.': '.attention.k_proj.',
        '.attention.attention.value.': '.attention.v_proj.',
        '.attention.output.dense.': '.attention.o_proj.',
        '.intermediate.dense.': '.mlp.fc1.',
        '.output.dense.': '.mlp.fc2.',
    }
    normalized = {}
    for key, value in state_dict.items():
        new_key = key.replace('encoder.encoder.layer.', 'encoder.layers.')
        for old, new in replacements.items():
            new_key = new_key.replace(old, new)
        normalized[new_key] = value
    return normalized

cache = Path(os.environ['STABLEWM_HOME'])
hf_model_dir = cache / 'hf_pusht'
download_dir = cache / 'downloads'
hf_model_dir.mkdir(parents=True, exist_ok=True)
download_dir.mkdir(parents=True, exist_ok=True)

print('model_repo_files', list(list_repo_files('quentinll/lewm-pusht', repo_type='model')))
print('dataset_repo_files', list(list_repo_files('quentinll/lewm-pusht', repo_type='dataset')))

for filename in ['config.json', 'weights.pt']:
    path = Path(hf_hub_download(
        repo_id='quentinll/lewm-pusht',
        repo_type='model',
        filename=filename,
        local_dir=hf_model_dir,
    ))
    print('downloaded_model_file', filename, path, path.stat().st_size)

zst_path = Path(hf_hub_download(
    repo_id='quentinll/lewm-pusht',
    repo_type='dataset',
    filename='pusht_expert_train.h5.zst',
    local_dir=download_dir,
))
print('downloaded_dataset_archive', zst_path, zst_path.stat().st_size)

h5_path = cache / 'pusht_expert_train.h5'
if not h5_path.exists() or h5_path.stat().st_size == 0:
    dctx = zstd.ZstdDecompressor()
    with zst_path.open('rb') as src, h5_path.open('wb') as dst:
        dctx.copy_stream(src, dst)
print('dataset_h5', h5_path, h5_path.stat().st_size)

cfg = json.loads((hf_model_dir / 'config.json').read_text())
spt_utils_path = Path(spt.__file__).parent / 'backbone' / 'utils.py'
spt_utils_spec = importlib.util.spec_from_file_location('spt_backbone_utils_direct', spt_utils_path)
spt_utils = importlib.util.module_from_spec(spt_utils_spec)
spt_utils_spec.loader.exec_module(spt_utils)

encoder = spt_utils.vit_hf(
    cfg['encoder']['size'],
    patch_size=cfg['encoder']['patch_size'],
    image_size=cfg['encoder']['image_size'],
    pretrained=False,
    use_mask_token=False,
)
mlp = lambda k: MLP(
    input_dim=cfg[k]['input_dim'],
    output_dim=cfg[k]['output_dim'],
    hidden_dim=cfg[k]['hidden_dim'],
    norm_fn=torch.nn.BatchNorm1d,
)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**clean_hydra_kwargs(cfg['predictor'])),
    action_encoder=Embedder(**clean_hydra_kwargs(cfg['action_encoder'])),
    projector=mlp('projector'),
    pred_proj=mlp('pred_proj'),
)
state_dict = normalize_legacy_vit_keys(
    torch.load(hf_model_dir / 'weights.pt', map_location='cpu', weights_only=False)
)
model.load_state_dict(state_dict, strict=True)
out = cache / 'pusht' / 'lewm_object.ckpt'
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
print('converted_checkpoint', out, out.stat().st_size)

cost = swm.policy.AutoCostModel('pusht/lewm', cache_dir=str(cache))
print('autocost_load_ok', type(cost).__name__)
print('lewm_official_pusht_assets_ok')
PY"
