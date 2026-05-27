#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/storage/project/r-agarg35-0/eliu354/envs/pwm/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export WANDB_DIR="${PROJECT_ROOT}/scripts/outputs/mjlab_qs/wandb"
mkdir -p "${WANDB_DIR}"

PROBE_STAGE="${PROBE_STAGE:-rerun_g1_stage_probe_20260522}"
FORMAL_STAGE="${FORMAL_STAGE:-rerun_a25_native_qs_g1stage4_expertboost_20260527}"
TASK_KEY="velocity_flat_unitree_g1"
MANIFEST_DIR="scripts/outputs/mjlab_qs/manifests"
AUDIT_DIR="scripts/outputs/mjlab_qs/audits"
RAW_DIR="scripts/outputs/mjlab_qs/raw"

RANKING="${AUDIT_DIR}/${PROBE_STAGE}_ranking.csv"
FORMAL_COLLECTION="${MANIFEST_DIR}/${FORMAL_STAGE}_collection.csv"

mkdir -p "${MANIFEST_DIR}" "${AUDIT_DIR}" "${RAW_DIR}/${FORMAL_STAGE}"

"${PYTHON_BIN}" scripts/experiments/mjlab_qs/build_native_qs_collection_manifest_from_stage_ranking.py \
  --ranking "${RANKING}" \
  --output "${FORMAL_COLLECTION}" \
  --stage "${FORMAL_STAGE}" \
  --tasks "${TASK_KEY}" \
  --roles random_smooth,medium,expert,expert_noisy \
  --num-envs 32 \
  --episode-overrides expert=1024,expert_noisy=256

N_FORMAL=$("${PYTHON_BIN}" - <<'PY' "${FORMAL_COLLECTION}"
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
)
for IDX in $(seq 0 "$((N_FORMAL - 1))"); do
  "${PYTHON_BIN}" scripts/experiments/mjlab_qs/run_native_collection_row.py \
    --manifest "${FORMAL_COLLECTION}" \
    --row-index "${IDX}" \
    --python-bin "${PYTHON_BIN}" \
    --device cuda:0
done

"${PYTHON_BIN}" scripts/experiments/mjlab_qs/audit_mjlab_qs_quality.py \
  --raw "${RAW_DIR}/${FORMAL_STAGE}" \
  --csv-output "${AUDIT_DIR}/${FORMAL_STAGE}.csv" \
  --json-output "${AUDIT_DIR}/${FORMAL_STAGE}.json" \
  --md-output "${AUDIT_DIR}/${FORMAL_STAGE}.md"

"${PYTHON_BIN}" scripts/experiments/mjlab_qs/build_windows_for_manifest.py \
  --collection-manifest "${FORMAL_COLLECTION}" \
  --mode "${FORMAL_STAGE}" \
  --python-bin "${PYTHON_BIN}" \
  --min-valid-train-windows-per-bucket 500

echo "G1 QS formal retry completed for ${FORMAL_STAGE}."
