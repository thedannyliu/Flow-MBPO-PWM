#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/storage/ice1/2/9/eliu354/conda_envs/flow-mbpo/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export WANDB_DIR="${PROJECT_ROOT}/scripts/outputs/mjlab_qs/wandb"
mkdir -p "${WANDB_DIR}"

STAGE="${STAGE:-a25_native_qs_g1only}"
TASKS="${TASKS:-velocity_flat_unitree_g1}"
MIN_VALID_TRAIN_WINDOWS_PER_BUCKET="${MIN_VALID_TRAIN_WINDOWS_PER_BUCKET:-500}"
MANIFEST_DIR="scripts/outputs/mjlab_qs/manifests"
AUDIT_DIR="scripts/outputs/mjlab_qs/audits"
RAW_DIR="scripts/outputs/mjlab_qs/raw"

COLLECTION="${MANIFEST_DIR}/${STAGE}_collection.csv"

"${PYTHON_BIN}" scripts/experiments/mjlab_qs/audit_mjlab_qs_quality.py \
  --raw "${RAW_DIR}/${STAGE}" \
  --csv-output "${AUDIT_DIR}/${STAGE}.csv" \
  --json-output "${AUDIT_DIR}/${STAGE}.json" \
  --md-output "${AUDIT_DIR}/${STAGE}.md"

# G1-only is a single-task diagnostic branch. Keep episode gates strict but use
# a lower per-bucket window gate because random_smooth G1 episodes terminate
# quickly by design and are not the expert-data anchor.
"${PYTHON_BIN}" scripts/experiments/mjlab_qs/build_windows_for_manifest.py \
  --collection-manifest "${COLLECTION}" \
  --mode "${STAGE}" \
  --python-bin "${PYTHON_BIN}" \
  --min-valid-train-windows-per-bucket "${MIN_VALID_TRAIN_WINDOWS_PER_BUCKET}"

TRAIN_MANIFEST="${MANIFEST_DIR}/${STAGE}_train.csv"
"${PYTHON_BIN}" scripts/experiments/mjlab_qs/build_phaseA_train_manifest_from_windows.py \
  --stage "${STAGE}" \
  --output "${TRAIN_MANIFEST}" \
  --tasks "${TASKS}" \
  --methods mlp_ref,flow_ref,residual_flow_frozen_mlp \
  --seeds 0,1,2 \
  --train-iters 50000 \
  --eval-every 5000

"${PYTHON_BIN}" - <<'PY' "${TRAIN_MANIFEST}" "${MANIFEST_DIR}"
import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
manifest_dir = Path(sys.argv[2])
rows = list(csv.DictReader(src.open()))
splits = {
    "h100": rows[0:3],
    "h200": rows[3:6],
    "l40s": rows[6:9],
}
for name, split_rows in splits.items():
    out = manifest_dir / f"{src.stem}_{name}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(split_rows)
    print(f"wrote {out} rows={len(split_rows)}")
PY

bash scripts/experiments/mjlab_qs/submit_array.sh \
  --kind train \
  --manifest "${MANIFEST_DIR}/${STAGE}_train_h100.csv" \
  --gpu-type H100 \
  --max-concurrent 3 \
  --time 08:00:00 \
  --mem 128G \
  --cpus 8 \
  --python-bin "${PYTHON_BIN}"

bash scripts/experiments/mjlab_qs/submit_array.sh \
  --kind train \
  --manifest "${MANIFEST_DIR}/${STAGE}_train_h200.csv" \
  --gpu-type H200 \
  --max-concurrent 3 \
  --time 08:00:00 \
  --mem 128G \
  --cpus 8 \
  --python-bin "${PYTHON_BIN}"

bash scripts/experiments/mjlab_qs/submit_array.sh \
  --kind train \
  --manifest "${MANIFEST_DIR}/${STAGE}_train_l40s.csv" \
  --gpu-type L40S \
  --max-concurrent 3 \
  --time 08:00:00 \
  --mem 128G \
  --cpus 8 \
  --python-bin "${PYTHON_BIN}"

echo "G1-only QS dataset was built and formal A2.5 WM training arrays were submitted."
