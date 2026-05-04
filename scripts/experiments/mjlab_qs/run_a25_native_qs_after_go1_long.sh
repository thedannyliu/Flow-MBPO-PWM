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

PROBE_STAGE="mjlab_native_quality_probe_go1_long_v2"
FORMAL_STAGE="a25_native_qs"
MANIFEST_DIR="scripts/outputs/mjlab_qs/manifests"
AUDIT_DIR="scripts/outputs/mjlab_qs/audits"
RAW_DIR="scripts/outputs/mjlab_qs/raw"

mkdir -p "${MANIFEST_DIR}" "${AUDIT_DIR}" "${RAW_DIR}/${PROBE_STAGE}"

GO1_ROOT_A="scripts/outputs/mjlab_qs/native_collectors/mjlab_native_collector_go1_long_v2"
GO1_ROOT_B="scripts/outputs/mjlab_qs/native_collectors/mjlab_native_collector_go1_long_v2_l40sbackup"

PROBE_A="${MANIFEST_DIR}/${PROBE_STAGE}_part_long.csv"
PROBE_B="${MANIFEST_DIR}/${PROBE_STAGE}_part_backup.csv"
PROBE_MANIFEST="${MANIFEST_DIR}/${PROBE_STAGE}.csv"

"${PYTHON_BIN}" scripts/experiments/mjlab_qs/build_native_collector_probe_manifest.py \
  --native-root "${GO1_ROOT_A}" \
  --output "${PROBE_A}" \
  --stage "${PROBE_STAGE}" \
  --episodes 100 \
  --num-envs 16 \
  --include-random

"${PYTHON_BIN}" scripts/experiments/mjlab_qs/build_native_collector_probe_manifest.py \
  --native-root "${GO1_ROOT_B}" \
  --output "${PROBE_B}" \
  --stage "${PROBE_STAGE}" \
  --episodes 100 \
  --num-envs 16

"${PYTHON_BIN}" - <<'PY' "${PROBE_A}" "${PROBE_B}" "${PROBE_MANIFEST}"
import csv
import sys
from pathlib import Path

parts = [Path(sys.argv[1]), Path(sys.argv[2])]
out = Path(sys.argv[3])
rows = []
for part in parts:
    if not part.exists():
        continue
    with part.open(newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
if not rows:
    raise RuntimeError("No Go1 long probe rows were built.")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote combined Go1 long probe manifest {out} rows={len(rows)}")
PY

N_PROBE=$("${PYTHON_BIN}" - <<'PY' "${PROBE_MANIFEST}"
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
)
for IDX in $(seq 0 "$((N_PROBE - 1))"); do
  "${PYTHON_BIN}" scripts/experiments/mjlab_qs/run_native_collection_row.py \
    --manifest "${PROBE_MANIFEST}" \
    --row-index "${IDX}" \
    --python-bin "${PYTHON_BIN}" \
    --device cuda:0
done

"${PYTHON_BIN}" scripts/experiments/mjlab_qs/audit_mjlab_qs_quality.py \
  --raw "${RAW_DIR}/${PROBE_STAGE}" \
  --csv-output "${AUDIT_DIR}/${PROBE_STAGE}.csv" \
  --json-output "${AUDIT_DIR}/${PROBE_STAGE}.json" \
  --md-output "${AUDIT_DIR}/${PROBE_STAGE}.md" \
  --allow-fail

"${PYTHON_BIN}" scripts/experiments/mjlab_qs/export_native_quality_probe_ranking.py \
  --raw "${RAW_DIR}/${PROBE_STAGE}" \
  --output "${AUDIT_DIR}/${PROBE_STAGE}_ranking.csv"

COMBINED_RANKING="${AUDIT_DIR}/a25_native_qs_combined_ranking.csv"
"${PYTHON_BIN}" - <<'PY' \
  "${AUDIT_DIR}/mjlab_native_quality_probe_v1_ranking.csv" \
  "${AUDIT_DIR}/${PROBE_STAGE}_ranking.csv" \
  "${COMBINED_RANKING}"
import csv
import sys
from pathlib import Path

old_rank = Path(sys.argv[1])
new_rank = Path(sys.argv[2])
out = Path(sys.argv[3])
rows = []
for path in (new_rank, old_rank):
    if not path.exists():
        continue
    with path.open(newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
if not rows:
    raise RuntimeError("No ranking rows were available.")
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote combined ranking {out} rows={len(rows)}")
PY

FORMAL_COLLECTION="${MANIFEST_DIR}/${FORMAL_STAGE}_collection.csv"
"${PYTHON_BIN}" scripts/experiments/mjlab_qs/build_native_qs_collection_manifest.py \
  --ranking "${COMBINED_RANKING}" \
  --output "${FORMAL_COLLECTION}" \
  --stage "${FORMAL_STAGE}" \
  --num-envs 32

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
  --python-bin "${PYTHON_BIN}"

TRAIN_MANIFEST="${MANIFEST_DIR}/${FORMAL_STAGE}_train.csv"
"${PYTHON_BIN}" scripts/experiments/mjlab_qs/build_phaseA_train_manifest_from_windows.py \
  --stage "${FORMAL_STAGE}" \
  --output "${TRAIN_MANIFEST}" \
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
    "h100": rows[0:6],
    "h200": rows[6:12],
    "l40s": rows[12:18],
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
  --manifest "${MANIFEST_DIR}/${FORMAL_STAGE}_train_h100.csv" \
  --gpu-type H100 \
  --max-concurrent 6 \
  --time 08:00:00 \
  --mem 128G \
  --cpus 8 \
  --python-bin "${PYTHON_BIN}"

bash scripts/experiments/mjlab_qs/submit_array.sh \
  --kind train \
  --manifest "${MANIFEST_DIR}/${FORMAL_STAGE}_train_h200.csv" \
  --gpu-type H200 \
  --max-concurrent 6 \
  --time 08:00:00 \
  --mem 128G \
  --cpus 8 \
  --python-bin "${PYTHON_BIN}"

bash scripts/experiments/mjlab_qs/submit_array.sh \
  --kind train \
  --manifest "${MANIFEST_DIR}/${FORMAL_STAGE}_train_l40s.csv" \
  --gpu-type L40S \
  --max-concurrent 6 \
  --time 08:00:00 \
  --mem 128G \
  --cpus 8 \
  --python-bin "${PYTHON_BIN}"

echo "A2.5 native QS dataset was built and formal WM training arrays were submitted."
