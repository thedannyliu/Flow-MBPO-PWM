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

COLLECTOR_STAGE="${COLLECTOR_STAGE:-rerun_g1_collectors_20260522}"
PROBE_STAGE="${PROBE_STAGE:-rerun_g1_stage_probe_20260522}"
FORMAL_STAGE="${FORMAL_STAGE:-rerun_a25_native_qs_g1stage4_20260522}"
TASK_KEY="velocity_flat_unitree_g1"
TASK_ID="Mjlab-Velocity-Flat-Unitree-G1"
METHOD="rslrl_ppo_conservative"
MANIFEST_DIR="scripts/outputs/mjlab_qs/manifests"
AUDIT_DIR="scripts/outputs/mjlab_qs/audits"
RAW_DIR="scripts/outputs/mjlab_qs/raw"
NATIVE_ROOT="scripts/outputs/mjlab_qs/native_collectors/${COLLECTOR_STAGE}"

mkdir -p "${MANIFEST_DIR}" "${AUDIT_DIR}" "${RAW_DIR}/${PROBE_STAGE}" "${RAW_DIR}/${FORMAL_STAGE}"

PROBE_MANIFEST="${MANIFEST_DIR}/${PROBE_STAGE}.csv"
"${PYTHON_BIN}" - <<'PY' "${NATIVE_ROOT}" "${PROBE_STAGE}" "${PROBE_MANIFEST}"
import csv
import re
import sys
from pathlib import Path

native_root = Path(sys.argv[1])
stage = sys.argv[2]
output = Path(sys.argv[3])
task_key = "velocity_flat_unitree_g1"
task_id = "Mjlab-Velocity-Flat-Unitree-G1"
method = "rslrl_ppo_conservative"
wanted = [500, 15000, 29999, 30000]
rows = []

random_out = Path("scripts/outputs/mjlab_qs/raw") / stage / f"{task_key}_random_smooth_seed0.pt"
rows.append({
    "stage": stage,
    "task_key": task_key,
    "task_id": task_id,
    "method": "random_smooth",
    "checkpoint": "",
    "quality_bin": "random_smooth",
    "collector_mode": "random_smooth",
    "collector_id": "native_random_smooth_reference",
    "output": str(random_out),
    "metadata_output": str(random_out.with_suffix(".json")),
    "seed": "0",
    "num_envs": "16",
    "episodes": "100",
    "episode_length": "1000",
    "teacher_blend": "1.0",
    "action_noise_std": "0.0",
    "command_dim": "3",
    "command_position": "tail",
})

for seed in (0, 1, 2):
    run_dir = native_root / task_key / method / f"seed_{seed}"
    ckpts = {}
    for ckpt in run_dir.glob("model_*.pt"):
        m = re.search(r"model_(\d+)\.pt$", ckpt.name)
        if m:
            ckpts[int(m.group(1))] = ckpt
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {run_dir}")
    selected = sorted({i for i in wanted if i in ckpts} | {max(ckpts)})
    for ckpt_id in selected:
        ckpt = ckpts[ckpt_id]
        out = Path("scripts/outputs/mjlab_qs/raw") / stage / f"{task_key}_{method}_seed{seed}_iter{ckpt_id}.pt"
        rows.append({
            "stage": stage,
            "task_key": task_key,
            "task_id": task_id,
            "method": method,
            "checkpoint": str(ckpt),
            "quality_bin": "stage_candidate",
            "collector_mode": "checkpoint",
            "collector_id": f"native_{method}_seed{seed}_iter{ckpt_id}",
            "output": str(out),
            "metadata_output": str(out.with_suffix(".json")),
            "seed": str(seed),
            "num_envs": "16",
            "episodes": "100",
            "episode_length": "1000",
            "teacher_blend": "1.0",
            "action_noise_std": "0.0",
            "command_dim": "3",
            "command_position": "tail",
        })

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} probe rows to {output}")
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

"${PYTHON_BIN}" scripts/experiments/mjlab_qs/export_native_stage_quality_ranking.py \
  --raw "${RAW_DIR}/${PROBE_STAGE}" \
  --output "${AUDIT_DIR}/${PROBE_STAGE}_ranking.csv"

FORMAL_COLLECTION="${MANIFEST_DIR}/${FORMAL_STAGE}_collection.csv"
"${PYTHON_BIN}" scripts/experiments/mjlab_qs/build_native_qs_collection_manifest_from_stage_ranking.py \
  --ranking "${AUDIT_DIR}/${PROBE_STAGE}_ranking.csv" \
  --output "${FORMAL_COLLECTION}" \
  --stage "${FORMAL_STAGE}" \
  --tasks "${TASK_KEY}" \
  --roles random_smooth,medium,expert,expert_noisy \
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
  --python-bin "${PYTHON_BIN}" \
  --min-valid-train-windows-per-bucket 500

echo "Rerun G1 QS data collection, quality audit, and H16 window build completed for ${FORMAL_STAGE}."
