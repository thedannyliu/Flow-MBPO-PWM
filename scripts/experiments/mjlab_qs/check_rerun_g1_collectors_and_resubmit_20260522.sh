#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/storage/project/r-agarg35-0/eliu354/envs/pwm/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

STAGE="rerun_g1_collectors_20260522"
MANIFEST_DIR="scripts/outputs/mjlab_qs/manifests"
BASE_MANIFEST="${MANIFEST_DIR}/${STAGE}.csv"
RESUME_MANIFEST="${MANIFEST_DIR}/${STAGE}_resume.csv"
PIPELINE_SCRIPT="scripts/experiments/mjlab_qs/run_rerun_g1_qs_after_collectors_20260522.sh"
LOG_DIR="logs/slurm/mjlab_qs/rerun_g1_qs_20260522"
mkdir -p "${LOG_DIR}"

set +e
"${PYTHON_BIN}" - <<'PY' "${BASE_MANIFEST}" "${RESUME_MANIFEST}"
import csv
import re
import sys
from pathlib import Path

base = Path(sys.argv[1])
resume = Path(sys.argv[2])
rows = list(csv.DictReader(base.open(newline="", encoding="utf-8")))
todo = []
for row in rows:
    out_dir = Path(row["output_dir"])
    max_iter = 0
    for ckpt in out_dir.glob("model_*.pt"):
        m = re.search(r"model_(\d+)\.pt$", ckpt.name)
        if m:
            max_iter = max(max_iter, int(m.group(1)))
    if max_iter < 29999:
        row = dict(row)
        row["resume"] = "true"
        todo.append(row)
        print(f"collector incomplete: seed={row['seed']} max_checkpoint={max_iter}")
    else:
        print(f"collector complete: seed={row['seed']} max_checkpoint={max_iter}")

if not todo:
    if resume.exists():
        resume.unlink()
    print("all collectors complete; no resume submission needed")
    raise SystemExit(0)

resume.parent.mkdir(parents=True, exist_ok=True)
with resume.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(todo)
print(f"wrote resume manifest {resume} rows={len(todo)}")
raise SystemExit(2)
PY
RC=$?
set -e

if [[ "${RC}" == "0" ]]; then
  exit 0
fi
if [[ "${RC}" != "2" ]]; then
  exit "${RC}"
fi

SUBMIT_OUT=$(bash scripts/experiments/mjlab_qs/submit_array.sh \
  --kind native_collector \
  --manifest "${RESUME_MANIFEST}" \
  --gpu-type H100 \
  --partition gpu-h100 \
  --qos embers \
  --account gts-agarg35 \
  --max-concurrent 3 \
  --time 08:00:00 \
  --mem 128G \
  --cpus 8 \
  --python-bin "${PYTHON_BIN}")
echo "${SUBMIT_OUT}"
RESUME_JOB=$(awk '/Submitted batch job/ {print $4}' <<<"${SUBMIT_OUT}")
if [[ -z "${RESUME_JOB}" ]]; then
  echo "failed to parse resume job id" >&2
  exit 1
fi

sbatch \
  --job-name=mjqs_rerun_g1_qs_resume \
  --account=gts-agarg35 \
  --partition=gpu-h100 \
  --qos=embers \
  --gres=gpu:h100:1 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=128G \
  --time=08:00:00 \
  --dependency=afterok:${RESUME_JOB} \
  --output="${LOG_DIR}/%x_%j.out" \
  --error="${LOG_DIR}/%x_%j.err" \
  --wrap="cd ${PROJECT_ROOT} && PYTHON_BIN=${PYTHON_BIN} ${PIPELINE_SCRIPT}"
