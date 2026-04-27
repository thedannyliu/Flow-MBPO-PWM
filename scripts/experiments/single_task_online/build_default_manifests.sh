#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MANIFEST_DIR="${PROJECT_ROOT}/scripts/experiments/single_task_online/manifests"
mkdir -p "${MANIFEST_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

${PYTHON_BIN} "${SCRIPT_DIR}/build_manifest.py" \
  --stage smoke \
  --output "${MANIFEST_DIR}/smoke_v1.csv"

${PYTHON_BIN} "${SCRIPT_DIR}/build_manifest.py" \
  --stage pilot \
  --output "${MANIFEST_DIR}/pilot_v1.csv"

${PYTHON_BIN} "${SCRIPT_DIR}/build_manifest.py" \
  --stage confirm \
  --output "${MANIFEST_DIR}/confirm_v1.csv"

echo "Built manifests under ${MANIFEST_DIR}"
