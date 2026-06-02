#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LEWM_ENV="${LEWM_ENV:-/storage/project/r-agarg35-0/eliu354/envs/lewm_official_20260602}"
VENDOR_DIR="${ROOT}/scripts/experiments/image_official/compat/vendor"

mkdir -p "${VENDOR_DIR}"
"${LEWM_ENV}/bin/python" -m pip install --no-cache-dir --target "${VENDOR_DIR}" hdf5plugin==6.0.0
