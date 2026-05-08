#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

INTERVAL_SECONDS="${MJLAB_QS_MONITOR_INTERVAL_SECONDS:-1800}"
LOG_DIR="logs/monitor"
LOG_FILE="${LOG_DIR}/mjlab_qs_tmux_monitor.log"

mkdir -p "$LOG_DIR"

echo "[$(date -Is)] Starting mjlab_qs monitor loop with interval=${INTERVAL_SECONDS}s"
echo "[$(date -Is)] Log file: ${LOG_FILE}"

while true; do
  {
    echo
    echo "===== $(date -Is) mjlab_qs monitor tick ====="
    rc=0
    ./scripts/monitor_mjlab_qs_with_codex.sh || rc=$?
    if [[ "$rc" -ne 0 ]]; then
      echo "[$(date -Is)] monitor command exited with rc=${rc}; continuing loop."
    fi
    echo "===== $(date -Is) mjlab_qs monitor tick complete rc=${rc} ====="
  } 2>&1 | tee -a "$LOG_FILE"

  sleep "$INTERVAL_SECONDS"
done
