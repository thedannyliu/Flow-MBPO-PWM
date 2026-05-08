#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

REPORT_DIR="reports/monitor"
LOG_DIR="logs/monitor"
CONTEXT_FILE="${REPORT_DIR}/mjlab_qs_current_context.md"
GATE_FILE="${REPORT_DIR}/mjlab_qs_gate.json"
LAST_JSON="${REPORT_DIR}/mjlab_qs_last_triage.json"
LAST_MD="${REPORT_DIR}/mjlab_qs_last_monitor.md"
SEEN_FILE="${REPORT_DIR}/mjlab_qs_seen_failures.json"
LOCK_FILE="${REPORT_DIR}/mjlab_qs_monitor.lock"
RUN_LOG="${LOG_DIR}/mjlab_qs_codex_exec.log"
PROMPT_FILE="prompts/mjlab_qs_monitor_prompt.md"
SCHEMA_FILE="schemas/mjlab_qs_monitor_triage.schema.json"
CODEX_BIN="${CODEX_BIN:-codex}"

mkdir -p "$REPORT_DIR" "$LOG_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date -Is)] Previous mjlab_qs monitor run is still active. Skipping."
  exit 0
fi

write_report_json() {
  local status="$1"
  local severity="$2"
  local failure_kind="$3"
  local root_cause="$4"
  local next_action="$5"
  python - "$LAST_JSON" "$status" "$severity" "$failure_kind" "$root_cause" "$next_action" <<'PY'
from __future__ import annotations

import json
import sys

path, status, severity, failure_kind, root_cause, next_action = sys.argv[1:7]
payload = {
    "status": status,
    "severity": severity,
    "failure_kind": failure_kind,
    "root_cause": root_cause,
    "evidence": [],
    "actions_taken": ["Collected Slurm and mjlab_qs context."],
    "commands_run": ["bash scripts/collect_mjlab_qs_monitor_context.sh"],
    "files_changed": [],
    "tests_run": [],
    "suggested_probe_command": "",
    "suggested_retry_command": "",
    "needs_human": status in {"needs_human", "blocked", "failed"},
    "next_action": next_action,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}

write_markdown_summary() {
  {
    echo "# mjlab_qs last monitor"
    echo
    echo "- time: $(date -Is)"
    echo "- context: $CONTEXT_FILE"
    echo "- gate: $GATE_FILE"
    echo "- triage: $LAST_JSON"
    echo
    echo "## JSON"
    echo
    if [[ -f "$LAST_JSON" ]]; then
      cat "$LAST_JSON"
    else
      echo "No triage JSON written."
    fi
  } > "$LAST_MD"
}

echo "[$(date -Is)] Collecting mjlab_qs monitor context..."
bash scripts/collect_mjlab_qs_monitor_context.sh > "$CONTEXT_FILE"

python - "$CONTEXT_FILE" "$SEEN_FILE" "$GATE_FILE" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

context_path = Path(sys.argv[1])
seen_path = Path(sys.argv[2])
gate_path = Path(sys.argv[3])
text = context_path.read_text(encoding="utf-8", errors="replace")

patterns = [
    r"\bFAILED\b",
    r"\bOUT_OF_MEMORY\b",
    r"\bTIMEOUT\b",
    r"\bNODE_FAIL\b",
    r"\bPREEMPTED\b",
    r"ExitCode\|.*[1-9][0-9]*:0",
    r"Traceback",
    r"RuntimeError",
    r"ValueError",
    r"KeyError",
    r"ModuleNotFoundError",
    r"FileNotFoundError",
    r"CUDA out of memory",
    r"CUDA error",
    r"device-side assert",
    r"\bNaN\b",
    r"nan loss",
    r"loss exploded",
    r"Disk quota exceeded",
    r"No space left on device",
    r"Missing ",
    r"missing ",
    r"corrupt",
    r"PytorchStreamReader",
    r"Error executing job",
]
regex = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
target_hint = re.compile(
    r"(?:\bmjqs\b|mjlab_qs|sto_mjlab_qs|logs/slurm/mjlab_qs|scripts/outputs/mjlab_qs)",
    re.IGNORECASE,
)
matched_lines: list[str] = []
in_mjlab_log_block = False
current_log_path = ""
for line in text.splitlines():
    if line.startswith("## "):
        in_mjlab_log_block = False
        current_log_path = ""
    elif line.startswith("### "):
        in_mjlab_log_block = "logs/slurm/mjlab_qs" in line
        current_log_path = line.removeprefix("### ").strip()

    # The context includes all user Slurm accounting for orientation, but the
    # gate should only wake Codex for the current mjlab_qs target.
    relevant = bool(target_hint.search(line)) or in_mjlab_log_block
    if relevant and regex.search(line):
        if in_mjlab_log_block and current_log_path:
            matched_lines.append(f"{current_log_path}: {line}"[:700])
        else:
            matched_lines.append(line[:500])

job_ids: set[str] = set()
for line in matched_lines:
    for match in re.findall(r"(?<!\d)\d{5,}(?:_\d+)?(?:\.(?:batch|extern))?", line):
        job_ids.add(match)

if matched_lines:
    fingerprint_source = "\n".join(matched_lines[-120:])
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8", errors="replace")).hexdigest()
else:
    fingerprint = ""

seen = {"fingerprints": []}
if seen_path.exists():
    try:
        seen = json.loads(seen_path.read_text(encoding="utf-8"))
    except Exception:
        seen = {"fingerprints": []}
seen_fingerprints = set(seen.get("fingerprints", []))

if not matched_lines:
    status = "no_failure_signal"
elif fingerprint in seen_fingerprints:
    status = "known_failure_signal"
else:
    status = "new_failure_signal"

payload = {
    "status": status,
    "fingerprint": fingerprint,
    "job_ids": sorted(job_ids),
    "match_count": len(matched_lines),
    "matches": matched_lines[-40:],
}
gate_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

GATE_STATUS="$(python - "$GATE_FILE" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)"

case "$GATE_STATUS" in
  no_failure_signal)
    echo "[$(date -Is)] No actionable failure signal detected."
    write_report_json "running" "none" "" "" "Continue monitoring."
    write_markdown_summary
    exit 0
    ;;
  known_failure_signal)
    echo "[$(date -Is)] Failure signal already triaged and unchanged."
    write_report_json "failed" "low" "known_failure_signal" "The current failure fingerprint has already been triaged." "Review the previous triage or wait for a changed failure signal."
    write_markdown_summary
    exit 0
    ;;
  new_failure_signal)
    echo "[$(date -Is)] New failure-like signal detected."
    ;;
  *)
    echo "[$(date -Is)] Unknown gate status: $GATE_STATUS" >&2
    write_report_json "blocked" "medium" "monitor_gate_error" "Monitor gate returned an unknown status." "Inspect $GATE_FILE."
    write_markdown_summary
    exit 1
    ;;
esac

if [[ "${CODEX_MONITOR_DISABLE_CODEX:-0}" == "1" ]]; then
  echo "[$(date -Is)] CODEX_MONITOR_DISABLE_CODEX=1; not invoking Codex."
  write_report_json "failed" "medium" "new_failure_signal" "A new failure-like signal was detected, but Codex invocation is disabled." "Inspect $CONTEXT_FILE and $GATE_FILE, or rerun with CODEX_MONITOR_DISABLE_CODEX=0."
  write_markdown_summary
  exit 0
fi

if [[ ! -f "$PROMPT_FILE" || ! -f "$SCHEMA_FILE" ]]; then
  echo "[$(date -Is)] Missing prompt or schema file." >&2
  write_report_json "blocked" "high" "monitor_config_missing" "Missing prompt or schema file for Codex triage." "Check $PROMPT_FILE and $SCHEMA_FILE."
  write_markdown_summary
  exit 1
fi

echo "[$(date -Is)] Running Codex triage..."
PROMPT_TEXT="$(< "$PROMPT_FILE")"
if "$CODEX_BIN" exec \
  --sandbox workspace-write \
  --output-schema "$SCHEMA_FILE" \
  --output-last-message "$LAST_JSON" \
  "$PROMPT_TEXT" \
  < "$CONTEXT_FILE" \
  >> "$RUN_LOG" 2>&1; then
  python - "$SEEN_FILE" "$GATE_FILE" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

seen_path = Path(sys.argv[1])
gate_path = Path(sys.argv[2])
gate = json.loads(gate_path.read_text(encoding="utf-8"))
seen = {"fingerprints": [], "entries": []}
if seen_path.exists():
    try:
        seen = json.loads(seen_path.read_text(encoding="utf-8"))
    except Exception:
        seen = {"fingerprints": [], "entries": []}
fingerprint = gate.get("fingerprint", "")
if fingerprint:
    fingerprints = list(dict.fromkeys([*seen.get("fingerprints", []), fingerprint]))
    seen["fingerprints"] = fingerprints[-200:]
    entries = seen.get("entries", [])
    entries.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "job_ids": gate.get("job_ids", []),
        "match_count": gate.get("match_count", 0),
    })
    seen["entries"] = entries[-200:]
seen_path.write_text(json.dumps(seen, indent=2) + "\n", encoding="utf-8")
PY
  write_markdown_summary
  echo "[$(date -Is)] Codex triage complete. See $LAST_JSON"
else
  rc=$?
  echo "[$(date -Is)] Codex triage failed with rc=$rc. See $RUN_LOG" >&2
  write_report_json "blocked" "high" "codex_exec_failed" "codex exec returned nonzero status $rc." "Inspect $RUN_LOG and rerun the monitor after fixing Codex/auth/environment."
  write_markdown_summary
  exit "$rc"
fi
