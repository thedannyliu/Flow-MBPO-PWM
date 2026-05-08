#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

SINCE="${1:-now-2days}"
LOG_MMIN="${MJLAB_QS_MONITOR_LOG_MMIN:-1440}"
MAX_LOG_FILES="${MJLAB_QS_MONITOR_MAX_LOG_FILES:-40}"
TAIL_LINES="${MJLAB_QS_MONITOR_TAIL_LINES:-140}"
MAX_JSON_FILES="${MJLAB_QS_MONITOR_MAX_JSON_FILES:-24}"

print_cmd() {
  printf '$'
  printf ' %q' "$@"
  printf '\n'
  "$@" || true
}

echo "# mjlab_qs monitor context"
echo
echo "timestamp: $(date -Is)"
echo "host: $(hostname)"
echo "user: ${USER:-unknown}"
echo "repo: $REPO_ROOT"
echo "since: $SINCE"
echo

echo "## Git"
print_cmd git branch --show-current
print_cmd git rev-parse --short HEAD
print_cmd git status --short
echo

echo "## Slurm active jobs"
if command -v squeue >/dev/null 2>&1; then
  print_cmd squeue -u "${USER:-}" -o "%.18i|%.40j|%.12T|%.12M|%.12L|%R"
else
  echo "squeue not available."
fi
echo

echo "## Slurm recent accounting"
if command -v sacct >/dev/null 2>&1; then
  print_cmd sacct -u "${USER:-}" \
    --starttime "$SINCE" \
    --format=JobID,JobName%50,State%24,ExitCode,Elapsed,Timelimit,Submit,Start,End,NodeList%40,ReqTRES%80,AllocTRES%80 \
    -P
else
  echo "sacct not available."
fi
echo

echo "## mjlab_qs recent terminal states"
if command -v sacct >/dev/null 2>&1; then
  sacct -u "${USER:-}" \
    --starttime "$SINCE" \
    --format=JobID,JobName%50,State%24,ExitCode,Elapsed,Timelimit,End \
    -P 2>/dev/null \
    | awk -F'|' 'NR == 1 || ($2 ~ /mjqs|mjlab|sto_mjlab_qs/ && $3 !~ /^(RUNNING|PENDING|COMPLETED)$/)' \
    || true
else
  echo "sacct not available."
fi
echo

echo "## Disk and quota hints"
print_cmd df -h .
if command -v pace-quota >/dev/null 2>&1; then
  print_cmd pace-quota
elif command -v quota >/dev/null 2>&1; then
  print_cmd quota -s
else
  echo "No pace-quota/quota command available."
fi
for path in logs/slurm/mjlab_qs scripts/outputs/mjlab_qs scripts/outputs/mjlab_qs/wandb wandb; do
  if [[ -e "$path" ]]; then
    print_cmd du -sh "$path"
  fi
done
echo

echo "## Recent mjlab_qs manifests"
find scripts/outputs/mjlab_qs scripts/experiments/mjlab_qs scripts/experiments/single_task_online/manifests \
  -type f \( -name "*.csv" -o -name "*.yaml" -o -name "*.yml" \) \
  -mtime -14 \
  -printf "%T@ %p\n" 2>/dev/null \
  | sort -nr \
  | head -40 \
  | cut -d' ' -f2- \
  || true
echo

echo "## Recent mjlab_qs JSON summaries"
python - "$MAX_JSON_FILES" <<'PY' || true
from __future__ import annotations

import json
import sys
from pathlib import Path

limit = int(sys.argv[1])
roots = [Path("scripts/outputs/mjlab_qs")]
names = {"summary.json", "train_match_summary.json", "eval_summary.json"}
paths: list[Path] = []
for root in roots:
    if root.exists():
        paths.extend(p for p in root.rglob("*.json") if p.name in names)
paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
for path in paths[:limit]:
    print()
    print(f"### {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"could not parse json: {exc}")
        continue
    if isinstance(data, dict):
        keys = [
            "stage",
            "task",
            "task_key",
            "method",
            "seed",
            "status",
            "success",
            "num_episodes",
            "num_windows",
            "mean_return",
            "episode_length_mean",
            "output_path",
            "checkpoint",
        ]
        compact = {key: data[key] for key in keys if key in data}
        if compact:
            print(json.dumps(compact, sort_keys=True))
        else:
            rendered = json.dumps(data, sort_keys=True)
            print(rendered[:2000])
    else:
        rendered = json.dumps(data)
        print(rendered[:2000])
PY
echo

echo "## Failure-like recent mjlab_qs logs"
FAILURE_REGEX='FAILED|OUT_OF_MEMORY|TIMEOUT|NODE_FAIL|PREEMPTED|Traceback|RuntimeError|ValueError|KeyError|ModuleNotFoundError|FileNotFoundError|CUDA out of memory|CUDA error|device-side assert|NaN|nan loss|loss exploded|Disk quota exceeded|No space left on device|Hydra|Missing|missing|corrupt|PytorchStreamReader|Error executing job'
mapfile -t failure_logs < <(
  find logs/slurm/mjlab_qs -type f \( -name "*.out" -o -name "*.err" \) -mmin "-$LOG_MMIN" -print 2>/dev/null \
    | while IFS= read -r f; do
        if grep -Eiq "$FAILURE_REGEX" "$f"; then
          printf "%s\n" "$f"
        fi
      done \
    | sort \
    | tail -"$MAX_LOG_FILES"
)
if [[ "${#failure_logs[@]}" -eq 0 ]]; then
  echo "No failure-like recent mjlab_qs logs found."
else
  for f in "${failure_logs[@]}"; do
    echo
    echo "### $f"
    grep -Ein "$FAILURE_REGEX" "$f" | tail -20 || true
    echo
    echo "--- tail ---"
    tail -n "$TAIL_LINES" "$f" || true
  done
fi
echo

echo "## Latest recent mjlab_qs logs"
find logs/slurm/mjlab_qs -type f \( -name "*.out" -o -name "*.err" \) -mmin "-$LOG_MMIN" -printf "%T@ %p\n" 2>/dev/null \
  | sort -nr \
  | head -"$MAX_LOG_FILES" \
  | cut -d' ' -f2- \
  | while IFS= read -r f; do
      echo
      echo "### $f"
      tail -n "$TAIL_LINES" "$f" || true
    done \
  || true
