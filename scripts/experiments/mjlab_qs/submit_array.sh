#!/usr/bin/env bash
set -euo pipefail

KIND=""
MANIFEST=""
GPU_TYPE="H100"
PARTITION="ice-gpu"
QOS="embers"
ACCOUNT="gts-agarg35"
MAX_CONCURRENT=2
TIME_LIMIT="04:00:00"
MEMORY="128G"
CPUS=8
PYTHON_BIN="python"
CONDA_ENV="${CONDA_ENV_NAME:-}"
DEPENDENCY=""
REQUIRE_FORMAL_METADATA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind) KIND="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    --require-formal-metadata) REQUIRE_FORMAL_METADATA=1; shift ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

if [[ -z "$KIND" || -z "$MANIFEST" ]]; then
  echo "--kind and --manifest are required" >&2
  exit 1
fi
if [[ "${QOS,,}" == "inferno" && "${ALLOW_INFERNO_QOS:-0}" != "1" ]]; then
  echo "Error: inferno QOS requires explicit user approval. Use embers for GPU jobs." >&2
  exit 1
fi

case "${GPU_TYPE}" in
  H100) GRES="gpu:h100:1" ;;
  H200) GRES="gpu:h200:1" ;;
  A100) GRES="gpu:a100:1" ;;
  L40S) GRES="gpu:l40s:1" ;;
  RTX6000|RTX_6000) GRES="gpu:rtx_6000:1" ;;
  PRO6000|RTXPRO6000|RTX_PRO_6000|RTX_PRO_6000_BLACKWELL) GRES="gpu:rtx_pro_6000_blackwell:1" ;;
  *) echo "unsupported gpu ${GPU_TYPE}" >&2; exit 1 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NUM_ROWS="$("${PYTHON_BIN}" - <<'PY' "${MANIFEST}"
import csv, sys
with open(sys.argv[1], newline='', encoding='utf-8') as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
)"
if [[ "${REQUIRE_FORMAL_METADATA}" == "1" ]]; then
  "${PYTHON_BIN}" - <<'PY' "${MANIFEST}" "${KIND}"
import csv
import sys

manifest, kind = sys.argv[1], sys.argv[2]
if kind not in {"policy_eval", "policy_rollout", "flow_mbpo_smoke", "flow_mbpo_replay"}:
    raise SystemExit(
        "--require-formal-metadata is only supported for policy_eval, policy_rollout, "
        "flow_mbpo_smoke, and flow_mbpo_replay"
    )

def present(row, key):
    return bool(str(row.get(key, "")).strip())

def disabled_wandb(row):
    return str(row.get("disable_wandb", "")).strip().lower() in {"1", "true", "yes"}

def enabled_wandb(row):
    return str(row.get("enable_wandb", "")).strip().lower() in {"1", "true", "yes"}

def baseline_present(row, prefix):
    return (
        present(row, f"{prefix}_baseline_return")
        and present(row, f"{prefix}_baseline_length")
        and present(row, f"{prefix}_baseline_fall")
    ) or (present(row, "baseline_return") and present(row, "baseline_length") and present(row, "baseline_fall"))

errors = []
with open(manifest, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
for idx, row in enumerate(rows):
    label = f"row {idx}"
    if kind in {"policy_eval", "policy_rollout"}:
        if disabled_wandb(row):
            errors.append(f"{label}: disable_wandb is set; formal metadata requires W&B enabled")
    else:
        if not enabled_wandb(row):
            errors.append(f"{label}: enable_wandb must be true for formal synthetic runs")
    for key in ("wandb_project", "wandb_group", "notes"):
        if not present(row, key):
            errors.append(f"{label}: missing {key}")
    if kind == "policy_eval":
        if not baseline_present(row, "eval"):
            errors.append(f"{label}: missing eval_baseline_* or baseline_* fields")
        if present(row, "policy_checkpoint") and not present(row, "eval_output_dir"):
            errors.append(f"{label}: direct checkpoint eval rows require eval_output_dir")
    if kind == "policy_rollout":
        if not baseline_present(row, "rollout"):
            errors.append(f"{label}: missing rollout_baseline_* or baseline_* fields")
        if present(row, "policy_checkpoint") and not present(row, "rollout_output_dir"):
            errors.append(f"{label}: direct checkpoint rollout rows require rollout_output_dir")
    if kind == "flow_mbpo_smoke":
        for key in ("dataset", "metadata", "normalization", "policy_checkpoint"):
            if not present(row, key):
                errors.append(f"{label}: missing {key}")
        if not (present(row, "wm_checkpoint") or present(row, "wm_checkpoints")):
            errors.append(f"{label}: missing wm_checkpoint or wm_checkpoints")
        if not (present(row, "output_dir") or present(row, "smoke_output_dir")):
            errors.append(f"{label}: flow_mbpo_smoke rows require output_dir or smoke_output_dir")
    if kind == "flow_mbpo_replay":
        if not present(row, "synthetic_buffer"):
            errors.append(f"{label}: missing synthetic_buffer")
        if not (present(row, "output_dir") or present(row, "replay_output_dir")):
            errors.append(f"{label}: flow_mbpo_replay rows require output_dir or replay_output_dir")
        if str(row.get("support_risk_termination", "")).strip().lower() in {"1", "true", "yes"}:
            for key in ("support_dataset", "support_metadata", "support_normalization"):
                if not present(row, key):
                    errors.append(f"{label}: support_risk_termination requires {key}")
    if (present(row, "policy_checkpoint") or kind in {"flow_mbpo_smoke", "flow_mbpo_replay"}) and not present(row, "wandb_name"):
        errors.append(f"{label}: formal direct-artifact rows require wandb_name")
if errors:
    raise SystemExit("Formal metadata validation failed:\n" + "\n".join(errors))
print(f"formal metadata validation passed for {len(rows)} {kind} rows")
PY
fi
ARRAY="0-$((NUM_ROWS - 1))%${MAX_CONCURRENT}"
LOG_DIR="${PROJECT_ROOT}/logs/slurm/mjlab_qs/${KIND}"
mkdir -p "${LOG_DIR}"

RUNNER="scripts/experiments/mjlab_qs/run_collection_row.py"
if [[ "$KIND" == "train" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_training_row.py"
elif [[ "$KIND" == "train_match" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_train_match_row.py"
elif [[ "$KIND" == "policy_extract" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_policy_extraction_row.py"
elif [[ "$KIND" == "policy_rollout" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_policy_rollout_row.py"
elif [[ "$KIND" == "policy_eval" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_policy_eval_row.py"
elif [[ "$KIND" == "flow_mbpo_smoke" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_flow_mbpo_smoke_row.py"
elif [[ "$KIND" == "flow_mbpo_replay" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_flow_mbpo_replay_row.py"
elif [[ "$KIND" == "original_pwm_adapter" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_original_pwm_adapter_row.py"
elif [[ "$KIND" == "native_collector" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_mjlab_native_collector_row.py"
elif [[ "$KIND" == "native_collection" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_native_collection_row.py"
elif [[ "$KIND" == "native_collector_rollout" ]]; then
  RUNNER="scripts/experiments/mjlab_qs/run_native_collector_rollout_row.py"
fi

WRAP="cd ${PROJECT_ROOT}"
if [[ -n "${CONDA_ENV}" ]]; then
  WRAP+=" && source ~/.bashrc && conda activate ${CONDA_ENV}"
fi
WRAP+=" && export PYTHONPATH=${PROJECT_ROOT}/src:\$PYTHONPATH"
WRAP+=" && export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless"
WRAP+=" && export WANDB_DIR=${PROJECT_ROOT}/scripts/outputs/mjlab_qs/wandb"
WRAP+=" && mkdir -p ${PROJECT_ROOT}/scripts/outputs/mjlab_qs/wandb"
WRAP+=" && ${PYTHON_BIN} ${RUNNER} --manifest ${MANIFEST} --row-index \$SLURM_ARRAY_TASK_ID --python-bin ${PYTHON_BIN}"

SBATCH_ARGS=(
  --job-name="mjqs_${KIND}_${GPU_TYPE}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --gres="${GRES}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS}" \
  --mem="${MEMORY}" \
  --time="${TIME_LIMIT}" \
  --array="${ARRAY}" \
  --output="${LOG_DIR}/mjqs_${KIND}_%A_%a.out" \
  --error="${LOG_DIR}/mjqs_${KIND}_%A_%a.err" \
  --wrap="${WRAP}"
)
if [[ -n "${DEPENDENCY}" ]]; then
  SBATCH_ARGS+=(--dependency="${DEPENDENCY}")
fi

sbatch "${SBATCH_ARGS[@]}"
