#!/bin/bash
################################################################################
# PACE Phoenix Job Submission Helper
# 
# Usage:
#   ./submit_job.sh single baseline ant 42
#   ./submit_job.sh multi multi_seed ant
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

declare -A GPU_ACCOUNT_MAP=(
    [H100]="gts-agarg35"
    [H200]="gts-agarg35"
    [A100]="gts-agarg35"
    [L40S]="gts-agarg35-ideas_l40s"
    [RTX6000]="gts-agarg35-ideas_l40s"   # adjust if cluster maps differently
    [RTX-PRO-BLACKWELL]="gts-agarg35-ideasci23_dgx"
)

declare -A GPU_GRES_BASE_MAP=(
    [H100]="gpu:H100"
    [H200]="gpu:h200"
    [A100]="gpu:A100"
    [L40S]="gpu:L40s"
    [RTX6000]="gpu:RTX6000"
    [RTX-PRO-BLACKWELL]="gpu:RTX-Pro-Blackwell"
)

declare -A GPU_PARTITION_MAP=(
    [H100]="gpu-h100"
    [H200]="gpu-h200"
    [A100]="gpu-a100"
    [L40S]="gpu-l40s"
    [RTX6000]="gpu-rtx6000"
    [RTX-PRO-BLACKWELL]="gpu-rtxpro-blackwell"
    [ANY]="gpu"
)

declare -A GPU_QOS_MAP=(
    [H100]="inferno"
    [H200]="inferno"
    [A100]="inferno"
    [L40S]="inferno"
    [RTX6000]="inferno"
    [RTX-PRO-BLACKWELL]="inferno"
    [ANY]="inferno"
)

declare -A GPU_CONSTRAINT_MAP=(
    [H100]="H100"
    [H200]="H200"
    [A100]="A100"
    [L40S]="L40s"
    [RTX6000]="RTX6000"
    [RTX-PRO-BLACKWELL]="RTX-Pro-Blackwell"
)

print_usage() {
    echo "Usage:"
    echo "  Single GPU:"
    echo "    $0 single <algorithm> <task> <seed> [gpu_type] [account]"
    echo ""
    echo "  Multi GPU:"
    echo "    $0 multi <strategy> <task> [base_seed] [gpu_type] [account]"
    echo ""
    echo "Examples:"
    echo "  # Single GPU, baseline, ant task, seed 42 (default H200)"
    echo "  $0 single pwm_48M dflex_ant 42"
    echo "  # Single GPU, flow on L40s"
    echo "  $0 single pwm_48M_flow dflex_ant 42 L40S gts-agarg35-ideas_l40s"
    echo ""
    echo "  # Single GPU, flow, humanoid task, seed 123"
    echo "  $0 single pwm_48M_flow dflex_humanoid 123"
    echo ""
    echo "  # Multi GPU, 4 seeds, ant task, request H200"
    echo "  $0 multi multi_seed dflex_ant 42 H200"
    echo "  $0 multi baseline_vs_flow dflex_ant 42 'H100|H200'"
    echo ""
    echo "  # Multi GPU, 4 different tasks"
    echo "  $0 multi multi_task dflex_ant"
    echo ""
    echo "  # Multi GPU, baseline vs flow comparison (2 seeds each)"
    echo "  $0 multi baseline_vs_flow dflex_ant 42"
    echo ""
    echo "Algorithms:"
    echo "  pwm_48M          - Baseline MLP dynamics"
    echo "  pwm_48M_flow     - Flow-matching dynamics"
    echo ""
    echo "Tasks:"
    echo "  dflex_ant"
    echo "  dflex_humanoid"
    echo "  dflex_hopper"
    echo "  dflex_anymal"
    echo ""
    echo "Multi-GPU Strategies:"
    echo "  multi_seed       - Same task, 4 different seeds"
    echo "  multi_task       - 4 different tasks, same seed"
    echo "  baseline_vs_flow - Compare baseline vs flow (2 seeds each)"
    echo ""
    echo "Notes:"
    echo "  - gpu_type accepts single values (H100, H200, L40S, RTX6000, RTX-Pro-Blackwell, etc.)"
    echo "    or '|' separated lists (e.g., 'H100|H200')."
    echo "  - When mixing GPU families that belong to different accounts (e.g., adding L40s),"
    echo "    specify the desired billing account explicitly as the final argument."
}

normalize_gpu_spec() {
    local raw="$1"
    if [ -z "$raw" ]; then
        raw="H200"
    fi
    raw=${raw//,/|}
    raw=${raw// /}
    echo "$raw"
}

canonicalize_gpu_type() {
    local token="$1"
    token=${token#GPU:}
    token=$(echo "$token" | tr '[:lower:]' '[:upper:]')
    token=${token//_/}
    case "$token" in
        "" ) echo "" ;;
        "ANY"|"ALL") echo "ANY" ;;
        "H100") echo "H100" ;;
        "H200") echo "H200" ;;
        "A100") echo "A100" ;;
        "L40"|"L40S") echo "L40S" ;;
        "RTX6000"|"RTX-6000") echo "RTX6000" ;;
        "RTXPROBLACKWELL"|"RTX-PRO-BLACKWELL"|"RTXPRO-BLACKWELL"|"BLACKWELL") echo "RTX-PRO-BLACKWELL" ;;
        *) echo "" ;;
    esac
}

GPU_TYPE_DISPLAY=""
GPU_ACCOUNT_RESOLVED=""
GPU_GRES_RESOLVED=""
GPU_CONSTRAINT_RESOLVED=""
GPU_PARTITION_RESOLVED=""
GPU_QOS_RESOLVED=""

resolve_gpu_settings() {
    local requested="$(normalize_gpu_spec "$1")"
    local gpu_count="$2"
    local account_override="$3"

    IFS='|' read -ra raw_types <<< "$requested"
    if [ ${#raw_types[@]} -eq 0 ]; then
        raw_types=("H100")
    fi

    declare -A seen_types=()
    local canonical_types=()
    local base_account=""
    local conflict=false

    for raw in "${raw_types[@]}"; do
        [ -z "$raw" ] && continue
        local canon="$(canonicalize_gpu_type "$raw")"
        if [ -z "$canon" ]; then
            echo -e "${RED}Error: Unsupported GPU type '$raw'.${NC}"
            exit 1
        fi
        if [ "$canon" = "ANY" ]; then
            canonical_types=("ANY")
            base_account=${base_account:-${GPU_ACCOUNT_MAP[H200]}}
            break
        fi
        if [ -z "${seen_types[$canon]}" ]; then
            seen_types[$canon]=1
            canonical_types+=("$canon")
            local acct="${GPU_ACCOUNT_MAP[$canon]}"
            if [ -z "$acct" ]; then
                echo -e "${RED}Error: No account mapping for GPU type '$canon'.${NC}"
                exit 1
            fi
            if [ -z "$base_account" ]; then
                base_account="$acct"
            elif [ "$base_account" != "$acct" ]; then
                conflict=true
            fi
        fi
    done

    if [ ${#canonical_types[@]} -eq 0 ]; then
        canonical_types=("H200")
        base_account="${GPU_ACCOUNT_MAP[H200]}"
    fi

    if [ -n "$account_override" ]; then
        GPU_ACCOUNT_RESOLVED="$account_override"
        if $conflict; then
            echo -e "${YELLOW}Warning: Using account override '$account_override' for GPU types requiring multiple allocation groups.${NC}"
        fi
    else
        if $conflict; then
            echo -e "${RED}Error: Requested GPU types span multiple allocation accounts. Please specify an account explicitly or split submissions.${NC}"
            exit 1
        fi
        GPU_ACCOUNT_RESOLVED="$base_account"
    fi

    if [ ${#canonical_types[@]} -eq 1 ] && [ "${canonical_types[0]}" != "ANY" ]; then
        local canon="${canonical_types[0]}"
        local gres_base="${GPU_GRES_BASE_MAP[$canon]}"
        if [ -z "$gres_base" ]; then
            gres_base="gpu"
        fi
        GPU_GRES_RESOLVED="${gres_base}:${gpu_count}"
        GPU_CONSTRAINT_RESOLVED=""
        GPU_PARTITION_RESOLVED="${GPU_PARTITION_MAP[$canon]}"
        GPU_QOS_RESOLVED="${GPU_QOS_MAP[$canon]}"
    else
        GPU_GRES_RESOLVED="gpu:${gpu_count}"
        local constraints=()
        if [ "${canonical_types[0]}" != "ANY" ]; then
            for canon in "${canonical_types[@]}"; do
                constraints+=("${GPU_CONSTRAINT_MAP[$canon]}")
            done
        fi
        if [ ${#constraints[@]} -gt 0 ]; then
            GPU_CONSTRAINT_RESOLVED=$(IFS='|'; echo "${constraints[*]}")
        else
            GPU_CONSTRAINT_RESOLVED=""
        fi
        GPU_PARTITION_RESOLVED="${GPU_PARTITION_MAP[ANY]}"
        GPU_QOS_RESOLVED="${GPU_QOS_MAP[ANY]}"
    fi

    GPU_TYPE_DISPLAY=$(IFS='|'; echo "${canonical_types[*]}")
}

# Check arguments
if [ $# -lt 3 ]; then
    print_usage
    exit 1
fi

MODE=$1
shift

case $MODE in
    "single")
        if [ $# -lt 3 ] || [ $# -gt 5 ]; then
            echo -e "${RED}Error: Single GPU mode requires 3–5 arguments${NC}"
            print_usage
            exit 1
        fi

        ALGORITHM=$1
        TASK=$2
        SEED=$3
        GPU_SPEC=${4:-H200}
        ACCOUNT_OVERRIDE=${5:-}

        resolve_gpu_settings "$GPU_SPEC" 1 "$ACCOUNT_OVERRIDE"

        echo -e "${GREEN}Submitting single GPU job...${NC}"
        echo "  Algorithm: $ALGORITHM"
        echo "  Task: $TASK"
        echo "  Seed: $SEED"
        echo "  GPU Request: $GPU_TYPE_DISPLAY"
        echo "  Account: $GPU_ACCOUNT_RESOLVED"
        echo "  GRES: $GPU_GRES_RESOLVED"
        if [ -n "$GPU_PARTITION_RESOLVED" ]; then
            echo "  Partition: $GPU_PARTITION_RESOLVED"
        fi
        if [ -n "$GPU_QOS_RESOLVED" ]; then
            echo "  QoS: $GPU_QOS_RESOLVED"
        fi
        if [ -n "$GPU_CONSTRAINT_RESOLVED" ]; then
            echo "  Constraint: $GPU_CONSTRAINT_RESOLVED"
        fi

        JOB_LABEL="${ALGORITHM}_${TASK}_seed${SEED}"
        SBATCH_ARGS=(
            "--account=$GPU_ACCOUNT_RESOLVED"
            "--gres=$GPU_GRES_RESOLVED"
            "--job-name=$JOB_LABEL"
            "--output=logs/slurm/${JOB_LABEL}_%j.out"
            "--error=logs/slurm/${JOB_LABEL}_%j.err"
        )
        if [ -n "$GPU_PARTITION_RESOLVED" ]; then
            SBATCH_ARGS+=("--partition=$GPU_PARTITION_RESOLVED")
        fi
        if [ -n "$GPU_QOS_RESOLVED" ]; then
            SBATCH_ARGS+=("--qos=$GPU_QOS_RESOLVED")
        fi
        if [ -n "$GPU_CONSTRAINT_RESOLVED" ]; then
            SBATCH_ARGS+=("--constraint=$GPU_CONSTRAINT_RESOLVED")
        fi
        EXPORT_VARS="ALL,TASK=$TASK,ALGORITHM=$ALGORITHM,SEED=$SEED,GPU_TYPE=$GPU_TYPE_DISPLAY,GPU_ACCOUNT=$GPU_ACCOUNT_RESOLVED"
        if [ -n "$GPU_PARTITION_RESOLVED" ]; then
            EXPORT_VARS="$EXPORT_VARS,GPU_PARTITION=$GPU_PARTITION_RESOLVED"
        fi
        if [ -n "$GPU_QOS_RESOLVED" ]; then
            EXPORT_VARS="$EXPORT_VARS,GPU_QOS=$GPU_QOS_RESOLVED"
        fi
        if [ -n "$GPU_CONSTRAINT_RESOLVED" ]; then
            EXPORT_VARS="$EXPORT_VARS,GPU_CONSTRAINT=$GPU_CONSTRAINT_RESOLVED"
        fi
        SBATCH_ARGS+=("--export=$EXPORT_VARS" "scripts/slurm_single_gpu.sh")

        sbatch "${SBATCH_ARGS[@]}"
        ;;

    "multi")
        if [ $# -lt 2 ] || [ $# -gt 5 ]; then
            echo -e "${RED}Error: Multi GPU mode requires 2–5 arguments${NC}"
            print_usage
            exit 1
        fi

        STRATEGY=$1
        TASK=$2
        BASE_SEED=${3:-42}
        GPU_SPEC=${4:-H200}
        ACCOUNT_OVERRIDE=${5:-}

        resolve_gpu_settings "$GPU_SPEC" 4 "$ACCOUNT_OVERRIDE"

        echo -e "${GREEN}Submitting multi GPU job...${NC}"
        echo "  Strategy: $STRATEGY"
        echo "  Task: $TASK"
        echo "  Base Seed: $BASE_SEED"
        echo "  GPU Request: $GPU_TYPE_DISPLAY"
        echo "  Account: $GPU_ACCOUNT_RESOLVED"
        echo "  GRES: $GPU_GRES_RESOLVED"
        if [ -n "$GPU_PARTITION_RESOLVED" ]; then
            echo "  Partition: $GPU_PARTITION_RESOLVED"
        fi
        if [ -n "$GPU_QOS_RESOLVED" ]; then
            echo "  QoS: $GPU_QOS_RESOLVED"
        fi
        if [ -n "$GPU_CONSTRAINT_RESOLVED" ]; then
            echo "  Constraint: $GPU_CONSTRAINT_RESOLVED"
        fi

        JOB_LABEL="multi_${STRATEGY}_${TASK}_seed${BASE_SEED}"
        SBATCH_ARGS=(
            "--account=$GPU_ACCOUNT_RESOLVED"
            "--gres=$GPU_GRES_RESOLVED"
            "--job-name=$JOB_LABEL"
            "--output=logs/slurm/${JOB_LABEL}_%j.out"
            "--error=logs/slurm/${JOB_LABEL}_%j.err"
        )
        if [ -n "$GPU_PARTITION_RESOLVED" ]; then
            SBATCH_ARGS+=("--partition=$GPU_PARTITION_RESOLVED")
        fi
        if [ -n "$GPU_QOS_RESOLVED" ]; then
            SBATCH_ARGS+=("--qos=$GPU_QOS_RESOLVED")
        fi
        if [ -n "$GPU_CONSTRAINT_RESOLVED" ]; then
            SBATCH_ARGS+=("--constraint=$GPU_CONSTRAINT_RESOLVED")
        fi
        EXPORT_VARS="ALL,STRATEGY=$STRATEGY,TASK=$TASK,SEED=$BASE_SEED,GPU_TYPE=$GPU_TYPE_DISPLAY,GPU_ACCOUNT=$GPU_ACCOUNT_RESOLVED"
        if [ -n "$GPU_PARTITION_RESOLVED" ]; then
            EXPORT_VARS="$EXPORT_VARS,GPU_PARTITION=$GPU_PARTITION_RESOLVED"
        fi
        if [ -n "$GPU_QOS_RESOLVED" ]; then
            EXPORT_VARS="$EXPORT_VARS,GPU_QOS=$GPU_QOS_RESOLVED"
        fi
        if [ -n "$GPU_CONSTRAINT_RESOLVED" ]; then
            EXPORT_VARS="$EXPORT_VARS,GPU_CONSTRAINT=$GPU_CONSTRAINT_RESOLVED"
        fi
        SBATCH_ARGS+=("--export=$EXPORT_VARS" "scripts/slurm_multi_gpu.sh")

        sbatch "${SBATCH_ARGS[@]}"
        ;;

    *)
        echo -e "${RED}Error: Unknown mode '$MODE'${NC}"
        print_usage
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Job submitted successfully!${NC}"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f logs/slurm/*.out"
    echo ""
    echo "Cancel with:"
    echo "  scancel <job_id>"
else
    echo -e "${RED}Job submission failed!${NC}"
    exit 1
fi
