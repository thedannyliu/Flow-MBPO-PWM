#!/bin/bash
################################################################################
# Evaluate PWM Policies - Baseline vs Flow Comparison
#
# This script evaluates trained policies and compares baseline vs flow dynamics
#
# Usage:
#   ./scripts/run_evaluation.sh [model_size] [task]
#
# Examples:
#   ./scripts/run_evaluation.sh 5M dflex_ant      # Evaluate 5M models on ant
#   ./scripts/run_evaluation.sh 48M dflex_ant     # Evaluate 48M models on ant
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

MODEL_SIZE=${1:-5M}
TASK=${2:-dflex_ant}
NUM_EPISODES=${3:-100}

echo -e "${GREEN}================================"
echo "PWM Policy Evaluation"
echo "================================${NC}"
echo "Model Size: $MODEL_SIZE"
echo "Task: $TASK"
echo "Episodes: $NUM_EPISODES"
echo ""

# Set checkpoint paths based on training outputs
if [ "$MODEL_SIZE" = "5M" ]; then
    BASELINE_CKPT="outputs/2025-11-08/23-48-46/logs/pwm_5M_dflex_ant_seed42/best_policy.pt"
    FLOW_CKPT="outputs/2025-11-09/06-18-53/logs/pwm_5M_flow_dflex_ant_seed42/best_policy.pt"
elif [ "$MODEL_SIZE" = "48M" ]; then
    BASELINE_CKPT="outputs/2025-11-09/02-50-34/logs/pwm_48M_dflex_ant_seed42/best_policy.pt"
    FLOW_CKPT="outputs/2025-11-09/10-49-03/logs/pwm_48M_flow_dflex_ant_seed42/best_policy.pt"
else
    echo "Error: Invalid model size. Use 5M or 48M"
    exit 1
fi

# Check if checkpoints exist
if [ ! -f "$BASELINE_CKPT" ]; then
    echo -e "${YELLOW}Warning: Baseline checkpoint not found: $BASELINE_CKPT${NC}"
    echo "Available checkpoints:"
    find outputs -name "best_policy.pt" | grep baseline
    exit 1
fi

if [ ! -f "$FLOW_CKPT" ]; then
    echo -e "${YELLOW}Warning: Flow checkpoint not found: $FLOW_CKPT${NC}"
    echo "Available checkpoints:"
    find outputs -name "best_policy.pt" | grep flow
    exit 1
fi

# Create output directory
OUTPUT_DIR="evaluation_results/${MODEL_SIZE}_${TASK}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Checkpoint paths:"
echo "  Baseline: $BASELINE_CKPT"
echo "  Flow:     $FLOW_CKPT"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate pwm
else
    echo "Warning: conda not found"
fi

# Run evaluation
echo -e "${GREEN}Starting evaluation...${NC}"
echo ""

python scripts/evaluate_policy.py \
    --baseline "$BASELINE_CKPT" \
    --flow "$FLOW_CKPT" \
    --env "$TASK" \
    --num-episodes "$NUM_EPISODES" \
    --output "$OUTPUT_DIR" \
    --seed 42

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================"
    echo "Evaluation Complete!"
    echo "================================${NC}"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "View results:"
    echo "  cat $OUTPUT_DIR/comparison.csv"
    echo "  open $OUTPUT_DIR/comparison.png"
    echo ""
else
    echo ""
    echo "Evaluation failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi
