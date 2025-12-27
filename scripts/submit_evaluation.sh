#!/bin/bash
################################################################################
# Submit Evaluation Job to SLURM (H200)
#
# Usage:
#   ./scripts/submit_evaluation.sh [model_size] [task] [num_episodes]
#
# Examples:
#   ./scripts/submit_evaluation.sh 5M dflex_ant 100
#   ./scripts/submit_evaluation.sh 48M dflex_ant 100
################################################################################

MODEL_SIZE=${1:-5M}
TASK=${2:-dflex_ant}
NUM_EPISODES=${3:-100}
GPU_TYPE="H200"

# Job name
JOB_NAME="eval_${MODEL_SIZE}_${TASK}"

# Checkpoint paths
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

# Output directory
OUTPUT_DIR="evaluation_results/${MODEL_SIZE}_${TASK}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Submitting evaluation job:"
echo "  Model Size: $MODEL_SIZE"
echo "  Task: $TASK"
echo "  Episodes: $NUM_EPISODES"
echo "  GPU: $GPU_TYPE"
echo "  Baseline: $BASELINE_CKPT"
echo "  Flow: $FLOW_CKPT"
echo "  Output: $OUTPUT_DIR"
echo ""

# Create SLURM script
SLURM_SCRIPT="slurm_eval_${MODEL_SIZE}_${TASK}.sh"

cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH -J${JOB_NAME}
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:${GPU_TYPE}:1
#SBATCH --mem=32G
#SBATCH -t2:00:00
#SBATCH -qinferno
#SBATCH -Agts-agarg35
#SBATCH -oeval_logs/slurm/${JOB_NAME}_%j.out
#SBATCH -eeval_logs/slurm/${JOB_NAME}_%j.err

echo "=========================================="
echo "Job: ${JOB_NAME}"
echo "=========================================="
echo "Node: \$(hostname)"
echo "Date: \$(date)"
echo "Model Size: ${MODEL_SIZE}"
echo "Task: ${TASK}"
echo "Episodes: ${NUM_EPISODES}"
echo ""

# Load modules
module load anaconda3/2022.05
module load cuda/12.1

# Activate conda environment
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM
source activate pwm

echo "Python: \$(which python)"
echo "CUDA: \$(nvcc --version | grep release)"
echo ""

# GPU info
nvidia-smi
echo ""

# Run evaluation
echo "=========================================="
echo "Starting Evaluation"
echo "=========================================="

python scripts/evaluate_policy.py \\
    --baseline "${BASELINE_CKPT}" \\
    --flow "${FLOW_CKPT}" \\
    --env "${TASK}" \\
    --num-episodes ${NUM_EPISODES} \\
    --output "${OUTPUT_DIR}" \\
    --seed 42 \\
    --device cuda:0

EXIT_CODE=\$?

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "Exit code: \$EXIT_CODE"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

if [ \$EXIT_CODE -eq 0 ]; then
    echo "Results:"
    cat "${OUTPUT_DIR}/comparison.csv"
    echo ""
    echo "SUCCESS!"
else
    echo "FAILED with exit code \$EXIT_CODE"
fi

exit \$EXIT_CODE
EOF

# Create log directory
mkdir -p eval_logs/slurm

# Submit job
echo "Submitting job..."
sbatch "$SLURM_SCRIPT"

echo ""
echo "Job submitted!"
echo "Monitor with: squeue -u \$USER"
echo "View logs: tail -f eval_logs/slurm/${JOB_NAME}_*.out"
echo ""

# Clean up SLURM script after submission
# rm "$SLURM_SCRIPT"
