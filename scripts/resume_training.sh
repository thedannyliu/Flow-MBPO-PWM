#!/bin/bash
#
# Resume training from checkpoint
# Usage: ./scripts/resume_training.sh <checkpoint_path> [gpu_type] [additional_args]
#
# Examples:
#   ./scripts/resume_training.sh outputs/2025-11-08/02-32-38/logs/pwm_5M_flow_dflex_ant_seed42/checkpoint_1000.pt
#   ./scripts/resume_training.sh outputs/2025-11-08/02-32-38/logs/pwm_5M_flow_dflex_ant_seed42/checkpoint_1000.pt H100
#   ./scripts/resume_training.sh path/to/checkpoint.pt H200 general.seed=43
#

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Error: Checkpoint path required"
    echo "Usage: $0 <checkpoint_path> [gpu_type] [additional_args]"
    exit 1
fi

CHECKPOINT_PATH="$1"
GPU_TYPE="${2:-H200}"  # Default to H200
shift 2 || shift 1  # Remove first 1-2 arguments
EXTRA_ARGS="$@"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# Get absolute path
CHECKPOINT_PATH=$(realpath "$CHECKPOINT_PATH")

# Extract algorithm and task from checkpoint path
# Path format: outputs/DATE/TIME/logs/ALG_TASK_seedXX/checkpoint_XXX.pt
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
DIR_NAME=$(basename "$CHECKPOINT_DIR")

# Try to parse algorithm and task from directory name
# Format: pwm_5M_flow_dflex_ant_seed42
if [[ "$DIR_NAME" =~ ([^_]+(_[0-9]+[KM])?(_flow)?)_([^_]+_[^_]+)_seed([0-9]+) ]]; then
    ALG="${BASH_REMATCH[1]}"
    TASK="${BASH_REMATCH[4]}"
    SEED="${BASH_REMATCH[5]}"
else
    echo "Error: Cannot parse algorithm/task from directory name: $DIR_NAME"
    echo "Expected format: ALG_TASK_seedXX"
    exit 1
fi

echo "======================================"
echo "Resuming Training from Checkpoint"
echo "======================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Algorithm: $ALG"
echo "Task: $TASK"
echo "Seed: $SEED"
echo "GPU Type: $GPU_TYPE"
echo "Extra Args: $EXTRA_ARGS"
echo "======================================"

# Check if buffer file exists
BUFFER_PATH="${CHECKPOINT_PATH%.pt}.buffer"
if [ -f "$BUFFER_PATH" ]; then
    echo "Buffer file found: $BUFFER_PATH"
    LOAD_BUFFER="True"
else
    echo "No buffer file found, starting with empty buffer"
    LOAD_BUFFER="False"
fi

# Submit SLURM job
cat > /tmp/resume_job_$$.sh <<EOF
#!/bin/bash
#SBATCH --job-name=resume_${ALG}_${TASK}
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:${GPU_TYPE,,}:1
#SBATCH --partition=gpu-${GPU_TYPE,,}
#SBATCH --qos=inferno
#SBATCH --output=logs/slurm/resume_${ALG}_${TASK}_seed${SEED}_%j.out
#SBATCH --error=logs/slurm/resume_${ALG}_${TASK}_seed${SEED}_%j.err

echo "======================================="
echo "SLURM Job Information"
echo "======================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Node: \$SLURMD_NODENAME"
echo "Start Time: \$(date)"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "======================================="

# Load environment
source ~/.bashrc
conda activate pwm || source activate pwm || echo "Warning: Could not activate conda env"

# Environment info
echo ""
echo "======================================="
echo "Environment Information"
echo "======================================="
echo "Python: \$(which python)"
echo "Python version: \$(python --version)"
echo "PyTorch version: \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: \$(python -c 'import torch; print(torch.cuda.device_count())')"
nvidia-smi
echo "======================================="

cd \$SLURM_SUBMIT_DIR

# Run training with resume
python scripts/train_dflex.py \\
    env=$TASK \\
    alg=$ALG \\
    general.seed=$SEED \\
    general.checkpoint=$CHECKPOINT_PATH \\
    general.checkpoint_with_buffer=$LOAD_BUFFER \\
    general.resume_training=True \\
    general.run_wandb=True \\
    wandb.project=flow-pwm-comparison \\
    wandb.group=${ALG}-${TASK}-resumed \\
    $EXTRA_ARGS

echo ""
echo "======================================="
echo "Job completed!"
echo "End Time: \$(date)"
echo "======================================="
EOF

# Submit the job
JOB_ID=$(sbatch /tmp/resume_job_$$.sh | awk '{print $NF}')
rm /tmp/resume_job_$$.sh

echo ""
echo "Job submitted successfully!"
echo "Job ID: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm/resume_${ALG}_${TASK}_seed${SEED}_${JOB_ID}.out"
echo ""
echo "Cancel with:"
echo "  scancel $JOB_ID"
