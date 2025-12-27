#!/bin/bash
#SBATCH --job-name=pwm_flow_single
#SBATCH --account=gts-agarg35-ideasci23_dgx    # default; submit_job.sh may override
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1        # default GPU request; submit_job.sh may override
#SBATCH --mem=256GB              # Maximum RAM for flow-matching with large buffers
#SBATCH --time=48:00:00          # 最長48小時
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

################################################################################
# PACE Phoenix Single GPU Training Script for PWM Flow-Matching
################################################################################

echo "=============================================="
echo "SLURM Job Information"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Requested GPU Type(s): ${GPU_TYPE:-H200}"
echo "GPU Constraint: ${GPU_CONSTRAINT:-<none>}"
echo "Billing Account: ${GPU_ACCOUNT:-${SLURM_JOB_ACCOUNT:-unknown}}"
echo "Partition: ${GPU_PARTITION:-${SLURM_JOB_PARTITION:-unknown}}"
echo "QoS: ${GPU_QOS:-${SLURM_JOB_QOS:-unknown}}"
echo "Working Directory: $(pwd)"
echo "=============================================="

# Load required modules
module load anaconda3
module load cuda/12.1    # 根據需要調整CUDA版本

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate pwm       # 替換為你的環境名稱
else
    echo "ERROR: conda command not found. Make sure Anaconda is installed and available."
    exit 1
fi

# Print environment info
echo ""
echo "=============================================="
echo "Environment Information"
echo "=============================================="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi
echo "=============================================="
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

# Check if WandB is logged in
USE_WANDB=${USE_WANDB:-true}  # Set to 'false' to disable WandB
if [ "$USE_WANDB" = "true" ]; then
    if python -c "import wandb; wandb.login()" 2>&1 | grep -q "logged in"; then
        echo "WandB: Already logged in ✓"
        export WANDB_MODE=online
        WANDB_ENABLED=True
    else
        echo "================================================"
        echo "WARNING: WandB not logged in!"
        echo "================================================"
        echo "To enable WandB logging, run this on login node:"
        echo "  conda activate pwm"
        echo "  wandb login YOUR_API_KEY"
        echo ""
        echo "Get your API key from: https://wandb.ai/authorize"
        echo "================================================"
        echo "Continuing without WandB logging..."
        echo ""
        export WANDB_MODE=disabled
        WANDB_ENABLED=False
    fi
else
    echo "WandB: Disabled by user"
    export WANDB_MODE=disabled
    WANDB_ENABLED=False
fi

# Create necessary directories
mkdir -p logs/slurm
mkdir -p logs/baseline
mkdir -p logs/flow

# Configuration
TASK=${TASK:-dflex_ant}           # Can override with: sbatch --export=TASK=dflex_humanoid ...
ALGORITHM=${ALGORITHM:-pwm_48M}   # pwm_48M or pwm_48M_flow
SEED=${SEED:-42}
RUN_NAME="${TASK}_${ALGORITHM}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

# WandB configuration (only if enabled)
if [ "$WANDB_ENABLED" = "True" ]; then
    export WANDB_PROJECT=${WANDB_PROJECT:-"flow-pwm-comparison"}
    # Don't set WANDB_ENTITY - let WandB use default personal account
fi

echo "=============================================="
echo "Experiment Configuration"
echo "=============================================="
echo "Task: $TASK"
echo "Algorithm: $ALGORITHM"
echo "Seed: $SEED"
echo "Run Name: $RUN_NAME"
echo "WandB Enabled: $WANDB_ENABLED"
if [ "$WANDB_ENABLED" = "True" ]; then
    echo "WandB Project: $WANDB_PROJECT"
fi
echo "=============================================="
echo ""

# Change to project directory
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

# Run training
echo "Starting training..."
echo "=============================================="

if [ "$WANDB_ENABLED" = "True" ]; then
    # With WandB - don't pass empty entity
    python scripts/train_dflex.py \
        env=$TASK \
        alg=$ALGORITHM \
        general.seed=$SEED \
        general.run_wandb=True \
        general.logdir=logs/${ALGORITHM}_${TASK}_seed${SEED} \
        wandb.project=$WANDB_PROJECT \
        wandb.group=${ALGORITHM}-${TASK} \
        2>&1 | tee logs/slurm/training_${SLURM_JOB_ID}.log
else
    # Without WandB
    python scripts/train_dflex.py \
        env=$TASK \
        alg=$ALGORITHM \
        general.seed=$SEED \
        general.run_wandb=False \
        general.logdir=logs/${ALGORITHM}_${TASK}_seed${SEED} \
        2>&1 | tee logs/slurm/training_${SLURM_JOB_ID}.log
fi

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=============================================="

# Generate visualizations if training succeeded
if [ $EXIT_CODE -eq 0 ]; then
    echo "Generating visualizations..."
    # Hydra changes working directory, so use absolute path
    LOG_DIR="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/logs/${ALGORITHM}_${TASK}_seed${SEED}"
    if [ -d "$LOG_DIR" ]; then
        python scripts/generate_visualizations.py \
            --log-dir "$LOG_DIR" \
            2>&1 | tee logs/slurm/viz_${SLURM_JOB_ID}.log
    else
        echo "Warning: Log directory not found at $LOG_DIR"
        echo "Skipping visualization generation."
    fi
fi

echo ""
echo "=============================================="
echo "Job completed!"
echo "Results saved to: logs/${ALGORITHM}_${TASK}_seed${SEED}"
echo "=============================================="

exit $EXIT_CODE
