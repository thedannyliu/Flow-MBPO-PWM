#!/bin/bash
#SBATCH --job-name=pwm_flow_multi
#SBATCH --account=gts-agarg35-ideasci23_dgx     # default; submit_job.sh may override
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4個任務 (每GPU一個)
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:4        # default request; submit_job.sh may override
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

################################################################################
# PACE Phoenix Multi-GPU Training Script for PWM Flow-Matching
# 
# Strategy: Run multiple experiments in parallel (different seeds or tasks)
# Each GPU handles one independent experiment
################################################################################

echo "=============================================="
echo "SLURM Job Information"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Requested GPU Type(s): ${GPU_TYPE:-H200}"
echo "GPU Constraint: ${GPU_CONSTRAINT:-<none>}"
echo "Billing Account: ${GPU_ACCOUNT:-${SLURM_JOB_ACCOUNT:-unknown}}"
echo "Partition: ${GPU_PARTITION:-${SLURM_JOB_PARTITION:-unknown}}"
echo "QoS: ${GPU_QOS:-${SLURM_JOB_QOS:-unknown}}"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "=============================================="

# Load required modules
module load anaconda3
module load cuda/12.1

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate pwm
else
    echo "ERROR: conda command not found. Make sure Anaconda is installed and available."
    exit 1
fi

# Print environment info
echo ""
echo "=============================================="
echo "Environment Information"
echo "=============================================="
nvidia-smi
echo "=============================================="
echo ""

# Create necessary directories
mkdir -p logs/slurm
mkdir -p logs/baseline
mkdir -p logs/flow

# Configuration
TASK=${TASK:-dflex_ant}
ALGORITHM=${ALGORITHM:-pwm_48M}
BASE_SEED=${SEED:-42}

# Check if WandB is logged in
USE_WANDB=${USE_WANDB:-true}
if [ "$USE_WANDB" = "true" ]; then
    if python -c "import wandb; wandb.login()" 2>&1 | grep -q "logged in"; then
        echo "WandB: Already logged in ✓"
        export WANDB_MODE=online
        WANDB_ENABLED=True
        export WANDB_PROJECT=${WANDB_PROJECT:-"flow-pwm-comparison"}
    else
        echo "WARNING: WandB not logged in! Continuing without WandB..."
        export WANDB_MODE=disabled
        WANDB_ENABLED=False
    fi
else
    echo "WandB: Disabled by user"
    export WANDB_MODE=disabled
    WANDB_ENABLED=False
fi

# Change to project directory
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

# ==============================================================================
# Multi-GPU Strategy: Run 4 experiments in parallel
# ==============================================================================
# Option 1: Different seeds for same task (for statistical significance)
# Option 2: Different tasks with same seed (for multi-task evaluation)
# Option 3: Mix of baseline and flow on different GPUs (for direct comparison)
# ==============================================================================

# Choose strategy here:
STRATEGY=${STRATEGY:-"multi_seed"}  # "multi_seed", "multi_task", or "baseline_vs_flow"

case $STRATEGY in
    "multi_seed")
        echo "Strategy: Running same task with 4 different seeds"
        SEEDS=(42 123 456 789)
        TASKS=($TASK $TASK $TASK $TASK)
        ALGOS=($ALGORITHM $ALGORITHM $ALGORITHM $ALGORITHM)
        ;;
    
    "multi_task")
        echo "Strategy: Running 4 different tasks with same seed"
        SEEDS=($BASE_SEED $BASE_SEED $BASE_SEED $BASE_SEED)
        TASKS=(dflex_ant dflex_humanoid dflex_hopper dflex_anymal)
        ALGOS=($ALGORITHM $ALGORITHM $ALGORITHM $ALGORITHM)
        ;;
    
    "baseline_vs_flow")
        echo "Strategy: Comparing baseline vs flow (2 seeds each)"
        SEEDS=(42 123 42 123)
        TASKS=($TASK $TASK $TASK $TASK)
        ALGOS=(pwm_48M pwm_48M pwm_48M_flow pwm_48M_flow)
        ;;
    
    *)
        echo "Unknown strategy: $STRATEGY"
        exit 1
        ;;
esac

echo "=============================================="
echo "Launching 4 parallel experiments:"
for i in {0..3}; do
    echo "  GPU $i: ${ALGOS[$i]} on ${TASKS[$i]} (seed=${SEEDS[$i]})"
done
echo "=============================================="
echo ""

# Function to run single experiment
run_experiment() {
    local GPU_ID=$1
    local TASK=$2
    local ALGO=$3
    local SEED=$4
    
    local RUN_NAME="${TASK}_${ALGO}_seed${SEED}_gpu${GPU_ID}"
    local LOG_DIR="logs/${ALGO}_${TASK}_seed${SEED}"
    
    echo "[GPU $GPU_ID] Starting: $RUN_NAME"
    
    # Set GPU visibility for this process
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Run training
    if [ "$WANDB_ENABLED" = "True" ]; then
        python scripts/train_dflex.py \
            env=$TASK \
            alg=$ALGO \
            general.seed=$SEED \
            general.run_wandb=True \
            general.logdir=$LOG_DIR \
            wandb.project=$WANDB_PROJECT \
            wandb.group=${ALGO}-${TASK} \
            2>&1 | tee logs/slurm/training_${SLURM_JOB_ID}_gpu${GPU_ID}.log
    else
        python scripts/train_dflex.py \
            env=$TASK \
            alg=$ALGO \
            general.seed=$SEED \
            general.run_wandb=False \
            general.logdir=$LOG_DIR \
            2>&1 | tee logs/slurm/training_${SLURM_JOB_ID}_gpu${GPU_ID}.log
    fi
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[GPU $GPU_ID] Training completed successfully"
        
        # Generate visualizations
        python scripts/generate_visualizations.py \
            --log-dir $LOG_DIR \
            2>&1 | tee logs/slurm/viz_${SLURM_JOB_ID}_gpu${GPU_ID}.log
    else
        echo "[GPU $GPU_ID] Training failed with exit code: $EXIT_CODE"
    fi
    
    return $EXIT_CODE
}

# Export function for parallel execution
export -f run_experiment

# Launch all experiments in parallel (one per GPU)
PIDS=()
for GPU_ID in {0..3}; do
    run_experiment $GPU_ID "${TASKS[$GPU_ID]}" "${ALGOS[$GPU_ID]}" "${SEEDS[$GPU_ID]}" &
    PIDS+=($!)
    sleep 5  # Stagger starts slightly to avoid race conditions
done

# Wait for all experiments to complete
echo ""
echo "=============================================="
echo "Waiting for all experiments to complete..."
echo "=============================================="
echo ""

EXIT_CODES=()
for i in {0..3}; do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    EXIT_CODES+=($EXIT_CODE)
    echo "GPU $i completed with exit code: $EXIT_CODE"
done

# Check if all succeeded
ALL_SUCCESS=true
for code in "${EXIT_CODES[@]}"; do
    if [ $code -ne 0 ]; then
        ALL_SUCCESS=false
        break
    fi
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "End Time: $(date)"
echo "=============================================="

if [ "$ALL_SUCCESS" = true ]; then
    echo "✓ All experiments succeeded"
    
    # Generate comparison plots if same task with different seeds
    if [ "$STRATEGY" = "multi_seed" ]; then
        echo ""
        echo "Generating comparison plots across seeds..."
        python scripts/compare_runs.py \
            --task $TASK \
            --algorithm $ALGORITHM \
            --seeds "${SEEDS[@]}" \
            2>&1 | tee logs/slurm/comparison_${SLURM_JOB_ID}.log
    fi
    
    exit 0
else
    echo "✗ Some experiments failed"
    exit 1
fi
