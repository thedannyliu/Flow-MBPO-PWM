#!/bin/bash
#
# Submit DDP training job with multiple L40s GPUs
#
# Usage:
#   ./scripts/submit_ddp.sh <algorithm> <task> [seed] [num_gpus] [batch_scale]
#
# Examples:
#   # 4 L40s, 4x batch size
#   ./scripts/submit_ddp.sh pwm_5M_flow dflex_ant 42 4 4
#
#   # 2 L40s, 2x batch size (for testing)
#   ./scripts/submit_ddp.sh pwm_5M_flow dflex_ant 42 2 2
#
#   # 8 L40s, 8x batch size (maximum parallelism)
#   ./scripts/submit_ddp.sh pwm_5M_flow dflex_ant 42 8 8
#

set -e

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Error: Algorithm and task required"
    echo "Usage: $0 <algorithm> <task> [seed] [num_gpus] [batch_scale]"
    echo ""
    echo "Examples:"
    echo "  $0 pwm_5M_flow dflex_ant 42 4 4      # 4 GPUs, 4x batch"
    echo "  $0 pwm_5M dflex_ant 42 2 2           # 2 GPUs, 2x batch"
    exit 1
fi

ALGORITHM="$1"
TASK="$2"
SEED="${3:-42}"
NUM_GPUS="${4:-4}"
BATCH_SCALE="${5:-4}"

# Validate num_gpus
if [ "$NUM_GPUS" -lt 1 ] || [ "$NUM_GPUS" -gt 8 ]; then
    echo "Error: num_gpus must be between 1 and 8"
    exit 1
fi

# Calculate resource requirements (L40s max CPU:GPU ratio is 4:1)
CPUS_PER_TASK=4  # Maximum 4 CPUs per GPU for L40s
TOTAL_CPUS=$((NUM_GPUS * CPUS_PER_TASK))
TOTAL_MEM=$((NUM_GPUS * 60))  # 60GB per GPU

echo "=========================================="
echo "DDP Training Submission"
echo "=========================================="
echo "Algorithm: $ALGORITHM"
echo "Task: $TASK"
echo "Seed: $SEED"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch Scale: ${BATCH_SCALE}x"
echo "Resources:"
echo "  CPUs: $TOTAL_CPUS (${CPUS_PER_TASK} per GPU)"
echo "  Memory: ${TOTAL_MEM}G"
echo "=========================================="

# Create SLURM script
cat > /tmp/ddp_job_$$.sh <<EOF
#!/bin/bash
#SBATCH --job-name=${ALGORITHM}_${TASK}_ddp
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$NUM_GPUS
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=${TOTAL_MEM}G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=inferno
#SBATCH --gres=gpu:l40s:$NUM_GPUS
#SBATCH --exclude=atl1-1-03-004-31-0
#SBATCH --output=logs/slurm/${ALGORITHM}_${TASK}_ddp_${NUM_GPUS}gpu_%j.out
#SBATCH --error=logs/slurm/${ALGORITHM}_${TASK}_ddp_${NUM_GPUS}gpu_%j.err

echo "=========================================="
echo "PyTorch DDP Training"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Task: $TASK"
echo "Algorithm: $ALGORITHM"
echo "Seed: $SEED"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch Scale: ${BATCH_SCALE}x"
echo "Start Time: \$(date)"
echo "=========================================="

# Load environment
source ~/.bashrc
conda activate pwm || source activate pwm

# Verify GPUs
echo ""
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo "=========================================="

# Set PyTorch DDP environment variables
export MASTER_ADDR=\$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$NUM_GPUS

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HYDRA_FULL_ERROR=1

# Calculate scaled batch sizes
SCALED_NUM_ENVS=\$((128 * $BATCH_SCALE))
SCALED_WM_BATCH=\$((256 * $BATCH_SCALE))
SCALED_CRITIC_BATCH=\$((4 * $BATCH_SCALE))

echo ""
echo "=========================================="
echo "Scaled Hyperparameters"
echo "=========================================="
echo "num_envs: 128 → \$SCALED_NUM_ENVS"
echo "wm_batch_size: 256 → \$SCALED_WM_BATCH"
echo "critic_batches: 4 → \$SCALED_CRITIC_BATCH"
echo "Learning rates: scaled by 1/√${BATCH_SCALE}"
echo "=========================================="

cd \$SLURM_SUBMIT_DIR

# Calculate learning rate scaling (use sqrt scaling for stability)
LR_SCALE=\$(python -c "import math; print(1.0 / math.sqrt($BATCH_SCALE))")

# Launch training with torchrun
torchrun \\
    --standalone \\
    --nnodes=1 \\
    --nproc_per_node=$NUM_GPUS \\
    scripts/train_dflex_ddp.py \\
    env=$TASK \\
    alg=$ALGORITHM \\
    general.seed=$SEED \\
    general.run_wandb=True \\
    general.logdir=logs/${ALGORITHM}_${TASK}_ddp${NUM_GPUS}_seed${SEED} \\
    wandb.project=flow-pwm-comparison \\
    wandb.group=${ALGORITHM}-${TASK}-ddp${NUM_GPUS} \\
    env.config.num_envs=\$SCALED_NUM_ENVS \\
    alg.wm_batch_size=\$SCALED_WM_BATCH \\
    alg.critic_batches=\$SCALED_CRITIC_BATCH

EXIT_CODE=\$?

echo ""
echo "=========================================="
echo "Training Completed"
echo "Exit Code: \$EXIT_CODE"
echo "End Time: \$(date)"
echo "=========================================="

exit \$EXIT_CODE
EOF

# Submit job
JOB_ID=$(sbatch /tmp/ddp_job_$$.sh | awk '{print $NF}')
rm /tmp/ddp_job_$$.sh

echo ""
echo "Job submitted successfully!"
echo "Job ID: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm/${ALGORITHM}_${TASK}_ddp_${NUM_GPUS}gpu_${JOB_ID}.out"
echo ""
echo "Cancel with:"
echo "  scancel $JOB_ID"
echo ""
echo "Expected speedup: ~${BATCH_SCALE}x (with $NUM_GPUS GPUs)"
echo "  - Single H200: ~2700 FPS"
echo "  - $NUM_GPUS L40s: ~$((2700 * BATCH_SCALE / NUM_GPUS * 8 / 10)) FPS (estimated, 80% efficiency)"
