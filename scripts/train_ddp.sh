#!/bin/bash
#SBATCH --job-name=pwm_ddp_train
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=inferno
#SBATCH --gres=gpu:l40s:4
#SBATCH --exclude=atl1-1-03-004-31-0,atl1-1-03-007-29-0,atl1-1-03-007-31-0  # Exclude nodes with potential ECC issues
#SBATCH --output=logs/slurm/ddp_%j.out
#SBATCH --error=logs/slurm/ddp_%j.err

#
# PyTorch Distributed Data Parallel (DDP) Training Script
# 
# This script uses multiple L40s GPUs to accelerate training via data parallelism.
# Training batches are split across GPUs and gradients are synchronized.
#
# Usage:
#   sbatch scripts/train_ddp.sh
# 
# Environment Variables (set before sbatch):
#   TASK           - Task name (e.g., dflex_ant)
#   ALGORITHM      - Algorithm config (e.g., pwm_5M_flow)
#   SEED           - Random seed
#   NUM_GPUS       - Number of GPUs to use (default: 4)
#   BATCH_SCALE    - Batch size scale factor (default: 4, i.e., 4x larger batch)
#

# Default values
TASK=${TASK:-dflex_ant}
ALGORITHM=${ALGORITHM:-pwm_5M_flow}
SEED=${SEED:-42}
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SCALE=${BATCH_SCALE:-4}

echo "=========================================="
echo "PyTorch DDP Training Configuration"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Task: $TASK"
echo "Algorithm: $ALGORITHM"
echo "Seed: $SEED"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch Scale: ${BATCH_SCALE}x"
echo "Start Time: $(date)"
echo "=========================================="

# Load environment
source ~/.bashrc
conda activate pwm || source activate pwm || echo "Warning: Could not activate conda env"

# Verify GPUs
echo ""
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo "=========================================="

# Set PyTorch DDP environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$NUM_GPUS

# Calculate scaled batch sizes for fair comparison with H200
# H200 single GPU: num_envs=256, wm_batch_size=1024, critic_batches=4
# L40s 4x DDP: Keep total workload same, split across GPUs
SCALED_NUM_ENVS=$((256 / NUM_GPUS))  # 256/4 = 64 per GPU
SCALED_WM_BATCH=$((1024 / NUM_GPUS))  # 1024/4 = 256 per GPU
SCALED_CRITIC_BATCH=4  # Keep same

echo ""
echo "=========================================="
echo "DDP Hyperparameters (Fair Comparison)"
echo "=========================================="
echo "H200 Single GPU:     num_envs=256, wm_batch=1024"
echo "L40s 4x DDP (total): num_envs=256, wm_batch=1024"
echo "L40s Per GPU:        num_envs=$SCALED_NUM_ENVS, wm_batch=$SCALED_WM_BATCH"
echo "critic_batches:      $SCALED_CRITIC_BATCH"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

# Launch training with torchrun (PyTorch's distributed launcher)
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    scripts/train_dflex_ddp.py \
    env=$TASK \
    alg=$ALGORITHM \
    general.seed=$SEED \
    general.run_wandb=True \
    general.logdir=logs/${ALGORITHM}_${TASK}_ddp_seed${SEED} \
    wandb.project=flow-pwm-comparison \
    wandb.group=${ALGORITHM}-${TASK}-ddp \
    env.config.num_envs=$SCALED_NUM_ENVS \
    alg.wm_batch_size=$SCALED_WM_BATCH \
    alg.critic_batches=$SCALED_CRITIC_BATCH

echo ""
echo "=========================================="
echo "Training Completed"
echo "Exit Code: $?"
echo "End Time: $(date)"
echo "=========================================="
