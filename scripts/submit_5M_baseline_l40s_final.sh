#!/bin/bash
#SBATCH -J pwm_5M_baseline_l40s
#SBATCH -A gts-agarg35
#SBATCH --gres=gpu:L40S:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=128GB
#SBATCH -t 8:00:00
#SBATCH -o /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/logs/train_5M_baseline_l40s_%j.out
#SBATCH -e /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/logs/train_5M_baseline_l40s_%j.err

# Training 5M Baseline on L40s (Reproduce PWM Paper Results)
# Expected: R ~ 1200 (based on Nov 8 successful run)
# Config: Linear LR schedule, batch_size=1024, task_dim=0

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

# Activate conda environment
source ~/.bashrc
conda activate pwm

# Set environment variables
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

# Train with baseline config (replicates PWM paper setup)
python scripts/train_dflex.py \
    general.device=cuda:0 \
    general.seed=42 \
    alg=pwm_5M_baseline_final \
    env.env.num_envs=256 \
    env.env.episode_length=1000 \
    horizon=4 \
    wandb.group="5M_baseline_l40s" \
    wandb.name="baseline_final_seed42" \
    wandb.project="flow-mbpo-pwm"

echo ""
echo "========================================="
echo "Training completed at $(date)"
echo "Expected: R ~ 1200 (PWM paper baseline)"
echo "========================================="
