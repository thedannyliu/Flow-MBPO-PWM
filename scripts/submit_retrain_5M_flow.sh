#!/bin/bash
#SBATCH -J pwm_retrain_5M_flow
#SBATCH -A gts-agarg35
#SBATCH --gres=gpu:H200:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=256GB
#SBATCH -t 8:00:00
#SBATCH -o logs/retrain_5M_flow_%j.out
#SBATCH -e logs/retrain_5M_flow_%j.err

# 5M Flow-matching model with improved configuration
# Expected training time: ~6-8 hours on H200
# Target reward: 1100-1200 (vs current 900)

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

# Activate conda environment
source ~/.bashrc
conda activate pwm

# Train with improved configuration
python scripts/train_dflex.py \
    --config-name=config \
    alg=pwm_5M_flow_improved \
    env=dflex_ant \
    general.device=cuda:0 \
    general.seed=42 \
    alg.save_interval=2500 \
    horizon=8

echo "Training completed at $(date)"
