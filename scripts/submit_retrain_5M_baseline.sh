#!/bin/bash
#SBATCH -J pwm_retrain_5M_baseline
#SBATCH -A gts-agarg35
#SBATCH --gres=gpu:H200:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=256GB
#SBATCH -t 6:00:00
#SBATCH -o logs/retrain_5M_baseline_%j.out
#SBATCH -e logs/retrain_5M_baseline_%j.err

# 5M Baseline model with improved LR schedule
# Expected training time: ~4-5 hours on H200
# Target: Stable 1200 reward throughout training

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

# Activate conda environment
source ~/.bashrc
conda activate pwm

# Train with improved configuration
python scripts/train_dflex.py \
    --config-name=config \
    alg=pwm_5M_improved \
    env=dflex_ant \
    general.device=cuda:0 \
    general.seed=42 \
    alg.save_interval=2500 \
    horizon=8

echo "Training completed at $(date)"
