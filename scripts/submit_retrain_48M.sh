#!/bin/bash
#SBATCH -J pwm_retrain_48M
#SBATCH -A gts-agarg35
#SBATCH --gres=gpu:H200:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64GB
#SBATCH -t 12:00:00
#SBATCH -o logs/retrain_48M_%j.out
#SBATCH -e logs/retrain_48M_%j.err

# 48M Baseline model with corrected configuration
# Expected training time: ~10-12 hours on H200
# Target: 1200-1300 reward (if proper config works)

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

# Activate conda environment
source ~/.bashrc
conda activate pwm

# Train with improved configuration
python scripts/train_dflex.py \
    --config-name=config \
    alg=pwm_48M_improved \
    env=dflex_ant \
    general.device=cuda:0 \
    general.seed=42 \
    alg.save_interval=5000 \
    horizon=8

echo "Training completed at $(date)"
