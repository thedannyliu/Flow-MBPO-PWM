#!/bin/bash
#SBATCH -J pwm_retrain_5M_baseline_l40s
#SBATCH -A gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=embers
#SBATCH --output=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/logs/retrain_5M_baseline_l40s_%j.out
#SBATCH --error=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/logs/retrain_5M_baseline_l40s_%j.err

# 5M Baseline model with improved LR schedule (L40s GPU)
# Expected training time: ~6-8 hours on L40s (vs 4-5 hours on H200)
# Target: Stable 1200 reward throughout training

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate pwm

# GPU info
echo "GPU Information:"
nvidia-smi
echo ""

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
