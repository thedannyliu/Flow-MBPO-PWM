#!/bin/bash
#SBATCH -J pwm_retrain_5M_flow_l40s
#SBATCH -A gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=embers
#SBATCH --output=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/logs/retrain_5M_flow_l40s_%j.out
#SBATCH --error=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/logs/retrain_5M_flow_l40s_%j.err

# 5M Flow-matching model with improved configuration (L40s GPU)
# L40s: 48GB GDDR6, 350W TDP (slower than H200 but more available)
# Expected training time: ~10-12 hours on L40s (vs 6-8 hours on H200)
# Target reward: 1100-1200 (vs current 900)

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
    alg=pwm_5M_flow_improved \
    env=dflex_ant \
    general.device=cuda:0 \
    general.seed=42 \
    alg.save_interval=2500 \
    horizon=8

echo "Training completed at $(date)"
