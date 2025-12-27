#!/bin/bash
#SBATCH -J pwm_retrain_48M_l40s
#SBATCH -A gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=embers
#SBATCH --output=logs/retrain_48M_l40s_%j.out
#SBATCH --error=logs/retrain_48M_l40s_%j.err

# 48M Baseline model with corrected configuration (L40s GPU)
# Expected training time: ~14-16 hours on L40s (vs 10-12 hours on H200)
# Target: 1200-1300 reward (if proper config works)

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
    alg=pwm_48M_improved \
    env=dflex_ant \
    general.device=cuda:0 \
    general.seed=42 \
    alg.save_interval=5000 \
    horizon=8

echo "Training completed at $(date)"
