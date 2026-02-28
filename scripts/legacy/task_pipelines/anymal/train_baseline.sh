#!/bin/bash
# Full training script for Anymal baseline (MLP WM + MLP Policy)
# Usage: ./train_anymal_baseline.sh [SEED]

SEED=${1:-42}

#SBATCH --job-name=anymal_base_s${SEED}
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=384GB
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/anymal/baseline_s${SEED}_%j.out
#SBATCH --error=logs/slurm/anymal/baseline_s${SEED}_%j.err

echo "==========================================="
echo "Anymal Baseline Training - Seed $SEED"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "==========================================="

# Setup
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
source ~/.bashrc
conda activate pwm
export PYTHONPATH=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/src

# Create log directory
mkdir -p logs/slurm/anymal

cd scripts
python train_dflex.py \
    env=dflex_anymal \
    alg=pwm_5M_baseline_final \
    general.seed=$SEED \
    general.run_wandb=true \
    ++wandb.project="flow-mbpo-single" \
    ++wandb.name="anymal_MLPWM_MLPpol_s${SEED}" \
    ++wandb.notes="Anymal baseline: MLP WM + MLP Policy, 5M params, seed ${SEED}"

echo "==========================================="
echo "Training completed at $(date)"
echo "==========================================="
