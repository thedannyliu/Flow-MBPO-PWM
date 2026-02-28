#!/bin/bash
#SBATCH --job-name=anymal_baseline_smoke
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/anymal/smoke_baseline_%j.out
#SBATCH --error=logs/slurm/anymal/smoke_baseline_%j.err

# Smoke test for Anymal baseline training (short epochs)
echo "==========================================="
echo "Anymal Baseline Smoke Test"
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

# Run short training (100 epochs) to validate everything works
cd scripts
python train_dflex.py \
    env=dflex_anymal \
    alg=pwm_5M_baseline_final \
    general.seed=42 \
    general.run_wandb=true \
    alg.max_epochs=100 \
    "++wandb.project=flow-mbpo-single" \
    "++wandb.name=smoke_anymal_baseline_s42" \
    "++wandb.notes=Smoke test Anymal baseline 100 epochs seed 42"

echo "==========================================="
echo "Smoke test completed at $(date)"
echo "==========================================="
