#!/bin/bash
# Smoke test for Ant Baseline - verify config works before full training
# Runs 5 epochs only to validate setup

#SBATCH --job-name=ant_smoke_baseline
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/ant/smoke_baseline_%j.out
#SBATCH --error=logs/slurm/ant/smoke_baseline_%j.err

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
source ~/.bashrc
conda activate pwm
export PYTHONPATH=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/src

cd scripts
python train_dflex.py \
    env=dflex_ant \
    alg=pwm_5M_baseline_final \
    general.seed=42 \
    alg.max_epochs=5 \
    general.run_wandb=false

echo "Smoke test completed!"
