#!/bin/bash
#SBATCH --job-name=anymal_flow_smoke
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/anymal/smoke_flow_%j.out
#SBATCH --error=logs/slurm/anymal/smoke_flow_%j.err

# Smoke test for Anymal Flow WM training (short epochs)
echo "==========================================="
echo "Anymal Flow WM Smoke Test"
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

# Run short training (100 epochs) to validate Flow WM works
cd scripts
python train_dflex.py \
    env=dflex_anymal \
    alg=pwm_5M_flow_v2_substeps4 \
    general.seed=42 \
    general.run_wandb=true \
    alg.max_epochs=100 \
    ++wandb.project=flow-mbpo-single \
    ++wandb.name=smoke_anymal_flowWM_K4_s42 \
    ++wandb.notes=SmokeTesFlowWMK4Heun100epochs

echo "==========================================="
echo "Smoke test completed at $(date)"
echo "==========================================="
