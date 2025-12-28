#!/bin/bash
#SBATCH --job-name=test_wandb_naming
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/anymal/test_naming_%j.out
#SBATCH --error=logs/slurm/anymal/test_naming_%j.err

# Quick test to verify WandB naming fix
echo "==========================================="
echo "Testing WandB Naming Fix"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "==========================================="

# Setup
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
source ~/.bashrc
conda activate pwm
export PYTHONPATH=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/src

mkdir -p logs/slurm/anymal

echo "Testing baseline naming..."
cd scripts
python train_dflex.py \
    env=dflex_anymal \
    alg=pwm_5M_baseline_final \
    general.seed=42 \
    general.run_wandb=true \
    alg.max_epochs=50 \
    ++wandb.project=flow-mbpo-single \
    ++wandb.name=TEST_Anymal_Baseline_s42 \
    ++wandb.notes="Validation test for WandB naming fix - baseline s42"

echo "==========================================="
echo "Test completed at $(date)"
echo "==========================================="
