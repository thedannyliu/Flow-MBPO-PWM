#!/bin/bash
#SBATCH --job-name=test_fullflow
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/anymal/test_fullflow_%j.out
#SBATCH --error=logs/slurm/anymal/test_fullflow_%j.err

# Smoke test for Full Flow (Flow WM + Flow ODE Policy)
echo "==========================================="
echo "Testing Full Flow (Flow WM + Flow ODE Policy)"
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

# Run short training (100 epochs) to validate Full Flow works
cd scripts
python train_dflex.py \
    env=dflex_anymal \
    alg=pwm_5M_fullflow \
    general.seed=42 \
    general.run_wandb=true \
    alg.max_epochs=100 \
    ++wandb.project=flow-mbpo-single \
    ++wandb.name=TEST_FullFlow_FlowWM_FlowPol_s42 \
    ++wandb.notes=SmokeTestFullFlowFlowWMFlowPolicy100epochs

echo "==========================================="
echo "Test completed at $(date)"
echo "==========================================="
