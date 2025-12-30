#!/bin/bash
#SBATCH --job-name=test_flowpolicy
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/anymal/test_flowpolicy_%j.out
#SBATCH --error=logs/slurm/anymal/test_flowpolicy_%j.err

# Smoke test for Flow Policy (MLP WM + Flow ODE Policy)
echo "==========================================="
echo "Testing Flow Policy (MLP WM + Flow ODE Policy)"
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

# Run short training (100 epochs) to validate Flow Policy works
cd scripts
python train_dflex.py \
    env=dflex_anymal \
    alg=pwm_5M_flowpolicy \
    general.seed=42 \
    general.run_wandb=true \
    alg.max_epochs=100 \
    ++wandb.project=flow-mbpo-single \
    ++wandb.name=TEST_FlowPolicy_MLP_WM_s42 \
    ++wandb.notes=SmokeTestFlowPolicyMLPWM100epochs

echo "==========================================="
echo "Test completed at $(date)"
echo "==========================================="
