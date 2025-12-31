#!/bin/bash
# Full training script for Anymal Flow WM + MLP Policy
# Usage: ./train_anymal_flowWM.sh [SEED] [SUBSTEPS]

SEED=${1:-42}
SUBSTEPS=${2:-4}  # Default K=4 (Heun)

#SBATCH --job-name=anymal_flowWM_K${SUBSTEPS}_s${SEED}
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=384GB
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/anymal/flowWM_K${SUBSTEPS}_s${SEED}_%j.out
#SBATCH --error=logs/slurm/anymal/flowWM_K${SUBSTEPS}_s${SEED}_%j.err

echo "==========================================="
echo "Anymal Flow WM Training - Seed $SEED, K=$SUBSTEPS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "==========================================="

# Select config based on substeps
if [ "$SUBSTEPS" -eq 2 ]; then
    ALG_CONFIG="pwm_5M_flow_v1_substeps2"
elif [ "$SUBSTEPS" -eq 8 ]; then
    ALG_CONFIG="pwm_5M_flow_v3_substeps8_euler"
else
    ALG_CONFIG="pwm_5M_flow_v2_substeps4"
fi

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
    alg=$ALG_CONFIG \
    general.seed=$SEED \
    general.run_wandb=true \
    ++wandb.project="flow-mbpo-single" \
    ++wandb.name="anymal_FlowWM_K${SUBSTEPS}_MLPpol_s${SEED}" \
    ++wandb.notes="Anymal Flow WM (K=${SUBSTEPS}) + MLP Policy, 5M params, seed ${SEED}"

echo "==========================================="
echo "Training completed at $(date)"
echo "==========================================="
