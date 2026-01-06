#!/bin/bash
#SBATCH --job-name=resume_flow_50k
#SBATCH --output=logs/slurm/mt30/resume_flow_50k_%A_%a.out
#SBATCH --error=logs/slurm/mt30/resume_flow_50k_%A_%a.err
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=16:00:00
#SBATCH --array=0-8

# Resume Flow 50k experiments from timeout checkpoints
# Using H200 with 16h time limit

source ~/.bashrc
conda activate flow-mbpo

export PYTHONPATH="${PYTHONPATH}:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/src"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1

cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM

TASKS=("reacher-easy" "reacher-easy" "reacher-easy" "walker-stand" "walker-stand" "walker-stand" "cheetah-run" "cheetah-run" "cheetah-run")
SEEDS=(42 123 456 42 123 456 42 123 456)

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Checkpoint pattern: outputs/epoch_sweep/flow_50k/4012536/${SLURM_ARRAY_TASK_ID}_s${SEED}/logs/model_last.pt
CHECKPOINT_DIR="outputs/epoch_sweep/flow_50k/4012536/${SLURM_ARRAY_TASK_ID}_s${SEED}"
CHECKPOINT="${CHECKPOINT_DIR}/logs/model_last.pt"

echo "=== Resume Flow 50k ==="
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "Checkpoint: ${CHECKPOINT}"
echo "========================"

if [ ! -f "$CHECKPOINT" ]; then
    echo "WARNING: Checkpoint not found, starting from scratch"
    python scripts/train_multitask.py \
        alg=pwm_48M_mt_fullflow \
        task=dmcontrol-${TASK} \
        seed=${SEED} \
        general.epochs=50000 \
        general.run_wandb=True \
        wandb.project=MT30-Detailed \
        wandb.name="resume_flow_50k_${TASK}_s${SEED}"
else
    python scripts/train_multitask.py \
        alg=pwm_48M_mt_fullflow \
        task=dmcontrol-${TASK} \
        seed=${SEED} \
        general.epochs=50000 \
        general.resume_from=${CHECKPOINT} \
        general.run_wandb=True \
        wandb.project=MT30-Detailed \
        wandb.name="resume_flow_50k_${TASK}_s${SEED}"
fi
