#!/bin/bash
#SBATCH --job-name=resume_flow_100k
#SBATCH --output=logs/slurm/mt30/resume_flow_100k_%A_%a.out
#SBATCH --error=logs/slurm/mt30/resume_flow_100k_%A_%a.err
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=16:00:00
#SBATCH --array=0,2-8

# Resume Flow 100k experiments from timeout checkpoints
# Array excludes index 1 which failed early
# Using H200 with 24h time limit

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

# Find checkpoint in various possible locations
CHECKPOINT=""
for dir in outputs/epoch_sweep/flow_100k/*/; do
    potential="${dir}${SLURM_ARRAY_TASK_ID}_s${SEED}/logs/model_last.pt"
    if [ -f "$potential" ]; then
        CHECKPOINT="$potential"
        break
    fi
done

echo "=== Resume Flow 100k ==="
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "Checkpoint: ${CHECKPOINT}"
echo "========================="

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "WARNING: Checkpoint not found, starting from scratch"
    python scripts/train_multitask.py \
        alg=pwm_48M_mt_fullflow \
        task=dmcontrol-${TASK} \
        seed=${SEED} \
        general.epochs=100000 \
        general.run_wandb=True \
        wandb.project=MT30-Detailed \
        wandb.name="resume_flow_100k_${TASK}_s${SEED}"
else
    python scripts/train_multitask.py \
        alg=pwm_48M_mt_fullflow \
        task=dmcontrol-${TASK} \
        seed=${SEED} \
        general.epochs=100000 \
        general.resume_from=${CHECKPOINT} \
        general.run_wandb=True \
        wandb.project=MT30-Detailed \
        wandb.name="resume_flow_100k_${TASK}_s${SEED}"
fi
