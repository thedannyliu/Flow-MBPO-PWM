#!/bin/bash
#SBATCH --job-name=epoch_150k
#SBATCH --output=logs/slurm/mt30/epoch_150k_%A_%a.out
#SBATCH --error=logs/slurm/mt30/epoch_150k_%A_%a.err
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=16:00:00
#SBATCH --array=0-17

# 150k Epoch experiments (Baseline 0-8, FullFlow 9-17)
# Using H200 with 16h time limit, reduced batch size to avoid OOM

# === ENVIRONMENT === (same as working pretrain script)
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo
export PATH=$CONDA_PREFIX/bin:$PATH

cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM

export PYTHONPATH=/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/src:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/scripts:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/external/tdmpc2:$PYTHONPATH
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1

TASKS=("reacher-easy" "reacher-easy" "reacher-easy" "walker-stand" "walker-stand" "walker-stand" "cheetah-run" "cheetah-run" "cheetah-run")
SEEDS=(42 123 456 42 123 456 42 123 456)

IDX=$((SLURM_ARRAY_TASK_ID % 9))
TASK=${TASKS[$IDX]}
SEED=${SEEDS[$IDX]}

if [ $SLURM_ARRAY_TASK_ID -lt 9 ]; then
    CONFIG="pwm_48M_mt_baseline"
    NAME="baseline_150k_${TASK}_s${SEED}"
else
    CONFIG="pwm_48M_mt_fullflow"
    NAME="flow_150k_${TASK}_s${SEED}"
fi

echo "=== 150k Epoch Experiment ==="
echo "Config: ${CONFIG}"
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "=============================="

# Reduced batch size to avoid OOM
python scripts/train_multitask.py \
    alg=${CONFIG} \
    task=dmcontrol-${TASK} \
    seed=${SEED} \
    general.epochs=150000 \
    alg.wm_batch_size=128 \
    general.run_wandb=True \
    wandb.project=MT30-Detailed \
    wandb.name="${NAME}"

echo "=== Training Complete ==="
