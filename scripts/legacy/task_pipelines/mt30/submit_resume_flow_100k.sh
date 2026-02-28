#!/bin/bash
#SBATCH --job-name=resume_flow_100k
#SBATCH --output=logs/slurm/mt30/resume_flow_100k_%A_%a.out
#SBATCH --error=logs/slurm/mt30/resume_flow_100k_%A_%a.err
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=16:00:00
#SBATCH --array=0-8

# Flow 100k experiments (fresh start since resume failed)
# Using H200 with 16h time limit

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

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Flow 100k Epoch ==="
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "========================"

python scripts/train_multitask.py \
    alg=pwm_48M_mt_fullflow \
    task=${TASK} \
    seed=${SEED} \
    general.epochs=100000 \
    general.data_dir="/home/hice1/eliu354/scratch/Data/tdmpc2/mt30" \
    general.run_wandb=True \
    +wandb.project=MT30-Detailed \
    +wandb.name="resume_flow_100k_${TASK}_s${SEED}"
