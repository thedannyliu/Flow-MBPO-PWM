#!/bin/bash
#SBATCH --job-name=mt30_baseline_scratch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --partition=ice-gpu
#SBATCH --account=coc
#SBATCH --array=0-8
#SBATCH --output=logs/slurm/mt30/baseline_%A_%a.out
#SBATCH --error=logs/slurm/mt30/baseline_%A_%a.err

# Arrays for tasks and seeds
# 3 Tasks x 3 Seeds = 9 Jobs
TASKS=("reacher-easy" "walker-stand" "cheetah-run")
SEEDS=(42 123 456)

TASK_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))
TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# === ENVIRONMENT ===
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo
export PATH=$CONDA_PREFIX/bin:$PATH

cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM

# Set PYTHONPATH with absolute paths
export PYTHONPATH=/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/src:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/scripts:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/external/tdmpc2:$PYTHONPATH
export MUJOCO_GL=egl
export LAZY_LEGACY_OP=0
export HYDRA_FULL_ERROR=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs/slurm/mt30
mkdir -p outputs/mt30_baseline/${TASK}/seed${SEED}

echo "=== MT30 Baseline (From Scratch) ==="
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Config: pwm_48M_mt_baseline"
echo "================================"

# Baseline from scratch: finetune_wm=True ensures WM is trained (not frozen)
python -u scripts/train_multitask.py -cn config_mt30 \
  alg=pwm_48M_mt_baseline \
  task=${TASK} \
  general.seed=${SEED} \
  general.data_dir=/home/hice1/eliu354/scratch/Data/tdmpc2/mt30 \
  general.epochs=15000 \
  general.eval_freq=200 \
  general.eval_runs=10 \
  general.finetune_wm=True \
  general.run_wandb=True \
  ++wandb.project=MT30-Detailed \
  ++wandb.group=mt30_baseline_scratch \
  ++wandb.name=baseline_${TASK}_s${SEED} \
  hydra.run.dir="outputs/mt30_baseline/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}_s${SEED}" \
  2>&1
