#!/bin/bash
#SBATCH --job-name=mt30_baseline_100k
#SBATCH --output=logs/slurm/mt30/baseline_100k_%A_%a.out
#SBATCH --error=logs/slurm/mt30/baseline_100k_%A_%a.err
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=450GB
#SBATCH --cpus-per-task=16
#SBATCH -t 16:00:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu
#SBATCH --array=0-8

# MT30 Baseline 100K: Aligned with Original PWM (baselines/original_pwm)
# Training from scratch with finetune_wm=True

# === CONFIGURATION ===
TASKS=("reacher-easy" "walker-stand" "cheetah-run")
SEEDS=(42 123 456)
EPOCHS=100000

TASK_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))
TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

DATA_DIR="${DATA_DIR:-/home/hice1/eliu354/scratch/Data/tdmpc2/mt30}"

# === ENVIRONMENT ===
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo
export PATH=$CONDA_PREFIX/bin:$PATH

cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM

export PYTHONPATH=/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/src:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/scripts:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/external/tdmpc2:$PYTHONPATH
export MUJOCO_GL=egl
export LAZY_LEGACY_OP=0
export HYDRA_FULL_ERROR=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs/slurm/mt30

echo "=== MT30 Baseline ${EPOCHS} Epochs ==="
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "Epochs: ${EPOCHS}"
echo "Config: pwm_48M_mt_baseline (Original PWM Aligned)"
echo "================================"

python -u scripts/train_multitask.py -cn config_mt30 \
  alg=pwm_48M_mt_baseline \
  task=${TASK} \
  general.seed=${SEED} \
  general.data_dir=${DATA_DIR} \
  general.epochs=${EPOCHS} \
  general.eval_freq=500 \
  general.eval_runs=10 \
  general.finetune_wm=True \
  general.run_wandb=True \
  ++wandb.project=MT30-Detailed \
  ++wandb.group=epoch_sweep_100k_baseline \
  ++wandb.name=baseline_100k_${TASK}_s${SEED} \
  hydra.run.dir="outputs/epoch_sweep/baseline_100k/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}_s${SEED}"

echo "=== Training Complete ==="
