#!/bin/bash
#SBATCH --job-name=mt30_flow_50k
#SBATCH --output=logs/slurm/mt30/flow_50k_%A_%a.out
#SBATCH --error=logs/slurm/mt30/flow_50k_%A_%a.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=450GB
#SBATCH --cpus-per-task=16
#SBATCH -t 08:00:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu
#SBATCH --array=0-8

# MT30 Full Flow 50K: High Precision Flow WM + Flow Policy
# Training from scratch with finetune_wm=True

# === CONFIGURATION ===
TASKS=("reacher-easy" "walker-stand" "cheetah-run")
SEEDS=(42 123 456)
EPOCHS=50000

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

echo "=== MT30 Full Flow ${EPOCHS} Epochs (High Precision) ==="
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "Epochs: ${EPOCHS}"
echo "Config: pwm_48M_mt_fullflow + High Precision"
echo "Flow Substeps (WM): 8, Flow Substeps (Policy): 4"
echo "================================"

python -u scripts/train_multitask.py -cn config_mt30 \
  alg=pwm_48M_mt_fullflow \
  task=${TASK} \
  general.seed=${SEED} \
  general.data_dir=${DATA_DIR} \
  general.epochs=${EPOCHS} \
  general.eval_freq=500 \
  general.eval_runs=10 \
  general.finetune_wm=True \
  general.run_wandb=True \
  ++alg.flow_substeps=8 \
  ++alg.actor_config.flow_substeps=4 \
  ++wandb.project=MT30-Detailed \
  ++wandb.group=epoch_sweep_50k_flow \
  ++wandb.name=flow_50k_${TASK}_s${SEED} \
  hydra.run.dir="outputs/epoch_sweep/flow_50k/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}_s${SEED}"

echo "=== Training Complete ==="
