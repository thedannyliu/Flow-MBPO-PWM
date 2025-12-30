#!/bin/bash
#SBATCH --job-name=mt30_baseline
#SBATCH --output=logs/slurm/mt30/baseline_%j.out
#SBATCH --error=logs/slurm/mt30/baseline_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=16
#SBATCH -t 40:00:00
#SBATCH -A gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s

# MT30 Baseline: MLP World Model + MLP Policy
# Policy-only comparison baseline for multitask experiments

# === CONFIGURATION ===
TASK="${1:-reach-v2}"  # Default task, can override via command line
SEED="${2:-42}"
DATA_DIR="${3:-/path/to/tdmpc2/mt30}"  # MUST BE SET BY USER
CHECKPOINT="${4:-/path/to/pretrained_wm.pt}"  # MUST BE SET BY USER

# === ENVIRONMENT ===
source ~/.bashrc
conda activate pwm
cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM
export PYTHONPATH=src:$PYTHONPATH
export MUJOCO_GL=egl
export LAZY_LEGACY_OP=0

# Create log directories
mkdir -p logs/slurm/mt30
mkdir -p outputs/mt30/baseline/seed${SEED}

echo "=== MT30 Baseline Experiment ==="
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "Data dir: ${DATA_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "================================"

# === RUN TRAINING ===
python scripts/train_multitask.py -cn config_mt30 \
  alg=pwm_48M \
  task=${TASK} \
  general.seed=${SEED} \
  general.data_dir=${DATA_DIR} \
  general.checkpoint=${CHECKPOINT} \
  general.epochs=10000 \
  general.eval_freq=200 \
  general.eval_runs=10 \
  general.finetune_wm=False \
  general.run_wandb=True \
  ++wandb.project=flow-mbpo-multitask \
  ++wandb.group=mt30_baseline \
  ++wandb.name=baseline_${TASK}_s${SEED}

echo "=== Training Complete ==="
