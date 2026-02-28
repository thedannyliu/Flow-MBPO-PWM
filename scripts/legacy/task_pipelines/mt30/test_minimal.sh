#!/bin/bash
#SBATCH --job-name=mt30_test
#SBATCH --output=logs/slurm/mt30/test_%j.out
#SBATCH --error=logs/slurm/mt30/test_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=450GB
#SBATCH --cpus-per-task=16
#SBATCH -t 00:30:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Minimal test for MT30 training
# Runs for 10 epochs on reacher-easy

# === ENVIRONMENT ===
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo
export PATH=$CONDA_PREFIX/bin:$PATH

cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM

# Set PYTHONPATH with absolute paths
export PYTHONPATH=/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/src:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/scripts:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/external/tdmpc2:$PYTHONPATH

DATA_DIR="/home/hice1/eliu354/scratch/Data/tdmpc2/mt30"
CHECKPOINT="/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/checkpoints/multitask/mt30_48M_4900000.pt"

# Create log directories
mkdir -p logs/slurm/mt30

# === RUN TRAINING (Minimal Test) ===
python scripts/train_multitask.py -cn config_mt30 \
  alg=pwm_48M_mt_baseline \
  task=reacher-easy \
  general.seed=42 \
  general.data_dir=${DATA_DIR} \
  general.checkpoint=${CHECKPOINT} \
  general.epochs=10 \
  general.eval_freq=5 \
  general.eval_runs=2 \
  general.finetune_wm=False \
  general.run_wandb=False \
  ++wandb.project=flow-mbpo-multitask \
  ++wandb.group=mt30_test \
  ++wandb.name=test_reacher-easy_s42

echo "=== Minimal Test Complete ==="
