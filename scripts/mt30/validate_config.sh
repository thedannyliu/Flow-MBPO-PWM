#!/bin/bash
#SBATCH --job-name=mt30_test
#SBATCH --output=logs/slurm/mt30/test_%j.out
#SBATCH --error=logs/slurm/mt30/test_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=450GB
#SBATCH --cpus-per-task=16
#SBATCH -t 00:20:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Quick validation test before full training submission
# Tests baseline config with walker-stand for 200 epochs

# === ENVIRONMENT ===
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo
export PATH=$CONDA_PREFIX/bin:$PATH

cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM
export PYTHONPATH=/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/src:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/scripts:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/external/tdmpc2:$PYTHONPATH
export MUJOCO_GL=egl
export LAZY_LEGACY_OP=0

DATA_DIR="/home/hice1/eliu354/scratch/Data/tdmpc2/mt30"
CHECKPOINT="/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/checkpoints/multitask/mt30_48M_4900000.pt"

echo "=== Quick Validation Test ==="
echo "Testing: walker-stand (seed 42)"
echo "Epochs: 200"
echo "Purpose: Verify training works before full submission"
echo "=============================="

# Run short training
python scripts/train_multitask.py -cn config_mt30 \
  alg=pwm_48M_mt_baseline \
  task=walker-stand \
  general.seed=42 \
  general.data_dir=${DATA_DIR} \
  general.checkpoint=${CHECKPOINT} \
  general.epochs=200 \
  general.eval_freq=100 \
  general.eval_runs=3 \
  general.finetune_wm=False \
  general.run_wandb=False

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Validation test PASSED"
else
    echo "✗ Validation test FAILED"
fi
