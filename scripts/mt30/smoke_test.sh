#!/bin/bash
#SBATCH --job-name=mt30_smoke
#SBATCH --output=logs/slurm/mt30/smoke_%j.out
#SBATCH --error=logs/slurm/mt30/smoke_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=495GB
#SBATCH --cpus-per-task=16
#SBATCH -t 16:00:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# MT30 Smoke Test: Minimal test to verify training pipeline works

# === CONFIGURATION ===
TASK="${1:-reacher-easy}"
SEED="${2:-42}"
EPOCHS="${3:-100}"

DATA_DIR="${DATA_DIR:-/home/hice1/eliu354/scratch/Data/tdmpc2/mt30}"
CHECKPOINT="${CHECKPOINT:-/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/checkpoints/multitask/mt30_48M_4900000.pt}"

# === ENVIRONMENT ===
# Properly activate conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo

# Force correct Python from conda env
export PATH=$CONDA_PREFIX/bin:$PATH

# Set working directory
cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM

# Set PYTHONPATH to include src, scripts, and external/tdmpc2
export PYTHONPATH=/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/src:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/scripts:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/external/tdmpc2:$PYTHONPATH
export MUJOCO_GL=egl
export LAZY_LEGACY_OP=0

# Create log directories
mkdir -p logs/slurm/mt30
mkdir -p outputs/mt30/smoke/${TASK}/seed${SEED}

echo "=== MT30 SMOKE TEST ==="
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "Epochs: ${EPOCHS}"
echo "Data dir: ${DATA_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(which python)"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo "========================"

# Verify paths exist
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

# List data files
echo "Data files found:"
ls -la ${DATA_DIR}/*.pt 2>/dev/null | head -5

# Test DMControl environment creation first
echo ""
echo "=== Testing DMControl Environment ==="
python -c "
from dm_control import suite
print('dm_control imported successfully')
env = suite.load('walker', 'stand')
print('walker-stand loaded successfully')
obs = env.reset()
print('Environment reset successful')
" || { echo "ERROR: DMControl test failed"; exit 1; }

# === RUN MINIMAL TRAINING ===
echo ""
echo "Starting training..."
python scripts/train_multitask.py -cn config_mt30 \
  alg=pwm_48M_mt_baseline \
  task=${TASK} \
  general.seed=${SEED} \
  general.data_dir=${DATA_DIR} \
  general.checkpoint=${CHECKPOINT} \
  general.epochs=${EPOCHS} \
  general.eval_freq=50 \
  general.eval_runs=1 \
  general.finetune_wm=False \
  general.run_wandb=False

EXIT_CODE=$?
echo ""
echo "=== SMOKE TEST COMPLETE ==="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
