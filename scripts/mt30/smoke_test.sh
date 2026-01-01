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
# - Uses baseline config (MLP WM + MLP Policy)
# - Runs only 100 epochs with 1 eval run
# - Single task (reach-v2), single seed (42)

# === CONFIGURATION ===
TASK="${1:-reach-v2}"
SEED="${2:-42}"
EPOCHS="${3:-100}"

DATA_DIR="${DATA_DIR:-/home/hice1/eliu354/scratch/Data/tdmpc2/mt30}"
CHECKPOINT="${CHECKPOINT:-/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/checkpoints/multitask/mt30_48M_4900000.pt}"

# === ENVIRONMENT ===
source ~/.bashrc
conda activate flow-mbpo
cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM
export PYTHONPATH=src:$PYTHONPATH
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
