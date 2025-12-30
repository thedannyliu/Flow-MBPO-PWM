#!/bin/bash
#SBATCH --job-name=mt30_fullflow
#SBATCH --output=logs/slurm/mt30/fullflow_%A_%a.out
#SBATCH --error=logs/slurm/mt30/fullflow_%A_%a.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=450GB
#SBATCH --cpus-per-task=16
#SBATCH -t 40:00:00
#SBATCH -A gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --array=0-2

# MT30 Full Flow: Flow World Model + Flow ODE Policy
# Uses pwm_48M_mt_fullflow.yaml - Full Flow comparison (do after Policy-only)
# NOTE: Higher memory allocation due to Flow WM ODE integration overhead

# === CONFIGURATION ===
TASKS=("reach-v2" "push-v2" "pick-place-v2")
SEEDS=(42 123 456)
TASK="${1:-${TASKS[0]}}"
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR="${DATA_DIR:-/home/hice1/eliu354/scratch/Data/tdmpc2/mt30}"
# NOTE: For Full Flow, you need a FLOW WM checkpoint, not the baseline MLP WM
# This requires pretraining a Flow WM first OR testing with finetune_wm=True
CHECKPOINT="${CHECKPOINT:-/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/checkpoints/mt30_48M_4900000.pt}"

# === ENVIRONMENT ===
source ~/.bashrc
conda activate flow-mbpo
cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM
export PYTHONPATH=src:$PYTHONPATH
export MUJOCO_GL=egl
export LAZY_LEGACY_OP=0

mkdir -p logs/slurm/mt30
mkdir -p outputs/mt30/fullflow/${TASK}/seed${SEED}

echo "=== MT30 Full Flow Experiment ==="
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "Data dir: ${DATA_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Config: pwm_48M_mt_fullflow"
echo "WARNING: Full Flow requires Flow WM checkpoint or finetune_wm=True"
echo "================================"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

python scripts/train_multitask.py -cn config_mt30 \
  alg=pwm_48M_mt_fullflow \
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
  ++wandb.group=mt30_fullflow \
  ++wandb.name=fullflow_${TASK}_s${SEED}

echo "=== Training Complete ==="
