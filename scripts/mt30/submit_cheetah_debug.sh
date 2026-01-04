#!/bin/bash
#SBATCH --job-name=mt30_cheetah_h30
#SBATCH --output=logs/slurm/mt30/cheetah_h30_%A_%a.out
#SBATCH --error=logs/slurm/mt30/cheetah_h30_%A_%a.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=450GB
#SBATCH --cpus-per-task=16
#SBATCH -t 16:00:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu
#SBATCH --array=0-2

# MT30 Full Flow: Flow World Model + Flow ODE Policy
# Uses pwm_48M_mt_fullflow.yaml - Full Flow comparison (do after Policy-only)
# NOTE: Higher memory allocation due to Flow WM ODE integration overhead

# === CONFIGURATION ===
TASKS=("cheetah-run")
SEEDS=(42 123 456)

TASK=${TASKS[0]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR="${DATA_DIR:-/home/hice1/eliu354/scratch/Data/tdmpc2/mt30}"
# NOTE: For Full Flow, you need a FLOW WM checkpoint, not the baseline MLP WM
# This requires pretraining a Flow WM first OR testing with finetune_wm=True
# CHECKPOINT not used for Full Flow from scratch
# CHECKPOINT="${CHECKPOINT:-...}"

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

# if [ ! -f "${CHECKPOINT}" ]; then
#     echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
#     exit 1
# fi

python -u scripts/train_multitask.py -cn config_mt30 \
  alg=pwm_48M_mt_fullflow \
  task=${TASK} \
  general.seed=${SEED} \
  general.data_dir=${DATA_DIR} \
  general.epochs=5000 \
  general.eval_freq=200 \
  general.eval_runs=10 \
  general.finetune_wm=True \
  general.run_wandb=True \
  ++wandb.project=MT30-Detailed \
  ++wandb.group=mt30_debug_h30 \
  ++wandb.name=debug_cheetah_h30_s${SEED} \
  hydra.run.dir="outputs/mt30_debug/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}_s${SEED}" \
  ++horizon=30

echo "=== Training Complete ==="
