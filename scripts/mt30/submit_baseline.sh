#!/bin/bash
#SBATCH --job-name=mt30_baseline
#SBATCH --output=logs/slurm/mt30/baseline_%A_%a.out
#SBATCH --error=logs/slurm/mt30/baseline_%A_%a.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=450GB
#SBATCH --cpus-per-task=16
#SBATCH -t 16:00:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu
#SBATCH --array=0-8

# MT30 Baseline: MLP World Model + MLP Policy
# Uses pwm_48M_mt_baseline.yaml - ALIGNED WITH ORIGINAL PWM
# Array job: 3 tasks x 3 seeds = 9 jobs (indices 0-8)

# === CONFIGURATION ===
# DMControl tasks for MT30 (not MetaWorld)
TASKS=("reacher-easy" "walker-stand" "cheetah-run")
SEEDS=(42 123 456)

# Calculate task and seed from array index
TASK_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))
TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Data paths
DATA_DIR="${DATA_DIR:-/home/hice1/eliu354/scratch/Data/tdmpc2/mt30}"
CHECKPOINT="${CHECKPOINT:-/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/checkpoints/multitask/mt30_48M_4900000.pt}"

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

# Diagnostics
echo "--- GPU Info ---"
nvidia-smi
echo "--- CUDA Visible Devices ---"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "----------------"

# Create log directories
mkdir -p logs/slurm/mt30
mkdir -p outputs/mt30/baseline/${TASK}/seed${SEED}

echo "=== MT30 Baseline Experiment ==="
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "Task Index: ${TASK_IDX}, Seed Index: ${SEED_IDX}"
echo "Data dir: ${DATA_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(which python)"
echo "Config: pwm_48M_mt_baseline (aligned with original PWM)"
echo "================================"

# Verify paths exist
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

# === RUN TRAINING ===
python -u scripts/train_multitask.py -cn config_mt30 \
  alg=pwm_48M_mt_baseline \
  task=${TASK} \
  general.seed=${SEED} \
  general.data_dir=${DATA_DIR} \
  general.checkpoint=${CHECKPOINT} \
  general.epochs=10000 \
  general.eval_freq=200 \
  general.eval_runs=10 \
  general.finetune_wm=False \
  general.run_wandb=True \
  ++wandb.project=MT30-multitasks \
  ++wandb.group=mt30_baseline \
  ++wandb.name="baseline_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  hydra.run.dir="outputs/mt30/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}_s${SEED}"

echo "=== Training Complete ==="
