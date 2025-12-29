#!/bin/bash
#SBATCH --job-name=train_mt30_wm
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=embers
#SBATCH --output=/storage/project/r-agarg35-0/nnguyen349/logs/train_mt30_wm_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/nnguyen349/logs/train_mt30_wm_%j.err


set -e  # Exit on error

source $(conda info --base)/etc/profile.d/conda.sh
conda activate pwm

# Ignore user site-packages (~/.local) to avoid torch version conflicts
export PYTHONNOUSERSITE=1

# Suppress MuJoCo/dm-control EGL warnings (harmless but noisy)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Create logs directory
# TODO: change username
mkdir -p /storage/project/r-agarg35-0/nnguyen349/logs

# Set cache directories to avoid filling home directory
export PIP_CACHE_DIR="/storage/project/r-agarg35-0/nnguyen349/.pip_cache"
export HF_HOME="/storage/project/r-agarg35-0/nnguyen349/.huggingface"
export HF_DATASETS_CACHE="/storage/project/r-agarg35-0/nnguyen349/.huggingface/datasets"
export TRANSFORMERS_CACHE="/storage/project/r-agarg35-0/nnguyen349/.transformer"
export WANDB_DIR="/storage/project/r-agarg35-0/nnguyen349/.wandb"

# Configuration
BASE_DIR="/storage/home/hcoda1/6/nnguyen349/r-agarg35-0"
DATA_DIR="${BASE_DIR}/pwm_data/mt30"
TDMPC2_DIR="${BASE_DIR}/tdmpc2/tdmpc2"
OUTPUT_DIR="/storage/project/r-agarg35-0/nnguyen349/tdmpc2/tdmpc2/logs"
RESUME_CHECKPOINT="/storage/project/r-agarg35-0/nnguyen349/tdmpc2/tdmpc2/logs/mt30/1/default/models/100000.pt"

# Create output directory for training logs/checkpoints
mkdir -p "${OUTPUT_DIR}"

# Determine the actual data path (it might be in mt30 subfolder)
if [ -d "${DATA_DIR}/mt30" ]; then
    ACTUAL_DATA_DIR="${DATA_DIR}/mt30"
else
    ACTUAL_DATA_DIR="${DATA_DIR}"
fi

# Check if data exists
if [ -z "$(ls -A ${ACTUAL_DATA_DIR}/*.pt 2>/dev/null)" ]; then
    echo "ERROR: No .pt files found in ${ACTUAL_DATA_DIR}"
    echo "Please download the MT30 dataset first from:"
    echo "https://huggingface.co/datasets/nicklashansen/tdmpc2/tree/main/mt30"
    exit 1
fi

echo "============================================"
echo "Data directory: ${ACTUAL_DATA_DIR}"
echo "Contents:"
ls -la "${ACTUAL_DATA_DIR}" | head -20
echo "============================================"

echo ""
echo "============================================"
echo "Training TD-MPC2 Multi-task World Model"
echo "============================================"

cd "${TDMPC2_DIR}"
echo "Working directory: $(pwd)"

export WANDB_API_KEY="649ebee0bba0d06876e022242e4f9924fdefaf3e"
WANDB_PROJECT="flowmbpo" 

echo "Starting training..."
echo "Wandb project: ${WANDB_PROJECT}"

# Resume from checkpoint options:
# - auto_resume=true (default): Automatically finds and resumes from latest checkpoint
# - resume_checkpoint=/path/to/checkpoint.pt: Resume from specific checkpoint
# - auto_resume=false: Start fresh training (ignore existing checkpoints)
# Checkpoints are saved at: ${OUTPUT_DIR}/mt30/<seed>/default/models/<iteration>.pt

python -u train.py \
    task=mt30 \
    model_size=48 \
    horizon=16 \
    batch_size=1024 \
    rho=0.99 \
    mpc=false \
    compile=false \
    enable_wandb=true \
    wandb_project="${WANDB_PROJECT}" \
    checkpoint="${RESUME_CHECKPOINT}" \
    data_dir="${ACTUAL_DATA_DIR}" \
    hydra.run.dir="${OUTPUT_DIR}"

echo ""
echo "============================================"
echo "Training complete!"
echo "============================================"
