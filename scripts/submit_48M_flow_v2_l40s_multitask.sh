#!/bin/bash
#SBATCH --job-name=train_mt30_flow_v2
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=embers
#SBATCH --output=/storage/project/r-agarg35-0/nnguyen349/logs/train_mt30_flow_v2_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/nnguyen349/logs/train_mt30_flow_v2_%j.err


set -e  # Exit on error

source $(conda info --base)/etc/profile.d/conda.sh
conda activate pwm

# Ignore user site-packages (~/.local) to avoid torch version conflicts
export PYTHONNOUSERSITE=1

# Suppress MuJoCo/dm-control EGL warnings (harmless but noisy)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Create logs directory
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
OUTPUT_DIR="/storage/project/r-agarg35-0/nnguyen349/tdmpc2_flow/logs"

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
echo "Training Flow-MBPO V2 48M Multi-task (MT30)"
echo "============================================"
echo "Method: Flow-matching world model"
echo "Config: use_flow=true, heun integrator, 4 substeps"
echo "============================================"

cd "${TDMPC2_DIR}"
echo "Working directory: $(pwd)"

export WANDB_API_KEY="649ebee0bba0d06876e022242e4f9924fdefaf3e"
WANDB_PROJECT="flowmbpo"

echo "Starting training..."
echo "Wandb project: ${WANDB_PROJECT}"

# Train Flow world model on all 30 tasks (same as PWM baseline but with flow dynamics)
# Flow V2 config:
#   use_flow=true: Enables flow-matching dynamics
#   flow_integrator=heun: Uses Heun's method (2nd order)
#   flow_substeps=4: 4 integration substeps (more accurate than v1)
#   exp_name=flowv2_48M_mt30: For wandb tracking

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
    exp_name="flowv2_48M_mt30" \
    use_flow=true \
    flow_integrator=heun \
    flow_substeps=4 \
    data_dir="${ACTUAL_DATA_DIR}" \
    hydra.run.dir="${OUTPUT_DIR}"

echo ""
echo "============================================"
echo "Training complete!"
echo "============================================"
