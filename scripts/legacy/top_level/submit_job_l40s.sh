#!/bin/bash

# Submit training job to single L40s GPU for comparison with H200
# Usage: ./submit_job_l40s.sh <alg_config> <env_config> <seed>

ALG_CONFIG=${1:-pwm_5M}
ENV_CONFIG=${2:-dflex_ant}
SEED=${3:-42}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Submitting L40s Training Job"
echo "=========================================="
echo "Algorithm: $ALG_CONFIG"
echo "Environment: $ENV_CONFIG"
echo "Seed: $SEED"
echo "=========================================="

# Create SLURM job script
cat > "${PROJECT_DIR}/scripts/temp_l40s_job.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${ALG_CONFIG}_${ENV_CONFIG}_l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=inferno
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=atl1-1-03-004-31-0,atl1-1-03-007-29-0,atl1-1-03-007-31-0
#SBATCH --output=logs/slurm/${ALG_CONFIG}_${ENV_CONFIG}_seed${SEED}_l40s_%j.out
#SBATCH --error=logs/slurm/${ALG_CONFIG}_${ENV_CONFIG}_seed${SEED}_l40s_%j.err

# Load conda environment
source ~/.bashrc
conda activate pwm

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

# Run training with L40s-optimized config
cd ${PROJECT_DIR}
python scripts/train_dflex.py \\
    alg=${ALG_CONFIG}_l40s \\
    env=${ENV_CONFIG}_l40s \\
    general.seed=${SEED}

echo "Training completed!"
EOF

# Submit the job
sbatch "${PROJECT_DIR}/scripts/temp_l40s_job.sh"

# Clean up
rm "${PROJECT_DIR}/scripts/temp_l40s_job.sh"

echo "Job submitted to L40s partition"
