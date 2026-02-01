#!/bin/bash
#SBATCH --job-name=wm_pretrain_mlp
#SBATCH --output=logs/slurm/pretrain/mlp_wm_%j.out
#SBATCH --error=logs/slurm/pretrain/mlp_wm_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=450GB
#SBATCH --cpus-per-task=16
#SBATCH -t 16:00:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Phase 8: MLP World Model Pretraining (Control/Optional)
# This script pretrains an MLP WM on MT30 dataset for clean 2x2 factorial comparison.
# Output: <out_name>_best.pt and <out_name>_last.pt in Hydra output dir.

DATA_DIR="${DATA_DIR:-/home/hice1/eliu354/scratch/Data/tdmpc2/mt30}"
WM_ITERS="${WM_ITERS:-200000}"

# === ENVIRONMENT ===
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo
export PATH=$CONDA_PREFIX/bin:$PATH

cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM

export PYTHONPATH=/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/src:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/scripts:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/external/tdmpc2:$PYTHONPATH
export MUJOCO_GL=egl
export LAZY_LEGACY_OP=0
export HYDRA_FULL_ERROR=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs/slurm/pretrain

echo "=== Phase 8: MLP WM Pretraining (Control) ==="
echo "Data Dir: ${DATA_DIR}"
echo "WM Iterations: ${WM_ITERS}"
echo "GPU: H100"
echo "=============================================="

python -u scripts/pretrain_multitask_wm.py \
  -cn pretrain_mt30_wm \
  alg=pwm_48M_mt_baseline \
  general.data_dir=${DATA_DIR} \
  general.wm_pretrain_iters=${WM_ITERS} \
  general.out_name=mlpwm_mt30 \
  general.seed=42 \
  general.run_wandb=True \
  ++wandb.project=WM-Pretrain \
  ++wandb.group=phase8_mlp_wm \
  ++wandb.name=mlpwm_mt30_${WM_ITERS}

echo "=== MLP WM Pretraining Complete ==="
