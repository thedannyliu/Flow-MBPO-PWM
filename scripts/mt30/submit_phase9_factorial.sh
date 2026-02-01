#!/bin/bash
#SBATCH --job-name=phase9_factorial
#SBATCH --output=logs/slurm/mt30/phase9_factorial_%A_%a.out
#SBATCH --error=logs/slurm/mt30/phase9_factorial_%A_%a.err
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=16:00:00
#SBATCH --array=0-35

# Phase 9: 2x2 Factorial Design (Factorial of WM Type x Policy Type)
# 4 Conditions x 9 Task-Seed Pairs = 36 Total Experiment Runs

# === ENVIRONMENT ===
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo
export PATH=$CONDA_PREFIX/bin:$PATH

cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM

export PYTHONPATH=/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/src:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/scripts:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/external/tdmpc2:$PYTHONPATH
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1

# === TASKS (9 entries) ===
TASKS=("reacher-easy" "reacher-easy" "reacher-easy" "walker-stand" "walker-stand" "walker-stand" "cheetah-run" "cheetah-run" "cheetah-run")
SEEDS=(42 123 456 42 123 456 42 123 456)

# === CHECKPOINTS ===
MLP_WM_CKPT="/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt"
FLOW_WM_CKPT="/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt"

# === JOB CONFIGURATION ===
# idx % 4 selects condition
# idx // 4 selects task/seed pair

COND_IDX=$((SLURM_ARRAY_TASK_ID % 4))
TASK_IDX=$((SLURM_ARRAY_TASK_ID / 4))

TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$TASK_IDX]}

# Define Conditions
case $COND_IDX in
  0)
    # 1. MLP WM + MLP Policy (Baseline)
    CONFIG="pwm_48M_mt_baseline"
    CKPT=$MLP_WM_CKPT
    NAME="phase9_cond0_mlp_mlp_${TASK}_s${SEED}"
    ;;
  1)
    # 2. MLP WM + Flow Policy
    CONFIG="pwm_48M_mt_flowpolicy"
    CKPT=$MLP_WM_CKPT
    NAME="phase9_cond1_mlp_flowpol_${TASK}_s${SEED}"
    ;;
  2)
    # 3. Flow WM + MLP Policy
    CONFIG="pwm_48M_mt_flowwm"
    CKPT=$FLOW_WM_CKPT
    NAME="phase9_cond2_flowwm_mlp_${TASK}_s${SEED}"
    ;;
  3)
    # 4. Flow WM + Flow Policy (Full Flow)
    CONFIG="pwm_48M_mt_fullflow"
    CKPT=$FLOW_WM_CKPT
    NAME="phase9_cond3_allflow_${TASK}_s${SEED}"
    ;;
esac

echo "=== Phase 9 Factorial ==="
echo "Condition Index: ${COND_IDX}"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CKPT}"
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "========================="

python scripts/train_multitask.py \
    alg=${CONFIG} \
    task=${TASK} \
    seed=${SEED} \
    general.checkpoint=${CKPT} \
    +general.checkpoint_with_buffer=False \
    +general.resume_training=False \
    general.data_dir="/home/hice1/eliu354/scratch/Data/tdmpc2/mt30" \
    general.epochs=50000 \
    general.run_wandb=True \
    +wandb.project=MT30-Detailed \
    +wandb.name="${NAME}"

echo "=== Training Complete ==="
