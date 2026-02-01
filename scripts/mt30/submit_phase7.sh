#!/bin/bash
#SBATCH --job-name=mt30_phase7
#SBATCH --output=logs/slurm/mt30/phase7_%A_%a.out
#SBATCH --error=logs/slurm/mt30/phase7_%A_%a.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=450GB
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu
#SBATCH --array=0-26

# Phase 7: Flow Policy Fine-tuning (Pretrained WM)
# Goal: Compare Baseline vs Flow Policy (Std/High) with WM Fine-tuning enabled
# Variants:
# 0: Baseline (MLP)
# 1: Flow Standard (substeps=2)
# 2: Flow High Precision (substeps=4/8)

# Matrix: 3 Variants x 3 Tasks x 3 Seeds = 27 Jobs

VARIANTS=("baseline" "flow_std" "flow_high")
TASKS=("reacher-easy" "walker-stand" "cheetah-run")
SEEDS=(42 123 456)

# Calculate indices
# Total = 27
# Variant (3) -> Task (3) -> Seed (3)
VARIANT_IDX=$((SLURM_ARRAY_TASK_ID / 9))
REM_1=$((SLURM_ARRAY_TASK_ID % 9))
TASK_IDX=$((REM_1 / 3))
SEED_IDX=$((REM_1 % 3))

VARIANT=${VARIANTS[$VARIANT_IDX]}
TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

DATA_DIR="${DATA_DIR:-/home/hice1/eliu354/scratch/Data/tdmpc2/mt30}"
CHECKPOINT="/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/checkpoints/multitask/mt30_48M_4900000.pt"

# === CONFIGURATION SELECTOR ===
if [ "$VARIANT" == "baseline" ]; then
    ALG="pwm_48M_mt_baseline"
    EXTRA_ARGS=""
    # Baseline uses MLP, usually substeps irrelevant but keeping clean
elif [ "$VARIANT" == "flow_std" ]; then
    ALG="pwm_48M_mt_flowpolicy"
    # Standard Flow: Substeps=2 (Policy)
    EXTRA_ARGS="++alg.actor_config.flow_substeps=2"
elif [ "$VARIANT" == "flow_high" ]; then
    ALG="pwm_48M_mt_flowpolicy"
    # High Precision Flow: WM=8 (if finetuning allows), Policy=4
    # Note: WM is MLP here, so wm_substeps irrelevant, only Policy matters
    EXTRA_ARGS="++alg.actor_config.flow_substeps=4"
fi

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

mkdir -p logs/slurm/mt30

echo "=== Phase 7: Fine-tuning Comparison ==="
echo "Variant: ${VARIANT}"
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "Algorithm: ${ALG}"
echo "Extra Args: ${EXTRA_ARGS}"
echo "Finetune WM: True"
echo "======================================"

python -u scripts/train_multitask.py -cn config_mt30 \
  alg=${ALG} \
  task=${TASK} \
  general.seed=${SEED} \
  general.data_dir=${DATA_DIR} \
  general.checkpoint=${CHECKPOINT} \
  general.epochs=15000 \
  general.eval_freq=500 \
  general.eval_runs=10 \
  general.finetune_wm=True \
  general.run_wandb=True \
  ++wandb.project=MT30-Detailed \
  ++wandb.group=phase7_finetuning \
  ++wandb.name=${VARIANT}_${TASK}_s${SEED} \
  hydra.run.dir="outputs/phase7/${VARIANT}/${TASK}/${SEED}" \
  ${EXTRA_ARGS}

echo "=== Training Complete ==="
