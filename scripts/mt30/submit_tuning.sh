#!/bin/bash
#SBATCH --job-name=mt30_flow_tuning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --partition=ice-gpu
#SBATCH --account=coc
#SBATCH --array=0-17
#SBATCH --output=logs/slurm/mt30/tuning_%A_%a.out
#SBATCH --error=logs/slurm/mt30/tuning_%A_%a.err

# Tuning Scope: 2 Tasks x 3 Configs x 3 Seeds = 18 Jobs (Array 0-17)
TASKS=("walker-stand" "cheetah-run")
CONFIGS=("high_precision_wm" "high_precision_policy" "euler_fast")
SEEDS=(42 123 456)

# Calculate indices
# Indexing: Task -> Config -> Seed
# Total = 2 * 3 * 3 = 18
TASK_IDX=$((SLURM_ARRAY_TASK_ID / 9))
REM_1=$((SLURM_ARRAY_TASK_ID % 9))
CONFIG_IDX=$((REM_1 / 3))
SEED_IDX=$((REM_1 % 3))

TASK=${TASKS[$TASK_IDX]}
CONFIG_NAME=${CONFIGS[$CONFIG_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Define Hyperparameters based on Config Name
if [ "$CONFIG_NAME" == "high_precision_wm" ]; then
    # C1: Increase World Model substeps to 8 (Standard is 4)
    EXTRA_ARGS="alg.flow_substeps=8 alg.actor_config.flow_substeps=2"
elif [ "$CONFIG_NAME" == "high_precision_policy" ]; then
    # C2: Increase Policy substeps to 4 (Standard is 2)
    EXTRA_ARGS="alg.flow_substeps=4 alg.actor_config.flow_substeps=4"
elif [ "$CONFIG_NAME" == "euler_fast" ]; then
    # C3: Use Euler integrator for both (Faster, maybe simpler gradients)
    EXTRA_ARGS="alg.flow_integrator=euler alg.actor_config.flow_integrator=euler"
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
mkdir -p outputs/mt30_tuning/${TASK}/${CONFIG_NAME}/seed${SEED}

echo "=== MT30 Flow Tuning ==="
echo "Task: ${TASK}"
echo "Config: ${CONFIG_NAME}"
echo "Seed: ${SEED}"
echo "Extra Args: ${EXTRA_ARGS}"
echo "========================"

python -u scripts/train_multitask.py -cn config_mt30 \
  alg=pwm_48M_mt_fullflow \
  task=${TASK} \
  general.seed=${SEED} \
  general.data_dir=/home/hice1/eliu354/scratch/Data/tdmpc2/mt30 \
  general.epochs=15000 \
  general.eval_freq=200 \
  general.eval_runs=10 \
  general.finetune_wm=True \
  general.run_wandb=True \
  ++wandb.project=MT30-Detailed \
  ++wandb.group=mt30_tuning_${CONFIG_NAME} \
  ++wandb.name=${CONFIG_NAME}_${TASK}_s${SEED} \
  hydra.run.dir="outputs/mt30_tuning/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}_s${SEED}" \
  ${EXTRA_ARGS} \
  2>&1
