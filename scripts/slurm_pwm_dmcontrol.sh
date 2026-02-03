#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -N 1
#SBATCH --mem-per-gpu=61G
#SBATCH -q embers
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH -J pwm_dmcontrol
#SBATCH -o /storage/home/hcoda1/5/fchang40/Flow-MBPO-PWM/logs/slurm/pwm_dmcontrol_%j.out
#SBATCH -e /storage/home/hcoda1/5/fchang40/Flow-MBPO-PWM/logs/slurm/pwm_dmcontrol_%j.err

# PWM Online Training on All DMControl Environments
# Trains 1 trial per environment with 1000 epochs and 32 parallel envs

set -e

# Setup
REPO_ROOT="/storage/home/hcoda1/5/fchang40/Flow-MBPO-PWM"
cd "$REPO_ROOT"

# Create log directories
mkdir -p "$REPO_ROOT/logs/slurm"
mkdir -p "$REPO_ROOT/logs/pwm_dmcontrol"

# Training parameters
NUM_ENVS=32
MAX_EPOCHS=1000
SEED=42

# DMControl tasks available in MuJoCo Playground
TASKS=(
    "WalkerStand"
    "WalkerWalk"
    "WalkerRun"
    "CheetahRun"
    "HopperStand"
    "HopperHop"
    "CartpoleBalance"
    "CartpoleBalanceSparse"
    "CartpoleSwingup"
    "CartpoleSwingupSparse"
    "ReacherEasy"
    "ReacherHard"
    "FingerSpin"
    "FingerTurnEasy"
    "FingerTurnHard"
    "PendulumSwingup"
    "BallInCup"
    "FishSwim"
    "AcrobotSwingup"
    "HumanoidStand"
    "HumanoidWalk"
    "HumanoidRun"
    "PointMass"
    "SwimmerSwimmer6"
)

# Function to convert CamelCase to snake_case
to_snake_case() {
    echo "$1" | sed 's/\([A-Z]\)/_\L\1/g' | sed 's/^_//'
}

echo "=============================================="
echo "PWM Online Training - DMControl Environments"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Num Envs: $NUM_ENVS"
echo "Max Epochs: $MAX_EPOCHS"
echo "Seed: $SEED"
echo "Total Tasks: ${#TASKS[@]}"
echo "=============================================="

# Activate environment if needed
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

# Train on each task sequentially
for TASK in "${TASKS[@]}"; do
    TASK_SNAKE=$(to_snake_case "$TASK")
    LOG_FILE="logs/pwm_dmcontrol/${TASK_SNAKE}.log"

    echo ""
    echo "----------------------------------------------"
    echo "Starting: $TASK"
    echo "Log file: $LOG_FILE"
    echo "Time: $(date)"
    echo "----------------------------------------------"

    # Run training and tee output to both console and log file
    python scripts/train_pwm_dmcontrol.py \
        --task "$TASK" \
        --num_envs "$NUM_ENVS" \
        --max_epochs "$MAX_EPOCHS" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Completed: $TASK (success)"
    else
        echo "WARNING: $TASK failed with exit code $EXIT_CODE"
    fi
done

echo ""
echo "=============================================="
echo "ALL TRAINING COMPLETE"
echo "Time: $(date)"
echo "=============================================="

# Print summary of all results
echo ""
echo "Results Summary:"
echo "----------------"
for TASK in "${TASKS[@]}"; do
    TASK_SNAKE=$(to_snake_case "$TASK")
    SUMMARY_FILE="logs/pwm_dmcontrol/${TASK_SNAKE}/seed_${SEED}/summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        REWARD=$(python -c "import json; print(json.load(open('$SUMMARY_FILE'))['best_reward'])" 2>/dev/null || echo "N/A")
        printf "%-25s: %s\n" "$TASK" "$REWARD"
    else
        printf "%-25s: %s\n" "$TASK" "FAILED"
    fi
done
