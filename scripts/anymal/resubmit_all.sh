#!/bin/bash
# Resubmit all Anymal training jobs with fixed WandB naming
# Jobs to submit: Baseline (s42,123,456) + Flow K=2,4,8 (s42,123,456) = 12 jobs
# Note: Job 3076505 (Flow K=4 s123) is still running, skip it

set -e

PROJECT_ROOT="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
MEM="128GB"
TIME="40:00:00"
CPUS="16"

echo "==========================================="
echo "Resubmitting Anymal Training Jobs with Fixed WandB Naming"
echo "Project: $PROJECT_ROOT"
echo "Time: $(date)"
echo "==========================================="

mkdir -p $PROJECT_ROOT/logs/slurm/anymal

SEEDS=(42 123 456)

# =============================================
# Submit Baseline jobs (3 seeds)
# =============================================
echo ""
echo "=== Submitting Baseline (MLP WM + MLP Policy) ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_baseline_s${SEED}"
    RUN_NAME="Anymal_Baseline_MLP_s${SEED}"
    RUN_NOTES="Anymal baseline training MLP WM seed ${SEED}"
    
    echo "Submitting baseline seed=$SEED as '$RUN_NAME'..."
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=logs/slurm/anymal/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/anymal/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src
mkdir -p logs/slurm/anymal

cd scripts
python train_dflex.py \\
    env=dflex_anymal \\
    alg=pwm_5M_baseline_final \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name="${RUN_NAME}" \\
    ++wandb.notes="${RUN_NOTES}"
EOF
done

# =============================================
# Submit Flow K=4 Heun (best default) (3 seeds)
# =============================================
echo ""
echo "=== Submitting Flow WM K=4 (Heun) ==="
for SEED in "${SEEDS[@]}"; do
    # Skip job 3076505 which is still running
    if [ "$SEED" -eq 123 ]; then
        echo "Skipping Flow K=4 seed=123 (job 3076505 still running)"
        continue
    fi
    
    JOB_NAME="anymal_flowWM_K4_s${SEED}"
    RUN_NAME="Anymal_FlowWM_K4Heun_s${SEED}"
    RUN_NOTES="Anymal Flow WM K=4 Heun integrator seed ${SEED}"
    
    echo "Submitting FlowWM K=4 seed=$SEED as '$RUN_NAME'..."
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=logs/slurm/anymal/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/anymal/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src
mkdir -p logs/slurm/anymal

cd scripts
python train_dflex.py \\
    env=dflex_anymal \\
    alg=pwm_5M_flow_v2_substeps4 \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name="${RUN_NAME}" \\
    ++wandb.notes="${RUN_NOTES}"
EOF
done

# =============================================
# Submit Flow K=2 Heun (faster) (3 seeds)
# =============================================
echo ""
echo "=== Submitting Flow WM K=2 (Heun) ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowWM_K2_s${SEED}"
    RUN_NAME="Anymal_FlowWM_K2Heun_s${SEED}"
    RUN_NOTES="Anymal Flow WM K=2 Heun integrator seed ${SEED}"
    
    echo "Submitting FlowWM K=2 seed=$SEED as '$RUN_NAME'..."
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=logs/slurm/anymal/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/anymal/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src
mkdir -p logs/slurm/anymal

cd scripts
python train_dflex.py \\
    env=dflex_anymal \\
    alg=pwm_5M_flow_v1_substeps2 \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name="${RUN_NAME}" \\
    ++wandb.notes="${RUN_NOTES}"
EOF
done

# =============================================
# Submit Flow K=8 Euler (highest accuracy) (3 seeds)
# =============================================
echo ""
echo "=== Submitting Flow WM K=8 (Euler) ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowWM_K8_s${SEED}"
    RUN_NAME="Anymal_FlowWM_K8Euler_s${SEED}"
    RUN_NOTES="Anymal Flow WM K=8 Euler integrator seed ${SEED}"
    
    echo "Submitting FlowWM K=8 Euler seed=$SEED as '$RUN_NAME'..."
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=logs/slurm/anymal/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/anymal/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src
mkdir -p logs/slurm/anymal

cd scripts
python train_dflex.py \\
    env=dflex_anymal \\
    alg=pwm_5M_flow_v3_substeps8_euler \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name="${RUN_NAME}" \\
    ++wandb.notes="${RUN_NOTES}"
EOF
done

echo ""
echo "==========================================="
echo "All jobs submitted!"
echo "Expected jobs: 11 (3 baseline + 2 K=4 + 3 K=2 + 3 K=8)"
echo "Note: K=4 s123 skipped (job 3076505 still running)"
echo "Use 'squeue -u \$USER' to check status"
echo "==========================================="
