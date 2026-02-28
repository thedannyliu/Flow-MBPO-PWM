#!/bin/bash
# Resubmit Flow WM jobs with 256GB memory to fix OOM
# 128GB was not enough, 495GB was not available

set -e

PROJECT_ROOT="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
MEM="256GB"
TIME="40:00:00"
CPUS="16"

echo "==========================================="
echo "Resubmitting Flow WM Jobs with 256GB Memory"
echo "Time: $(date)"
echo "==========================================="

mkdir -p $PROJECT_ROOT/logs/slurm/anymal

SEEDS=(42 123 456)

# =============================================
# Submit Flow K=4 Heun
# =============================================
echo ""
echo "=== Submitting Flow WM K=4 (Heun) ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowWM_K4_s${SEED}"
    RUN_NAME="Anymal_FlowWM_K4Heun_s${SEED}"
    
    echo "Submitting FlowWM K=4 seed=$SEED..."
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
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowWMK4Heuns${SEED}
EOF
done

# =============================================
# Submit Flow K=2 Heun
# =============================================
echo ""
echo "=== Submitting Flow WM K=2 (Heun) ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowWM_K2_s${SEED}"
    RUN_NAME="Anymal_FlowWM_K2Heun_s${SEED}"
    
    echo "Submitting FlowWM K=2 seed=$SEED..."
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
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowWMK2Heuns${SEED}
EOF
done

# =============================================
# Submit Flow K=8 Euler
# =============================================
echo ""
echo "=== Submitting Flow WM K=8 (Euler) ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowWM_K8_s${SEED}"
    RUN_NAME="Anymal_FlowWM_K8Euler_s${SEED}"
    
    echo "Submitting FlowWM K=8 Euler seed=$SEED..."
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
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowWMK8Eulers${SEED}
EOF
done

echo ""
echo "==========================================="
echo "All Flow WM jobs submitted with 256GB memory!"
echo "Total: 9 jobs (3 K=4 + 3 K=2 + 3 K=8)"
echo "Use 'squeue -u \$USER' to check status"
echo "==========================================="
