#!/bin/bash
# Submit all Flow Policy training jobs with 400GB memory
# MLP WM + Flow Policy: 3 seeds
# Flow WM + Flow Policy (Full Flow): 3 seeds

set -e

PROJECT_ROOT="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
MEM="400GB"
TIME="40:00:00"
CPUS="16"

echo "==========================================="
echo "Submitting Flow Policy Training Jobs (400GB)"
echo "Time: $(date)"
echo "==========================================="

mkdir -p $PROJECT_ROOT/logs/slurm/anymal

SEEDS=(42 123 456)

# =============================================
# Submit MLP WM + Flow Policy (3 seeds)
# =============================================
echo ""
echo "=== Submitting MLP WM + Flow Policy ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowpolicy_s${SEED}"
    RUN_NAME="Anymal_FlowPolicy_MLPWM_s${SEED}"
    
    echo "Submitting Flow Policy seed=$SEED..."
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
    alg=pwm_5M_flowpolicy \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowPolicyMLPWMs${SEED}
EOF
done

# =============================================
# Submit Flow WM + Flow Policy / Full Flow (3 seeds)
# =============================================
echo ""
echo "=== Submitting Full Flow (Flow WM + Flow Policy) ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_fullflow_s${SEED}"
    RUN_NAME="Anymal_FullFlow_FlowWM_FlowPol_s${SEED}"
    
    echo "Submitting Full Flow seed=$SEED..."
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
    alg=pwm_5M_fullflow \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FullFlowFlowWMFlowPols${SEED}
EOF
done

echo ""
echo "==========================================="
echo "All Flow Policy jobs submitted with 400GB!"
echo "Total: 6 jobs (3 Flow Policy + 3 Full Flow)"
echo "Use 'squeue -u \$USER' to check status"
echo "==========================================="
