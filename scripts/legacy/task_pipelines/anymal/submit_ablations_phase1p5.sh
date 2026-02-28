#!/bin/bash
# Submit Phase 1.5 additional ablation experiments
# Strong regularization + Full Flow H=8 + Higher LR variants

set -e

PROJECT_ROOT="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
MEM="400GB"
TIME="40:00:00"
CPUS="16"

echo "==========================================="
echo "Submitting Phase 1.5 Additional Ablations"
echo "Time: $(date)"
echo "==========================================="

mkdir -p $PROJECT_ROOT/logs/slurm/anymal

SEEDS=(42 123 456)

# =============================================
# Ablation: Flow WM with Strong Regularization (3 seeds)
# =============================================
echo ""
echo "=== Ablation: Flow WM StrongReg ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowWM_strongReg_s${SEED}"
    RUN_NAME="Anymal_FlowWM_K4_StrongReg_s${SEED}"
    
    echo "Submitting $RUN_NAME..."
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

cd scripts
python train_dflex.py \\
    env=dflex_anymal \\
    alg=pwm_5M_flow_strongReg \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowWMK4StrongRegs${SEED}
EOF
done

# =============================================
# Ablation: Full Flow with H=8 (3 seeds)
# =============================================
echo ""
echo "=== Ablation: Full Flow H=8 ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_fullflow_H8_s${SEED}"
    RUN_NAME="Anymal_FullFlow_H8_s${SEED}"
    
    echo "Submitting $RUN_NAME..."
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

cd scripts
python train_dflex.py \\
    env=dflex_anymal \\
    alg=pwm_5M_fullflow_H8 \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FullFlowH8s${SEED}
EOF
done

# =============================================
# Ablation: Flow WM K=4 with Higher LR (7e-4) (3 seeds)
# =============================================
echo ""
echo "=== Ablation: Flow WM HighLR ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowWM_highLR_s${SEED}"
    RUN_NAME="Anymal_FlowWM_K4_LR7e4_s${SEED}"
    
    echo "Submitting $RUN_NAME..."
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

cd scripts
python train_dflex.py \\
    env=dflex_anymal \\
    alg=pwm_5M_flow_v2_substeps4 \\
    ++alg.actor_lr=7e-4 \\
    ++alg.critic_lr=7e-4 \\
    ++alg.model_lr=4e-4 \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowWMK4HighLRs${SEED}
EOF
done

echo ""
echo "==========================================="
echo "All additional ablation experiments submitted!"
echo ""
echo "Summary:"
echo "  - Flow WM StrongReg: 3 jobs"
echo "  - Full Flow H=8: 3 jobs"
echo "  - Flow WM HighLR: 3 jobs"
echo "  - TOTAL: 9 jobs"
echo ""
echo "Use 'squeue -u \$USER' to check status"
echo "==========================================="
