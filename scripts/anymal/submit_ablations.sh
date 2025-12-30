#!/bin/bash
# Submit hyperparameter ablation experiments
# Each ablation has 3 seeds (42, 123, 456)
# Based on master_plan.md Phase 1.5 ablation strategy

set -e

PROJECT_ROOT="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
MEM="400GB"
TIME="40:00:00"
CPUS="16"

echo "==========================================="
echo "Submitting Hyperparameter Ablation Experiments"
echo "Time: $(date)"
echo "==========================================="

mkdir -p $PROJECT_ROOT/logs/slurm/anymal

SEEDS=(42 123 456)

# =============================================
# Ablation 1: Flow WM with H=8 (3 seeds)
# Tests shorter horizon vs default H=16
# =============================================
echo ""
echo "=== Ablation: Flow WM H=8 ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowWM_H8_s${SEED}"
    RUN_NAME="Anymal_FlowWM_K4_H8_s${SEED}"
    
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
    alg=pwm_5M_flow_H8 \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowWMK4H8s${SEED}
EOF
done

# =============================================
# Ablation 2: Flow WM with LR=3e-4 (3 seeds)
# Tests lower learning rate vs default 5e-4
# =============================================
echo ""
echo "=== Ablation: Flow WM LowLR ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowWM_lowLR_s${SEED}"
    RUN_NAME="Anymal_FlowWM_K4_LR3e4_s${SEED}"
    
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
    alg=pwm_5M_flow_lowLR \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowWMK4LowLRs${SEED}
EOF
done

# =============================================
# Ablation 3: Flow Policy with H=8 (3 seeds)
# Tests shorter horizon for Flow Policy
# =============================================
echo ""
echo "=== Ablation: Flow Policy H=8 ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowpol_H8_s${SEED}"
    RUN_NAME="Anymal_FlowPolicy_H8_s${SEED}"
    
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
    alg=pwm_5M_flowpolicy_H8 \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowPolicyH8s${SEED}
EOF
done

# =============================================
# Resubmit Failed: Full Flow s123 and s456
# =============================================
echo ""
echo "=== Resubmit: Full Flow s123, s456 ==="
for SEED in 123 456; do
    JOB_NAME="anymal_fullflow_s${SEED}"
    RUN_NAME="Anymal_FullFlow_FlowWM_FlowPol_s${SEED}"
    
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
echo "All ablation experiments submitted!"
echo ""
echo "Summary:"
echo "  - Flow WM H=8: 3 jobs"
echo "  - Flow WM LowLR: 3 jobs"
echo "  - Flow Policy H=8: 3 jobs"
echo "  - Full Flow s123, s456: 2 jobs"
echo "  - TOTAL: 11 jobs"
echo ""
echo "Use 'squeue -u \$USER' to check status"
echo "==========================================="
