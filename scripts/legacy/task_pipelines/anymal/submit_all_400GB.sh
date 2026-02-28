#!/bin/bash
# Submit ALL Anymal experiments with 400GB memory
# Flow WM (9 jobs) + Flow Policy (3 jobs) + Full Flow (3 jobs) = 15 jobs

set -e

PROJECT_ROOT="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
MEM="400GB"
TIME="40:00:00"
CPUS="16"

echo "==========================================="
echo "Submitting ALL Anymal Experiments (400GB)"
echo "Time: $(date)"
echo "==========================================="

mkdir -p $PROJECT_ROOT/logs/slurm/anymal

SEEDS=(42 123 456)

# =============================================
# Flow WM K=4 Heun (3 seeds)
# =============================================
echo ""
echo "=== Flow WM K=4 Heun ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowWM_K4_s${SEED}"
    RUN_NAME="Anymal_FlowWM_K4Heun_s${SEED}"
    
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
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowWMK4Heuns${SEED}
EOF
done

# =============================================
# Flow WM K=2 Heun (3 seeds)
# =============================================
echo ""
echo "=== Flow WM K=2 Heun ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowWM_K2_s${SEED}"
    RUN_NAME="Anymal_FlowWM_K2Heun_s${SEED}"
    
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
    alg=pwm_5M_flow_v1_substeps2 \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowWMK2Heuns${SEED}
EOF
done

# =============================================
# Flow WM K=8 Euler (3 seeds)
# =============================================
echo ""
echo "=== Flow WM K=8 Euler ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowWM_K8_s${SEED}"
    RUN_NAME="Anymal_FlowWM_K8Euler_s${SEED}"
    
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
    alg=pwm_5M_flow_v3_substeps8_euler \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowWMK8Eulers${SEED}
EOF
done

# =============================================
# MLP WM + Flow Policy (3 seeds)
# =============================================
echo ""
echo "=== MLP WM + Flow Policy ==="
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowpolicy_s${SEED}"
    RUN_NAME="Anymal_FlowPolicy_MLPWM_s${SEED}"
    
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
    alg=pwm_5M_flowpolicy \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=flow-mbpo-single \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowPolicyMLPWMs${SEED}
EOF
done

# =============================================
# Full Flow: Flow WM + Flow Policy (3 seeds)
# =============================================
echo ""
echo "=== Full Flow (Flow WM + Flow Policy) ==="
for SEED in "${SEEDS[@]}"; do
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
echo "All jobs submitted with 400GB memory!"
echo ""
echo "Summary:"
echo "  - Flow WM K=4 Heun: 3 jobs"
echo "  - Flow WM K=2 Heun: 3 jobs"  
echo "  - Flow WM K=8 Euler: 3 jobs"
echo "  - MLP WM + Flow Policy: 3 jobs"
echo "  - Full Flow: 3 jobs"
echo "  - TOTAL: 15 jobs"
echo ""
echo "Use 'squeue -u \$USER' to check status"
echo "==========================================="
