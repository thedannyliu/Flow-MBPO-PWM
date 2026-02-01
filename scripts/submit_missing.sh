#!/bin/bash
# Submit Missing Training Experiments
# Anymal Baseline (3) + Humanoid FlowWM K=8 (1) + Humanoid FlowPolicy (3) + Humanoid FullFlow (3)
# Total: 10 training jobs

set -e

PROJECT_ROOT="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
MEM="400GB"
TIME="40:00:00"
CPUS="16"

echo "==========================================="
echo "Submitting MISSING Training Experiments"
echo "Time: $(date)"
echo "==========================================="

mkdir -p $PROJECT_ROOT/logs/slurm/anymal
mkdir -p $PROJECT_ROOT/logs/slurm/humanoid

SEEDS=(42 123 456)

# =============================================
# Anymal Baseline (MLP WM + MLP Policy) - 3 seeds
# =============================================
echo ""
echo "=== Anymal Baseline ===" 
WANDB_PROJECT="flow-mbpo-single"

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_baseline_s${SEED}"
    RUN_NAME="Anymal_Baseline_MLP_s${SEED}"
    
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
    alg=pwm_5M_baseline_final \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=${WANDB_PROJECT} \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=Baseline_Retrain_s${SEED}
EOF
done

# =============================================
# Humanoid FlowWM K=8 s456 (1 job)
# =============================================
WANDB_PROJECT="flow-mbpo-single-task-Humanoid"
echo ""
echo "=== Humanoid FlowWM K=8 s456 ==="
SEED=456
JOB_NAME="humanoid_flowWM_K8_s${SEED}"
RUN_NAME="Humanoid_FlowWM_K8Euler_s${SEED}"

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
#SBATCH --output=logs/slurm/humanoid/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/humanoid/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src

cd scripts
python train_dflex.py \\
    env=dflex_humanoid \\
    alg=pwm_5M_flow_v3_substeps8_euler \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=${WANDB_PROJECT} \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowWMK8Euler_Retrain_s${SEED}
EOF

# =============================================
# Humanoid FlowPolicy - 3 seeds
# =============================================
echo ""
echo "=== Humanoid FlowPolicy ===" 

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="humanoid_flowpolicy_s${SEED}"
    RUN_NAME="Humanoid_FlowPolicy_MLPWM_s${SEED}"
    
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
#SBATCH --output=logs/slurm/humanoid/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/humanoid/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src

cd scripts
python train_dflex.py \\
    env=dflex_humanoid \\
    alg=pwm_5M_flowpolicy \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=${WANDB_PROJECT} \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowPolicy_Retrain_s${SEED}
EOF
done

# =============================================
# Humanoid FullFlow - 3 seeds
# =============================================
echo ""
echo "=== Humanoid FullFlow ===" 

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="humanoid_fullflow_s${SEED}"
    RUN_NAME="Humanoid_FullFlow_FlowWM_FlowPol_s${SEED}"
    
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
#SBATCH --output=logs/slurm/humanoid/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/humanoid/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src

cd scripts
python train_dflex.py \\
    env=dflex_humanoid \\
    alg=pwm_5M_fullflow \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=${WANDB_PROJECT} \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FullFlow_Retrain_s${SEED}
EOF
done

echo ""
echo "==========================================="
echo "Submitted All Missing Experiments:"
echo ""
echo "  Anymal:"
echo "    - Baseline: 3 jobs (s42, s123, s456)"
echo ""
echo "  Humanoid:"
echo "    - FlowWM K=8: 1 job (s456)"
echo "    - FlowPolicy: 3 jobs (s42, s123, s456)"
echo "    - FullFlow: 3 jobs (s42, s123, s456)"
echo ""
echo "  TOTAL: 10 training jobs"
echo ""
echo "Check status: squeue -u \$USER"
echo "==========================================="
