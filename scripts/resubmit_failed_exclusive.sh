#!/bin/bash
# Resubmit failed training jobs with exclusive node allocation
# This should fix the CUDA busy errors

set -e

PROJECT_ROOT="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
MEM="400GB"
TIME="40:00:00"
CPUS="16"

echo "==========================================="
echo "Resubmitting FAILED Jobs with --exclusive"
echo "Time: $(date)"
echo "==========================================="

mkdir -p $PROJECT_ROOT/logs/slurm/anymal
mkdir -p $PROJECT_ROOT/logs/slurm/humanoid

# =============================================
# Anymal Baseline s456 (1 job)
# =============================================
echo ""
echo "=== Anymal Baseline s456 ==="
WANDB_PROJECT="flow-mbpo-single"
SEED=456
JOB_NAME="anymal_baseline_s${SEED}"
RUN_NAME="Anymal_Baseline_MLP_s${SEED}"

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
#SBATCH --exclusive
#SBATCH --output=logs/slurm/anymal/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/anymal/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src
export CUDA_LAUNCH_BLOCKING=1

cd scripts
python train_dflex.py \\
    env=dflex_anymal \\
    alg=pwm_5M_baseline_final \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=${WANDB_PROJECT} \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=Baseline_Retrain_s${SEED}_exclusive
EOF

# =============================================
# Humanoid FlowWM K=8 s456
# =============================================
WANDB_PROJECT="flow-mbpo-single-task-Humanoid"
echo ""
echo "=== Humanoid FlowWM K=8 s456 ==="
SEED=456
JOB_NAME="humanoid_flowWM_K8_s${SEED}"
RUN_NAME="Humanoid_FlowWM_K8Euler_s${SEED}"

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
#SBATCH --exclusive
#SBATCH --output=logs/slurm/humanoid/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/humanoid/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src
export CUDA_LAUNCH_BLOCKING=1

cd scripts
python train_dflex.py \\
    env=dflex_humanoid \\
    alg=pwm_5M_flow_v3_substeps8_euler \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=${WANDB_PROJECT} \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowWMK8_exclusive_s${SEED}
EOF

# =============================================
# Humanoid FlowPolicy - 3 seeds
# =============================================
echo ""
echo "=== Humanoid FlowPolicy ===" 

for SEED in 42 123 456; do
    JOB_NAME="humanoid_flowpolicy_s${SEED}"
    RUN_NAME="Humanoid_FlowPolicy_MLPWM_s${SEED}"
    
    echo "Submitting $RUN_NAME with --exclusive..."
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
#SBATCH --exclusive
#SBATCH --output=logs/slurm/humanoid/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/humanoid/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src
export CUDA_LAUNCH_BLOCKING=1

cd scripts
python train_dflex.py \\
    env=dflex_humanoid \\
    alg=pwm_5M_flowpolicy \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=${WANDB_PROJECT} \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FlowPolicy_exclusive_s${SEED}
EOF
    sleep 1
done

# =============================================
# Humanoid FullFlow - 3 seeds
# =============================================
echo ""
echo "=== Humanoid FullFlow ===" 

for SEED in 42 123 456; do
    JOB_NAME="humanoid_fullflow_s${SEED}"
    RUN_NAME="Humanoid_FullFlow_FlowWM_FlowPol_s${SEED}"
    
    echo "Submitting $RUN_NAME with --exclusive..."
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
#SBATCH --exclusive
#SBATCH --output=logs/slurm/humanoid/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/humanoid/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src
export CUDA_LAUNCH_BLOCKING=1

cd scripts
python train_dflex.py \\
    env=dflex_humanoid \\
    alg=pwm_5M_fullflow \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=${WANDB_PROJECT} \\
    ++wandb.name=${RUN_NAME} \\
    ++wandb.notes=FullFlow_exclusive_s${SEED}
EOF
    sleep 1
done

echo ""
echo "==========================================="
echo "Resubmitted 8 failed jobs with --exclusive flag"
echo "  - Anymal Baseline s456: 1 job"
echo "  - Humanoid FlowWM K=8 s456: 1 job"
echo "  - Humanoid FlowPolicy: 3 jobs"
echo "  - Humanoid FullFlow: 3 jobs"
echo ""
echo "Also added CUDA_LAUNCH_BLOCKING=1 for debugging"
echo "Check status: squeue -u \$USER"
echo "==========================================="
