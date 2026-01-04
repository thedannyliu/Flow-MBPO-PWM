#!/bin/bash
# Submit aligned experiments: 6 configs × 10 seeds = 60 jobs
# All configs now have rew_rms: True for fair comparison

set -e

PROJECT_ROOT="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
MEM="400GB"
TIME="40:00:00"
CPUS="16"

echo "==========================================="
echo "Submitting ALIGNED Experiments (rew_rms: True)"
echo "6 configs × 10 seeds = 60 jobs"
echo "Time: $(date)"
echo "==========================================="

mkdir -p $PROJECT_ROOT/logs/slurm/aligned/{ant,anymal,humanoid}

# Seeds 0-9
SEEDS=(0 1 2 3 4 5 6 7 8 9)

JOB_COUNT=0

# =============================================
# ANT - Baseline (10 seeds)
# =============================================
echo ""
echo "=== Ant Baseline (10 seeds) ===" 
WANDB_PROJECT="flow-mbpo-aligned-ant"

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="ant_baseline_aligned_s${SEED}"
    RUN_NAME="Ant_Baseline_aligned_s${SEED}"
    
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
#SBATCH --output=logs/slurm/aligned/ant/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/aligned/ant/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src

cd scripts
python train_dflex.py \\
    env=dflex_ant \\
    alg=pwm_5M_baseline_final \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=${WANDB_PROJECT} \\
    ++wandb.name=${RUN_NAME}
EOF
    JOB_COUNT=$((JOB_COUNT + 1))
    sleep 0.2
done

# =============================================
# ANT - FlowWM K=8 aligned (10 seeds)
# =============================================
echo ""
echo "=== Ant FlowWM K=8 aligned (10 seeds) ===" 

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="ant_flowwm_k8_aligned_s${SEED}"
    RUN_NAME="Ant_FlowWM_K8_aligned_s${SEED}"
    
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
#SBATCH --output=logs/slurm/aligned/ant/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/aligned/ant/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src

cd scripts
python train_dflex.py \\
    env=dflex_ant \\
    alg=pwm_5M_flow_v3_aligned \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=${WANDB_PROJECT} \\
    ++wandb.name=${RUN_NAME}
EOF
    JOB_COUNT=$((JOB_COUNT + 1))
    sleep 0.2
done

# =============================================
# ANYMAL - Baseline (10 seeds)
# =============================================
echo ""
echo "=== Anymal Baseline (10 seeds) ===" 
WANDB_PROJECT="flow-mbpo-aligned-anymal"

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_baseline_aligned_s${SEED}"
    RUN_NAME="Anymal_Baseline_aligned_s${SEED}"
    
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
#SBATCH --output=logs/slurm/aligned/anymal/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/aligned/anymal/${JOB_NAME}_%j.err

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
    ++wandb.name=${RUN_NAME}
EOF
    JOB_COUNT=$((JOB_COUNT + 1))
    sleep 0.2
done

# =============================================
# ANYMAL - FlowPolicy aligned (10 seeds)
# =============================================
echo ""
echo "=== Anymal FlowPolicy aligned (10 seeds) ===" 

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="anymal_flowpolicy_aligned_s${SEED}"
    RUN_NAME="Anymal_FlowPolicy_aligned_s${SEED}"
    
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
#SBATCH --output=logs/slurm/aligned/anymal/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/aligned/anymal/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src

cd scripts
python train_dflex.py \\
    env=dflex_anymal \\
    alg=pwm_5M_flowpolicy_aligned \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=${WANDB_PROJECT} \\
    ++wandb.name=${RUN_NAME}
EOF
    JOB_COUNT=$((JOB_COUNT + 1))
    sleep 0.2
done

# =============================================
# HUMANOID - Baseline (10 seeds) - EXCLUSIVE
# =============================================
echo ""
echo "=== Humanoid Baseline (10 seeds) ===" 
WANDB_PROJECT="flow-mbpo-aligned-humanoid"

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="humanoid_baseline_aligned_s${SEED}"
    RUN_NAME="Humanoid_Baseline_aligned_s${SEED}"
    
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
#SBATCH --output=logs/slurm/aligned/humanoid/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/aligned/humanoid/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src
export CUDA_LAUNCH_BLOCKING=1

cd scripts
python train_dflex.py \\
    env=dflex_humanoid \\
    alg=pwm_5M_baseline_final \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=${WANDB_PROJECT} \\
    ++wandb.name=${RUN_NAME}
EOF
    JOB_COUNT=$((JOB_COUNT + 1))
    sleep 0.2
done

# =============================================
# HUMANOID - FlowPolicy aligned (10 seeds) - EXCLUSIVE
# =============================================
echo ""
echo "=== Humanoid FlowPolicy aligned (10 seeds) ===" 

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="humanoid_flowpolicy_aligned_s${SEED}"
    RUN_NAME="Humanoid_FlowPolicy_aligned_s${SEED}"
    
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
#SBATCH --output=logs/slurm/aligned/humanoid/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/aligned/humanoid/${JOB_NAME}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src
export CUDA_LAUNCH_BLOCKING=1

cd scripts
python train_dflex.py \\
    env=dflex_humanoid \\
    alg=pwm_5M_flowpolicy_aligned \\
    general.seed=${SEED} \\
    general.run_wandb=true \\
    ++wandb.project=${WANDB_PROJECT} \\
    ++wandb.name=${RUN_NAME}
EOF
    JOB_COUNT=$((JOB_COUNT + 1))
    sleep 0.2
done

echo ""
echo "==========================================="
echo "Submitted $JOB_COUNT aligned training jobs"
echo ""
echo "WandB Projects:"
echo "  - flow-mbpo-aligned-ant"
echo "  - flow-mbpo-aligned-anymal"
echo "  - flow-mbpo-aligned-humanoid"
echo ""
echo "Logs: logs/slurm/aligned/{ant,anymal,humanoid}/"
echo "Check status: squeue -u \$USER"
echo "==========================================="
