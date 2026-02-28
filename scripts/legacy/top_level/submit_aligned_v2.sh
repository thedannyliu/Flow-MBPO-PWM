#!/bin/bash
# Submit aligned experiments (v2) - with EXCLUSIVE node allocation
# All configs now have rew_rms: True for fair comparison
# Total: 60 jobs for ablation + 1 job for 48M + 2 jobs for K sweep = 63 jobs

set -e

PROJECT_ROOT="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
MEM="400GB"
TIME="40:00:00"
CPUS="16"

echo "================================================="
echo "Submitting ALIGNED Experiments v2 (Exclusive)"
echo "Time: $(date)"
echo "================================================="

mkdir -p $PROJECT_ROOT/logs/slurm/aligned_v2/{ant,anymal,humanoid}

# Seeds 0-9
SEEDS=(0 1 2 3 4 5 6 7 8 9)

JOB_COUNT=0

# Helper function to submit a job
submit_job() {
    local task=$1
    local variant=$2
    local config=$3
    local seed=$4
    local wandb_proj=$5
    local extra_args=$6
    local job_name="${task:0:3}_${variant:0:4}_s${seed}"
    
    # We use --exclusive to ensure dflex has the GPU to itself
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --exclusive
#SBATCH --output=logs/slurm/aligned_v2/${task}/${job_name}_%j.out
#SBATCH --error=logs/slurm/aligned_v2/${task}/${job_name}_%j.err

cd $PROJECT_ROOT
source ~/.bashrc
conda activate pwm
export PYTHONPATH=$PROJECT_ROOT/src
export CUDA_LAUNCH_BLOCKING=1

# Check GPU status
nvidia-smi

cd scripts
python train_dflex.py \\
    env=dflex_${task} \\
    alg=${config} \\
    general.seed=${seed} \\
    general.run_wandb=true \\
    ++wandb.project=${wandb_proj} \\
    ++wandb.name=${task}_${variant}_s${seed} \\
    ${extra_args}
EOF
    JOB_COUNT=$((JOB_COUNT + 1))
    sleep 1 # Sleep a bit more to avoid scheduler issues
}

# =================================================
# 1. Clean Ablation (10 seeds each)
# =================================================

# ANT
for SEED in "${SEEDS[@]}"; do
    submit_job "ant" "Baseline" "pwm_5M_baseline_final" ${SEED} "flow-mbpo-aligned-ant" ""
    submit_job "ant" "FlowWM_K8" "pwm_5M_flow_v3_aligned" ${SEED} "flow-mbpo-aligned-ant" ""
done

# ANYMAL
for SEED in "${SEEDS[@]}"; do
    submit_job "anymal" "Baseline" "pwm_5M_baseline_final" ${SEED} "flow-mbpo-aligned-anymal" ""
    submit_job "anymal" "FlowPolicy" "pwm_5M_flowpolicy_aligned" ${SEED} "flow-mbpo-aligned-anymal" ""
done

# HUMANOID
for SEED in "${SEEDS[@]}"; do
    submit_job "humanoid" "Baseline" "pwm_5M_baseline_final" ${SEED} "flow-mbpo-aligned-humanoid" ""
    submit_job "humanoid" "FlowPolicy" "pwm_5M_flowpolicy_aligned" ${SEED} "flow-mbpo-aligned-humanoid" ""
done

# =================================================
# 2. Large WM (48M)
# =================================================
echo "Submitting 48M Large WM..."
submit_job "ant" "FlowWM_48M" "pwm_48M_flow" 42 "flow-mbpo-scaling" "alg.flow_substeps=8"

# =================================================
# 3. Flow-specific Tuning (K substeps sweep on Ant)
# =================================================
echo "Submitting Flow Substeps sweep..."
submit_job "ant" "FlowWM_K16" "pwm_5M_flow_v3_aligned" 42 "flow-mbpo-tuning" "alg.flow_substeps=16"
submit_job "ant" "FlowWM_K32" "pwm_5M_flow_v3_aligned" 42 "flow-mbpo-tuning" "alg.flow_substeps=32"

echo ""
echo "================================================="
echo "Submitted $JOB_COUNT jobs with EXCLUSIVE flag"
echo "Check status: squeue -u \$USER"
echo "================================================="
