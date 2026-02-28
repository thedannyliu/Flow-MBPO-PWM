#!/bin/bash
# Submit all Anymal training jobs for baseline and flow experiments
# This script submits 3 seeds x (1 baseline + 3 flow variants) = 12 jobs

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "==========================================="
echo "Submitting Anymal Training Jobs"
echo "Project: $PROJECT_ROOT"
echo "Time: $(date)"
echo "==========================================="

SEEDS=(42 123 456)

# Submit baseline jobs (3 seeds)
echo ""
echo "=== Submitting Baseline (MLP WM + MLP Policy) ==="
for SEED in "${SEEDS[@]}"; do
    echo "Submitting baseline seed=$SEED..."
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=anymal_baseline_s${SEED}
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=384GB
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/anymal/baseline_s${SEED}_%j.out
#SBATCH --error=logs/slurm/anymal/baseline_s${SEED}_%j.err

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
    ++wandb.name=anymal_MLPWM_MLPpol_s${SEED} \\
    ++wandb.notes=AnymalBaselineMLPWMseed${SEED}
EOF
done

# Submit Flow WM jobs with K=4 (default, recommended) - 3 seeds
echo ""
echo "=== Submitting Flow WM K=4 (Heun) ==="
for SEED in "${SEEDS[@]}"; do
    echo "Submitting FlowWM K=4 seed=$SEED..."
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=anymal_flowWM_K4_s${SEED}
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=384GB
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/anymal/flowWM_K4_s${SEED}_%j.out
#SBATCH --error=logs/slurm/anymal/flowWM_K4_s${SEED}_%j.err

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
    ++wandb.name=anymal_FlowWM_K4_s${SEED} \\
    ++wandb.notes=AnymalFlowWMK4Heunseed${SEED}
EOF
done

# Submit Flow WM jobs with K=2 (faster) - 3 seeds
echo ""
echo "=== Submitting Flow WM K=2 (Heun) ==="
for SEED in "${SEEDS[@]}"; do
    echo "Submitting FlowWM K=2 seed=$SEED..."
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=anymal_flowWM_K2_s${SEED}
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=384GB
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/anymal/flowWM_K2_s${SEED}_%j.out
#SBATCH --error=logs/slurm/anymal/flowWM_K2_s${SEED}_%j.err

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
    ++wandb.name=anymal_FlowWM_K2_s${SEED} \\
    ++wandb.notes=AnymalFlowWMK2Heunseed${SEED}
EOF
done

# Submit Flow WM jobs with K=8 Euler - 3 seeds
echo ""
echo "=== Submitting Flow WM K=8 (Euler) ==="
for SEED in "${SEEDS[@]}"; do
    echo "Submitting FlowWM K=8 Euler seed=$SEED..."
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=anymal_flowWM_K8_s${SEED}
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=384GB
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/anymal/flowWM_K8_s${SEED}_%j.out
#SBATCH --error=logs/slurm/anymal/flowWM_K8_s${SEED}_%j.err

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
    ++wandb.name=anymal_FlowWM_K8Euler_s${SEED} \\
    ++wandb.notes=AnymalFlowWMK8Eulerseed${SEED}
EOF
done

echo ""
echo "==========================================="
echo "All jobs submitted!"
echo "Use 'squeue -u \$USER' to check status"
echo "==========================================="
