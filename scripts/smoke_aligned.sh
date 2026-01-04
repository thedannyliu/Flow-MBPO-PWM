#!/bin/bash
# Smoke test for aligned configs - 5 epochs only
# Validates configs before full 60-job submission

set -e

PROJECT_ROOT="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"

echo "==========================================="
echo "SMOKE TEST - Aligned Configs (5 epochs)"
echo "Time: $(date)"
echo "==========================================="

mkdir -p $PROJECT_ROOT/logs/slurm/smoke_aligned

# Test 1: Ant FlowWM K=8 aligned
echo ""
echo "=== TEST: Ant FlowWM K=8 aligned ==="
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=smoke_ant_flowwm_aligned
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/smoke_aligned/ant_flowwm_%j.out
#SBATCH --error=logs/slurm/smoke_aligned/ant_flowwm_%j.err

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
source ~/.bashrc
conda activate pwm
export PYTHONPATH=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/src

cd scripts
python train_dflex.py \
    env=dflex_ant \
    alg=pwm_5M_flow_v3_aligned \
    general.seed=42 \
    general.run_wandb=false \
    alg.max_epochs=5
EOF

# Test 2: Anymal FlowPolicy aligned
echo ""
echo "=== TEST: Anymal FlowPolicy aligned ==="
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=smoke_anymal_flowpolicy_aligned
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/smoke_aligned/anymal_flowpolicy_%j.out
#SBATCH --error=logs/slurm/smoke_aligned/anymal_flowpolicy_%j.err

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
source ~/.bashrc
conda activate pwm
export PYTHONPATH=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/src

cd scripts
python train_dflex.py \
    env=dflex_anymal \
    alg=pwm_5M_flowpolicy_aligned \
    general.seed=42 \
    general.run_wandb=false \
    alg.max_epochs=5
EOF

# Test 3: Humanoid FlowPolicy aligned (exclusive)
echo ""
echo "=== TEST: Humanoid FlowPolicy aligned ==="
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=smoke_humanoid_flowpolicy_aligned
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --output=logs/slurm/smoke_aligned/humanoid_flowpolicy_%j.out
#SBATCH --error=logs/slurm/smoke_aligned/humanoid_flowpolicy_%j.err

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
source ~/.bashrc
conda activate pwm
export PYTHONPATH=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/src
export CUDA_LAUNCH_BLOCKING=1

cd scripts
python train_dflex.py \
    env=dflex_humanoid \
    alg=pwm_5M_flowpolicy_aligned \
    general.seed=42 \
    general.run_wandb=false \
    alg.max_epochs=5
EOF

echo ""
echo "==========================================="
echo "Submitted 3 smoke tests"
echo "Check status: squeue -u \$USER"
echo "Logs: logs/slurm/smoke_aligned/"
echo "==========================================="
