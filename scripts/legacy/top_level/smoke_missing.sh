#!/bin/bash
# Smoke Test Script - Run minimal 5-epoch tests before full training
# Purpose: Validate configs work before submitting long jobs

set -e

PROJECT_ROOT="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
MEM="400GB"
TIME="01:00:00"  # Short time for smoke test
CPUS="16"
MAX_EPOCHS=5  # Smoke test epochs

echo "==========================================="
echo "SMOKE TESTS - Missing Experiments"
echo "Time: $(date)"
echo "==========================================="

mkdir -p $PROJECT_ROOT/logs/slurm/smoke

# Test 1: Anymal Baseline s42
echo ""
echo "=== TEST 1: Anymal Baseline s42 ==="
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=smoke_anymal_baseline
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/smoke/anymal_baseline_%j.out
#SBATCH --error=logs/slurm/smoke/anymal_baseline_%j.err

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
source ~/.bashrc
conda activate pwm
export PYTHONPATH=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/src

cd scripts
python train_dflex.py \
    env=dflex_anymal \
    alg=pwm_5M_baseline_final \
    general.seed=42 \
    general.run_wandb=false \
    alg.max_epochs=5
EOF

# Test 2: Humanoid FlowPolicy s42
echo ""
echo "=== TEST 2: Humanoid FlowPolicy s42 ==="
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=smoke_humanoid_flowpolicy
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/smoke/humanoid_flowpolicy_%j.out
#SBATCH --error=logs/slurm/smoke/humanoid_flowpolicy_%j.err

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
source ~/.bashrc
conda activate pwm
export PYTHONPATH=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/src

cd scripts
python train_dflex.py \
    env=dflex_humanoid \
    alg=pwm_5M_flowpolicy \
    general.seed=42 \
    general.run_wandb=false \
    alg.max_epochs=5
EOF

# Test 3: Humanoid FullFlow s42
echo ""
echo "=== TEST 3: Humanoid FullFlow s42 ==="
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=smoke_humanoid_fullflow
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/smoke/humanoid_fullflow_%j.out
#SBATCH --error=logs/slurm/smoke/humanoid_fullflow_%j.err

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
source ~/.bashrc
conda activate pwm
export PYTHONPATH=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/src

cd scripts
python train_dflex.py \
    env=dflex_humanoid \
    alg=pwm_5M_fullflow \
    general.seed=42 \
    general.run_wandb=false \
    alg.max_epochs=5
EOF

echo ""
echo "==========================================="
echo "Submitted 3 smoke test jobs"
echo "Check status with: squeue -u \$USER"
echo "Check logs in: logs/slurm/smoke/"
echo "==========================================="
