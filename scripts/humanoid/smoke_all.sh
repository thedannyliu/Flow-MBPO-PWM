#!/bin/bash
# Smoke tests for Humanoid experiments - verify configs work before full training

set -e

PROJECT_ROOT="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
MEM="400GB"
TIME="00:30:00"
CPUS="16"

echo "==========================================="
echo "Submitting Humanoid Smoke Tests"
echo "Time: $(date)"
echo "==========================================="

mkdir -p $PROJECT_ROOT/logs/slurm/humanoid

# Baseline smoke test
echo "Submitting Humanoid Baseline smoke test..."
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=humanoid_smoke_baseline
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/humanoid/smoke_baseline_%j.out
#SBATCH --error=logs/slurm/humanoid/smoke_baseline_%j.err

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
source ~/.bashrc
conda activate pwm
export PYTHONPATH=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/src

cd scripts
python train_dflex.py \
    env=dflex_humanoid \
    alg=pwm_5M_baseline_final \
    general.seed=42 \
    alg.max_epochs=5 \
    general.run_wandb=false

echo "Smoke test baseline completed!"
EOF

# Flow WM smoke test
echo "Submitting Humanoid Flow WM smoke test..."
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=humanoid_smoke_flowWM
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/humanoid/smoke_flowWM_%j.out
#SBATCH --error=logs/slurm/humanoid/smoke_flowWM_%j.err

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
source ~/.bashrc
conda activate pwm
export PYTHONPATH=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/src

cd scripts
python train_dflex.py \
    env=dflex_humanoid \
    alg=pwm_5M_flow_v2_substeps4 \
    general.seed=42 \
    alg.max_epochs=5 \
    general.run_wandb=false

echo "Smoke test flow WM completed!"
EOF

# Flow Policy smoke test
echo "Submitting Humanoid Flow Policy smoke test..."
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=humanoid_smoke_flowpolicy
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/humanoid/smoke_flowpolicy_%j.out
#SBATCH --error=logs/slurm/humanoid/smoke_flowpolicy_%j.err

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
source ~/.bashrc
conda activate pwm
export PYTHONPATH=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/src

cd scripts
python train_dflex.py \
    env=dflex_humanoid \
    alg=pwm_5M_flowpolicy \
    general.seed=42 \
    alg.max_epochs=5 \
    general.run_wandb=false

echo "Smoke test flow policy completed!"
EOF

# Full Flow smoke test
echo "Submitting Humanoid Full Flow smoke test..."
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=humanoid_smoke_fullflow
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/humanoid/smoke_fullflow_%j.out
#SBATCH --error=logs/slurm/humanoid/smoke_fullflow_%j.err

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM
source ~/.bashrc
conda activate pwm
export PYTHONPATH=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/src

cd scripts
python train_dflex.py \
    env=dflex_humanoid \
    alg=pwm_5M_fullflow \
    general.seed=42 \
    alg.max_epochs=5 \
    general.run_wandb=false

echo "Smoke test full flow completed!"
EOF

echo ""
echo "==========================================="
echo "Humanoid smoke tests submitted!"
echo "==========================================="
