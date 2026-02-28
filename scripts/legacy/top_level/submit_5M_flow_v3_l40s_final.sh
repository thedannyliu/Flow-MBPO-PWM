#!/bin/bash
#SBATCH -J pwm_5M_flow_v3_l40s
#SBATCH -A gts-agarg35
#SBATCH --gres=gpu:L40S:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=128GB
#SBATCH -t 10:00:00
#SBATCH -o /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/logs/train_5M_flow_v3_l40s_%j.out
#SBATCH -e /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/logs/train_5M_flow_v3_l40s_%j.err

# Training 5M Flow V3 on L40s
# Config: flow_substeps=8, euler integrator (high-fidelity)
# Target: Match or exceed baseline (R ~ 1200)

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

source ~/.bashrc
conda activate pwm

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

python scripts/train_dflex.py \
    general.device=cuda:0 \
    general.seed=42 \
    alg=pwm_5M_flow_v3_substeps8_euler \
    env.env.num_envs=256 \
    env.env.episode_length=1000 \
    horizon=4 \
    wandb.group="5M_flow_l40s" \
    wandb.name="flow_v3_substeps8_euler_seed42" \
    wandb.project="flow-mbpo-pwm"

echo ""
echo "========================================="
echo "Flow V3 training completed at $(date)"
echo "Config: substeps=8, euler integrator"
echo "========================================="
