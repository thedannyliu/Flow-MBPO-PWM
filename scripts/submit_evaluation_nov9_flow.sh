#!/bin/bash
#SBATCH -J eval_5M_flow_nov9
#SBATCH -A gts-agarg35
#SBATCH --gres=gpu:H200:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32GB
#SBATCH -t 2:00:00
#SBATCH -o /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/logs/eval_5M_flow_nov9_%j.out
#SBATCH -e /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/logs/eval_5M_flow_nov9_%j.err

# Evaluate 5M Flow checkpoint from Nov 9 (previously evaluated - R~900)
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

# Activate conda environment
source ~/.bashrc
conda activate pwm

# Run evaluation
python scripts/evaluate_policy.py \
    --flow outputs/2025-11-09/06-18-53/logs/pwm_5M_flow_dflex_ant_seed42/best_policy.pt \
    --env dflex.envs.AntEnv \
    --num-episodes 100 \
    --device cuda \
    --seed 42 \
    --output logs/eval_results_nov9_flow

echo "Evaluation completed at $(date)"
