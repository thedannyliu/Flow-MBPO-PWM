#!/bin/bash
#SBATCH -J eval_baseline_vs_flow
#SBATCH -A gts-agarg35
#SBATCH --gres=gpu:H200:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32GB
#SBATCH -t 2:00:00
#SBATCH -o /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/logs/eval_comparison_%j.out
#SBATCH -e /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/logs/eval_comparison_%j.err

# Evaluate and compare Baseline (Nov 8) vs Flow (Nov 9) checkpoints
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

# Activate conda environment
source ~/.bashrc
conda activate pwm

# Run evaluation with comparison
python scripts/evaluate_policy.py \
    --baseline outputs/2025-11-08/23-48-46/logs/pwm_5M_dflex_ant_seed42/best_policy.pt \
    --flow outputs/2025-11-09/06-18-53/logs/pwm_5M_flow_dflex_ant_seed42/best_policy.pt \
    --env dflex_ant \
    --num-episodes 100 \
    --device cuda \
    --seed 42 \
    --output logs/eval_results_comparison

echo ""
echo "========================================="
echo "Evaluation completed at $(date)"
echo "========================================="
