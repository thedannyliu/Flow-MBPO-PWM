#!/bin/bash
#SBATCH --job-name=test_buffer_l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=inferno
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=atl1-1-03-004-31-0
#SBATCH --output=logs/slurm/test_buffer_l40s_%j.out
#SBATCH --error=logs/slurm/test_buffer_l40s_%j.err

echo "=========================================="
echo "L40s Buffer Filling Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

# Load conda environment
source ~/.bashrc
conda activate pwm

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16

# Run test
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM
python scripts/test_buffer_filling.py

echo "=========================================="
echo "Test completed at $(date)"
echo "=========================================="
