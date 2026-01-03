#!/bin/bash
#SBATCH --job-name=fix_env
#SBATCH --output=logs/slurm/fix_env_%j.out
#SBATCH --error=logs/slurm/fix_env_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Quick fix for missing packages
source ~/.bashrc
conda activate flow-mbpo
pip install ipython

echo "=== Package Fix Complete ==="
