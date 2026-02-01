#!/bin/bash
#SBATCH --job-name=install_pd
#SBATCH --output=logs/slurm/install_pd_%j.out
#SBATCH --error=logs/slurm/install_pd_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Install pandas and remaining dependencies
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo
export PATH=$CONDA_PREFIX/bin:$PATH

pip install pandas scipy matplotlib seaborn

echo "Done"
