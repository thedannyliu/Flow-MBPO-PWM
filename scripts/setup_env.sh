#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=logs/slurm/setup_env_%j.out
#SBATCH --error=logs/slurm/setup_env_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Environment Setup Script
# Installs all required packages for Flow-MBPO in the flow-mbpo conda env

set -e  # Exit on error

source ~/.bashrc
conda activate flow-mbpo

cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM

echo "=== Environment Setup Script ==="
echo "Python: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Check current packages
echo "=== Checking current packages ==="
pip list | grep -i gym || echo "gym not installed"
pip list | grep -i metaworld || echo "metaworld not installed"
pip list | grep -i mujoco || echo "mujoco not installed"

# Install gym and gymnasium
echo ""
echo "=== Installing gym and gymnasium ==="
pip install gym==0.26.2 gymnasium

# Install metaworld for MT30 tasks
echo ""
echo "=== Installing metaworld ==="
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld

# Install this package in editable mode
echo ""
echo "=== Installing flow-mbpo-pwm package ==="
pip install -e .

# Install tdmpc2 external package
echo ""
echo "=== Installing external/tdmpc2 ==="
if [ -d "external/tdmpc2" ]; then
    pip install -e external/tdmpc2
else
    echo "Warning: external/tdmpc2 not found"
fi

# Install other missing dependencies
echo ""
echo "=== Installing other dependencies ==="
pip install wandb omegaconf hydra-core tensordict torchrl

# Final package check
echo ""
echo "=== Final package verification ==="
pip list | grep -E "gym|metaworld|mujoco|wandb|omegaconf"

# Test imports
echo ""
echo "=== Testing imports ==="
cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM
export PYTHONPATH=src:$PYTHONPATH
python -c "
import gym
print('gym OK')
import gymnasium
print('gymnasium OK')
try:
    import metaworld
    print('metaworld OK')
except Exception as e:
    print(f'metaworld failed: {e}')
import wandb
print('wandb OK')
import omegaconf
print('omegaconf OK')
from flow_mbpo_pwm.algorithms.pwm import PWM
print('flow_mbpo_pwm OK')
print('=== All imports successful ===')
"

echo ""
echo "=== Setup Complete ==="
