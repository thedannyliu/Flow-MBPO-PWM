#!/bin/bash
#SBATCH --job-name=install_dm
#SBATCH --output=logs/slurm/install_dm_%j.out
#SBATCH --error=logs/slurm/install_dm_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Install dm_control and all dependencies for MT30 experiments
source ~/.bashrc
conda activate flow-mbpo

echo "=== Installing dm_control dependencies ==="
pip install dm_control dm_env mujoco

echo ""
echo "=== Installing tensordict torchrl for PyTorch 2.6 compatibility ==="
pip install tensordict torchrl --upgrade

echo ""
echo "=== Verifying installations ==="
python -c "
import dm_control
from dm_control import suite
print('✓ dm_control installed successfully')
print(f'  Available domains: {list(suite.TASKS_BY_DOMAIN.keys())[:5]}...')

env = suite.load('walker', 'stand')
print('✓ walker-stand loaded successfully')

import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Installation Complete ==="
