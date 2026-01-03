#!/bin/bash
#SBATCH --job-name=install_dm5
#SBATCH --output=logs/slurm/install_dm5_%j.out
#SBATCH --error=logs/slurm/install_dm5_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Install dm_env in flow-mbpo conda environment with fixed path
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo
export PATH=$CONDA_PREFIX/bin:$PATH

echo "=== Environment Info ==="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "pip: $(which pip)"
echo ""

echo "=== Installing dm_env and dependencies ==="
pip install dm_env mujoco

echo ""
echo "=== Verifying installations ==="
python -c "
import dm_env
print('✓ dm_env imported successfully')

import dm_control
from dm_control import suite
print('✓ dm_control imported successfully')
print(f'  Available domains: {list(suite.TASKS_BY_DOMAIN.keys())[:5]}')

env = suite.load('walker', 'stand')
print('✓ walker-stand loaded successfully')

obs = env.reset()
print(f'✓ Environment reset successful')
print(f'  Observation keys: {list(obs.observation.keys())}')

import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Installation Complete ==="
