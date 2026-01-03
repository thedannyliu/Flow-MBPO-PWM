#!/bin/bash
#SBATCH --job-name=install_lxml
#SBATCH --output=logs/slurm/install_lxml_%j.out
#SBATCH --error=logs/slurm/install_lxml_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Install lxml and other dm_control dependencies
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo
export PATH=$CONDA_PREFIX/bin:$PATH

echo "Installing lxml..."
pip install lxml

echo ""
echo "Full dm_control test:"
python -c "
import dm_env
print('✓ dm_env OK')

from dm_control import suite
print('✓ dm_control suite OK')
print(f'  Domains: {list(suite.TASKS_BY_DOMAIN.keys())[:5]}')

env = suite.load('walker', 'stand')
print('✓ walker-stand loaded')
obs = env.reset()
print(f'✓ Reset OK, obs keys: {list(obs.observation.keys())}')

import torch
print(f'✓ PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')
"
