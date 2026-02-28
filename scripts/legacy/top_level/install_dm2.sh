#!/bin/bash
#SBATCH --job-name=install_dm2
#SBATCH --output=logs/slurm/install_dm2_%j.out
#SBATCH --error=logs/slurm/install_dm2_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Install dm_control without labmaze (not needed for MT30/MT80 tasks)
source ~/.bashrc
conda activate flow-mbpo

echo "=== Installing dm_control direct download ==="
# Install mujoco first
pip install mujoco

# Install dm_control directly without building from source
pip install dm_control --no-build-isolation

echo ""
echo "=== Installing dm_env ==="
pip install dm_env

echo ""
echo "=== Verifying installations ==="
python -c "
try:
    import dm_control
    from dm_control import suite
    print('✓ dm_control installed successfully')
    print(f'  Available domains: {list(suite.TASKS_BY_DOMAIN.keys())[:5]}...')
    
    env = suite.load('walker', 'stand')
    print('✓ walker-stand loaded successfully')
except Exception as e:
    print(f'✗ dm_control failed: {e}')
    
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Installation Complete ==="
