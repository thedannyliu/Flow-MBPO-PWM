#!/bin/bash
#SBATCH --job-name=install_dm4
#SBATCH --output=logs/slurm/install_dm4_%j.out
#SBATCH --error=logs/slurm/install_dm4_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Install dm_control using conda-forge (pre-built binaries, no bazel required)
source ~/.bashrc
conda activate flow-mbpo

echo "=== Current Python version ==="
python --version

echo ""
echo "=== Installing dm_control via conda-forge ==="
conda install -c conda-forge dm_control -y --no-deps || {
    echo "Conda install failed, trying pip with older version..."
    pip install dm_control==1.0.8
}

echo ""
echo "=== Checking pip list for dm_control ==="
pip list | grep -i dm

echo ""
echo "=== Verifying installations ==="
python -c "
try:
    import dm_control
    from dm_control import suite
    print('✓ dm_control installed successfully')
    print(f'  Version: {dm_control.__version__}')
    print(f'  Available domains: {list(suite.TASKS_BY_DOMAIN.keys())[:8]}')
    
    env = suite.load('walker', 'stand')
    print('✓ walker-stand loaded successfully')
    
    obs = env.reset()
    print(f'✓ Environment reset successful')
except Exception as e:
    print(f'✗ dm_control failed: {e}')
    import traceback
    traceback.print_exc()
    
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
"

echo ""
echo "=== Installation Complete ==="
