#!/bin/bash
#SBATCH --job-name=install_dm3
#SBATCH --output=logs/slurm/install_dm3_%j.out
#SBATCH --error=logs/slurm/install_dm3_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Install dm_control using pre-built wheel (no labmaze required)
source ~/.bashrc
conda activate flow-mbpo

echo "=== Installing mujoco ==="
pip install mujoco

echo ""
echo "=== Installing dm_control without optional dependencies ==="
# Install dm_control from pre-built binary (no labmaze)
pip install dm_control --ignore-requires dm_control[locomotion]

# If above fails, try:
if ! python -c "import dm_control" 2>/dev/null; then
    echo "Direct install failed, trying specific version..."
    pip install "dm_control>=1.0.0,<1.1.0"
fi

echo ""
echo "=== Installing dm_env ==="
pip install dm_env

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
    print(f'  Available domains: {list(suite.TASKS_BY_DOMAIN.keys())[:8]}')
    
    env = suite.load('walker', 'stand')
    print('✓ walker-stand loaded successfully')
    
    obs = env.reset()
    print(f'✓ Environment reset successful')
    print(f'  Observation keys: {list(obs.observation.keys())}')
except Exception as e:
    print(f'✗ dm_control failed: {e}')
    import traceback
    traceback.print_exc()
    
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Installation Complete ==="
