#!/bin/bash
#SBATCH --job-name=install_deps
#SBATCH --output=logs/slurm/install_deps_%j.out
#SBATCH --error=logs/slurm/install_deps_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 00:20:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Install all missing dependencies for train_multitask.py
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo
export PATH=$CONDA_PREFIX/bin:$PATH

echo "=== Python: $(which python) ==="

echo ""
echo "=== Installing training dependencies ==="
pip install hydra-core omegaconf tensordict torchrl wandb termcolor tqdm

echo ""
echo "=== Installing flow_mbpo_pwm package ==="
cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM
pip install -e .

echo ""
echo "=== Verifying key imports ==="
python -c "
import hydra
print('✓ hydra OK')
import omegaconf
print('✓ omegaconf OK')
import tensordict
print('✓ tensordict OK')
import torchrl
print('✓ torchrl OK')
import wandb
print('✓ wandb OK')
from flow_mbpo_pwm.algorithms.pwm import PWM
print('✓ flow_mbpo_pwm.algorithms.pwm OK')
import torch
print(f'✓ PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')
"

echo ""
echo "=== Installation Complete ==="
