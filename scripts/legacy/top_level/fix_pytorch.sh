#!/bin/bash
#SBATCH --job-name=fix_pytorch
#SBATCH --output=logs/slurm/fix_pytorch_%j.out
#SBATCH --error=logs/slurm/fix_pytorch_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Fix PyTorch CUDA installation
source ~/.bashrc
conda activate flow-mbpo

echo "=== Checking current PyTorch ==="
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "=== Installing PyTorch with CUDA ==="
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "=== Verifying CUDA support ==="
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "=== Fix Complete ==="
