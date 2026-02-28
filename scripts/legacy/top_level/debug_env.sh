#!/bin/bash
#SBATCH --job-name=debug_env
#SBATCH --output=logs/slurm/debug_env_%j.out
#SBATCH --error=logs/slurm/debug_env_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Debug the conda environment issue
source ~/.bashrc
conda activate flow-mbpo

echo "=== Environment Info ==="
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "PATH: $PATH"
echo ""

echo "=== Which python/pip ==="
which python
which pip
python --version
pip --version
echo ""

echo "=== Conda site-packages ==="
ls $CONDA_PREFIX/lib/python*/site-packages/ | grep dm || echo "No dm packages in conda site-packages"
echo ""

echo "=== Check where pip installs to ==="
pip show mujoco 2>/dev/null | head -5
pip show tensordict 2>/dev/null | head -5
echo ""

echo "=== sys.path ==="
python -c "import sys; print('\n'.join(sys.path))"
echo ""

echo "=== List all dm* packages in site-packages ==="
find /storage/ice1/2/9/eliu354/miniconda3/lib/python*/site-packages -name "dm*" -type d 2>/dev/null | head -10
python -c "import site; print('Site packages:', site.getsitepackages())"
echo ""

echo "=== Try importing dm_control ==="
python -c "
import sys
print('Python executable:', sys.executable)
print('sys.path:', sys.path[:5])

try:
    import dm_control
    print('dm_control imported successfully')
except ImportError as e:
    print(f'Import failed: {e}')
"
