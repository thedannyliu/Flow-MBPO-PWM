#!/bin/bash
#SBATCH --job-name=fix_dm_control
#SBATCH --output=logs/slurm/fix_dm_control_%j.out
#SBATCH --error=logs/slurm/fix_dm_control_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -A coc
#SBATCH --partition=ice-gpu

# Fix dm_control installation and test env creation
source ~/.bashrc
conda activate flow-mbpo

cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM

echo "=== Installing dm_control ==="
pip install dm_control mujoco

echo ""
echo "=== Testing dm_control import ==="
python -c "
import dm_control
from dm_control import suite
print('dm_control imported successfully')
print('Available domains:', list(suite.TASKS_BY_DOMAIN.keys())[:5])
"

echo ""
echo "=== Testing envs import ==="
export PYTHONPATH=/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/src:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/scripts:\$PYTHONPATH
python -c "
from envs import make_env
print('envs.make_env imported successfully')
"

echo ""
echo "=== Testing walker-stand creation ==="
python -c "
from dm_control import suite
env = suite.load('walker', 'stand')
print('walker-stand created successfully')
obs = env.reset()
print('Observation keys:', list(obs.observation.keys()))
"

echo ""
echo "=== Fix Complete ==="
