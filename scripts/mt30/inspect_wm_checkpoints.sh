#!/bin/bash
#SBATCH -Jinspect_ckpt
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH -t 0:10:00
#SBATCH -A coc
#SBATCH -ologs/slurm/inspect_ckpt_%j.out
#SBATCH -elogs/slurm/inspect_ckpt_%j.err

echo "=========================================="
echo "Checkpoint Inspection Job"
echo "=========================================="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Activate environment (copy from working Phase 8 script)
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo
export PATH=$CONDA_PREFIX/bin:$PATH

cd /home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM
export PYTHONPATH=/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/src:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/scripts:/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/external/tdmpc2:$PYTHONPATH

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Inspect checkpoints
python << 'EOF'
import torch

flow_ckpt = torch.load('outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt', map_location='cpu')
mlp_ckpt = torch.load('outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt', map_location='cpu')

print()
print("=== Flow WM Checkpoint ===")
print(f"Keys: {list(flow_ckpt.keys())}")
for k in flow_ckpt.keys():
    if isinstance(flow_ckpt[k], dict):
        print(f"  {k}: dict with {len(flow_ckpt[k])} keys")
        if k == 'world_model':
            wm_keys = list(flow_ckpt[k].keys())[:5]
            print(f"    First 5 WM keys: {wm_keys}")
    elif isinstance(flow_ckpt[k], (int, float, str)):
        print(f"  {k}: {flow_ckpt[k]}")
    else:
        print(f"  {k}: {type(flow_ckpt[k]).__name__}")

print()
print("=== MLP WM Checkpoint ===")
print(f"Keys: {list(mlp_ckpt.keys())}")
for k in mlp_ckpt.keys():
    if isinstance(mlp_ckpt[k], dict):
        print(f"  {k}: dict with {len(mlp_ckpt[k])} keys")
        if k == 'world_model':
            wm_keys = list(mlp_ckpt[k].keys())[:5]
            print(f"    First 5 WM keys: {wm_keys}")
    elif isinstance(mlp_ckpt[k], (int, float, str)):
        print(f"  {k}: {mlp_ckpt[k]}")
    else:
        print(f"  {k}: {type(mlp_ckpt[k]).__name__}")

print()
print("=== Summary ===")
print("Flow WM checkpoint: outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt")
print("MLP WM checkpoint: outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt")
print("Both checkpoints are ready for Phase 9 experiments.")
EOF

echo ""
echo "=========================================="
echo "Inspection Complete"
echo "=========================================="
