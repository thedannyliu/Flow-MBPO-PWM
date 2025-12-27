#!/bin/bash
#SBATCH --job-name=test_ddp
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=inferno
#SBATCH --gres=gpu:l40s:4
#SBATCH --exclude=atl1-1-03-004-31-0,atl1-1-03-007-29-0,atl1-1-03-007-31-0  # Exclude nodes with potential ECC issues
#SBATCH --output=logs/slurm/test_ddp_%j.out
#SBATCH --error=logs/slurm/test_ddp_%j.err

set -x  # Enable debug output

echo "=========================================="
echo "L40s DDP Hardware Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

echo ""
echo "Available GPUs:"
nvidia-smi --list-gpus

echo ""
echo "Detailed GPU Info:"
nvidia-smi

# Load environment
source ~/.bashrc
conda activate pwm || source activate pwm

# Test simple DDP initialization
cat > /tmp/test_ddp_$$.py <<'EOF'
import os
import torch
import torch.distributed as dist

def test_ddp():
    # Get distributed parameters
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    print(f"[Rank {rank}/{world_size}] Starting on GPU {local_rank}")
    
    # Initialize process group
    dist.init_process_group(backend='nccl', init_method='env://')
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    print(f"[Rank {rank}] Device set to: {device}")
    print(f"[Rank {rank}] CUDA available: {torch.cuda.is_available()}")
    print(f"[Rank {rank}] Current device: {torch.cuda.current_device()}")
    
    # Test tensor creation
    try:
        x = torch.randn(100, 100, device=device)
        print(f"[Rank {rank}] Successfully created tensor on GPU")
        
        # Test all_reduce
        dist.all_reduce(x.sum())
        print(f"[Rank {rank}] Successfully performed all_reduce")
        
        # Test tensor operations
        y = torch.mm(x, x)
        print(f"[Rank {rank}] Successfully performed matrix multiplication")
        
        print(f"[Rank {rank}] ✓ All tests passed!")
        
    except Exception as e:
        print(f"[Rank {rank}] ✗ Error: {e}")
        raise
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    test_ddp()
EOF

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=4

echo ""
echo "=========================================="
echo "Running DDP Test on 4 GPUs"
echo "=========================================="

torchrun --standalone --nnodes=1 --nproc_per_node=4 /tmp/test_ddp_$$.py

EXIT_CODE=$?
rm /tmp/test_ddp_$$.py

echo ""
echo "=========================================="
echo "Test Result"
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: All 4 L40s GPUs working correctly!"
else
    echo "❌ FAILED: Exit code $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
