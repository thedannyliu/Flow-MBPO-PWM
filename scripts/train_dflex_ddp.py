"""
PyTorch Distributed Data Parallel (DDP) Training Script for PWM

This script enables multi-GPU training using PyTorch's DistributedDataParallel.
It splits batches across GPUs and synchronizes gradients for faster training.

Usage:
    # With torchrun (recommended)
    torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_dflex_ddp.py env=dflex_ant alg=pwm_5M_flow

    # With python -m torch.distributed.launch (legacy)
    python -m torch.distributed.launch --nproc_per_node=4 scripts/train_dflex_ddp.py env=dflex_ant alg=pwm_5M_flow
"""

# Fix torch.onnx import error by mocking the missing import
import sys
from unittest.mock import MagicMock

mock_exporter = MagicMock()
mock_exporter.DiagnosticOptions = MagicMock
mock_exporter.ExportOutput = MagicMock
sys.modules['torch.onnx._internal.exporter'] = mock_exporter

import os
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from pwm.utils import hydra_utils
from pwm.utils.common import seeding
from hydra.utils import instantiate

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)


def setup_ddp():
    """Initialize distributed training."""
    # Check if distributed environment variables are set
    if 'RANK' not in os.environ:
        print("Warning: Not running in distributed mode. Using single GPU.")
        return {'rank': 0, 'local_rank': 0, 'world_size': 1, 'is_distributed': False}
    
    # Get distributed parameters
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPU
        init_method='env://',
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    print(f"[Rank {rank}/{world_size}] Initialized DDP on GPU {local_rank}")
    
    return {
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'is_distributed': True
    }


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_wandb_run(wandb_cfg, job_config, run_id=None, rank=0):
    """Create WandB run (only on rank 0)."""
    if rank != 0:
        return None
    
    env_name = job_config["env"]["config"]["_target_"].split(".")[-1]
    try:
        alg_name = job_config["alg"]["_target_"].split(".")[-1]
    except:
        alg_name = job_config["alg"]["name"].upper()
    
    try:
        job_id = HydraConfig().get().job.num
        name = f"{alg_name}_{env_name}_sweep_{job_config['general']['seed']}"
        notes = wandb_cfg.get("notes", None)
    except:
        name = f"{alg_name}_{env_name}_ddp"
        notes = wandb_cfg.get("notes", "")
    
    return wandb.init(
        project=wandb_cfg.project,
        config=job_config,
        group=wandb_cfg.group if wandb_cfg.group else None,
        entity=wandb_cfg.entity if wandb_cfg.entity else None,
        name=name,
        notes=notes,
        id=run_id,
        resume=run_id is not None,
    )


@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    # Setup DDP
    ddp_info = setup_ddp()
    rank = ddp_info['rank']
    local_rank = ddp_info['local_rank']
    world_size = ddp_info['world_size']
    is_distributed = ddp_info['is_distributed']
    
    # Only rank 0 prints configuration
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Distributed Training Configuration")
        print(f"{'='*60}")
        print(f"World Size: {world_size}")
        print(f"Backend: {'NCCL' if is_distributed else 'Single GPU'}")
        print(f"{'='*60}\n")
    
    cfg_full = OmegaConf.to_container(cfg, resolve=True)

    # Only rank 0 creates WandB run
    if cfg.general.run_wandb and rank == 0:
        create_wandb_run(cfg.wandb, cfg_full)

    # Patch logdir
    logdir = HydraConfig.get()["runtime"]["output_dir"]
    logdir = os.path.join(logdir, cfg.general.logdir)

    # Seeding (different seed per rank for diversity)
    seed = cfg.general.seed + rank
    seeding(seed, False)
    
    if rank == 0:
        print(f"Seeds: {[cfg.general.seed + r for r in range(world_size)]}")

    # Configure environment
    if "SHAC" in cfg.alg._target_ or "AHAC" in cfg.alg._target_:
        cfg.env.config.no_grad = False
    else:
        cfg.env.config.no_grad = True
    
    if rank == 0:
        print(f"Running sim with no_grad={cfg.env.config.no_grad}")

    # Ensure CUDA device is set before creating environment
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(local_rank)
    
    # Clear CUDA cache to avoid memory issues
    torch.cuda.empty_cache()
    
    # Create environment (one per rank) with rank-specific logdir to avoid conflicts
    env_logdir = os.path.join(logdir, f"rank_{rank}")
    os.makedirs(env_logdir, exist_ok=True)
    env = instantiate(cfg.env.config, logdir=env_logdir)
    
    # Synchronize after environment creation
    if is_distributed:
        dist.barrier()
    
    if rank == 0:
        print("num_envs = ", env.num_envs)
        print("num_actions = ", env.num_actions)
        print("num_obs = ", env.num_obs)
        print(f"Total envs across all GPUs: {env.num_envs * world_size}")

    # Create agent on correct device
    agent = instantiate(
        cfg.alg,
        env=env,
        obs_dim=env.num_obs,
        act_dim=env.num_actions,
        logdir=logdir,
        log=cfg.general.run_wandb and rank == 0,  # Only rank 0 logs
        device=device,
    )
    
    # Wrap models with DDP
    if is_distributed:
        agent.actor = DDP(
            agent.actor,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        agent.critic = DDP(
            agent.critic,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        agent.wm = DDP(
            agent.wm,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        
        if rank == 0:
            print("\n✓ Models wrapped with DistributedDataParallel")

    # Load checkpoint if specified
    if cfg.general.checkpoint:
        agent.load(
            cfg.general.checkpoint,
            buffer=cfg.general.checkpoint_with_buffer,
            resume_training=cfg.general.resume_training
        )
        agent.wm_bootstrapped = True

    # Pretrain if specified
    if cfg.general.pretrain:
        actually_train = True if not cfg.general.checkpoint else False
        agent.pretrain_wm(
            cfg.general.pretrain, cfg.general.pretrain_steps, actually_train
        )

    # Train
    if cfg.general.train:
        agent.train()

    # Evaluate (only rank 0)
    if rank == 0:
        loss, discounted_loss, ep_len = agent.eval(cfg.general.eval_runs)
        print(
            f"mean episode loss = {loss:.2f}, mean discounted loss = {discounted_loss:.2f}, mean episode length = {ep_len:.2f}"
        )

    # Cleanup
    if cfg.general.run_wandb and rank == 0:
        wandb.finish()
    
    cleanup_ddp()


if __name__ == "__main__":
    train()
