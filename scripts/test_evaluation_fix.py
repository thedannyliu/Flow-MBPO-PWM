#!/usr/bin/env python3
"""
Quick test of evaluation script fixes
"""

import sys
from pathlib import Path

# Add PWM to path
PWM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PWM_DIR / "src"))

import torch

# Test loading checkpoint
checkpoint_path = "outputs/2025-11-08/23-48-46/logs/pwm_5M_dflex_ant_seed42/best_policy.pt"

print("="*80)
print("Testing Checkpoint Loading")
print("="*80)
print(f"\nCheckpoint: {checkpoint_path}")

# Load and inspect checkpoint
ckpt = torch.load(checkpoint_path, map_location='cpu')
print(f"\nCheckpoint keys: {list(ckpt.keys())}")

for key in ckpt.keys():
    value = ckpt[key]
    if isinstance(value, dict):
        print(f"  {key}: dict with {len(value)} keys")
        if len(value) < 10:
            print(f"    Sub-keys: {list(value.keys())}")
    elif isinstance(value, torch.Tensor):
        print(f"  {key}: Tensor {value.shape}")
    elif isinstance(value, (int, float)):
        print(f"  {key}: {type(value).__name__} = {value}")
    else:
        print(f"  {key}: {type(value).__name__}")

print("\n" + "="*80)
print("Testing Environment Creation")
print("="*80)

try:
    from dflex.envs import AntEnv
    
    print("\nCreating AntEnv...")
    env = AntEnv(
        render=False,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        num_envs=1,
        stochastic_init=True,
        no_grad=True,
        episode_length=1000,
        MM_caching_frequency=1,
        early_termination=True,
        termination_height=0.27,
        action_penalty=0.0,
        joint_vel_obs_scaling=0.1,
        up_rew_scale=0.1,
    )
    
    print(f"✓ Environment created successfully")
    print(f"  Observation dim: {env.num_obs}")
    print(f"  Action dim: {env.num_actions}")
    
    # Test reset
    obs = env.reset()
    print(f"  Reset observation shape: {obs.shape}")
    
except Exception as e:
    print(f"✗ Environment creation failed:")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Testing Actor Loading")
print("="*80)

try:
    from pwm.models.actor import ActorStochasticMLP
    
    # Find actor state dict in checkpoint
    if 'actor_state_dict' in ckpt:
        actor_state = ckpt['actor_state_dict']
        print("\n✓ Found 'actor_state_dict' in checkpoint")
    elif 'actor' in ckpt:
        actor_state = ckpt['actor']
        print("\n✓ Found 'actor' in checkpoint")
    else:
        print(f"\n✗ Cannot find actor in checkpoint")
        print(f"Available keys: {list(ckpt.keys())}")
        sys.exit(1)
    
    print(f"  Actor state has {len(actor_state)} parameters")
    
    # Infer dimensions from state dict
    first_weight_key = [k for k in actor_state.keys() if 'weight' in k and 'fc1' in k][0]
    first_weight = actor_state[first_weight_key]
    print(f"  First layer weight shape: {first_weight.shape}")
    
    obs_dim = first_weight.shape[1]
    print(f"  Inferred obs_dim: {obs_dim}")
    
    # Find last layer to get action dim
    last_weight_key = [k for k in actor_state.keys() if 'mean' in k and 'weight' in k][-1]
    last_weight = actor_state[last_weight_key]
    action_dim = last_weight.shape[0]
    print(f"  Inferred action_dim: {action_dim}")
    
    # Create actor
    actor = ActorStochasticMLP(
        obs_dim=obs_dim,
        action_dim=action_dim,
        units=[400, 200, 100],
        activation='elu',
    )
    
    # Load weights
    actor.load_state_dict(actor_state)
    actor.eval()
    
    print(f"\n✓ Actor loaded successfully")
    print(f"  Total parameters: {sum(p.numel() for p in actor.parameters()):,}")
    
    # Test forward pass
    test_obs = torch.randn(1, obs_dim)
    with torch.no_grad():
        action_dist = actor(test_obs)
    print(f"  Test forward pass: OK")
    print(f"  Output type: {type(action_dist)}")
    
except Exception as e:
    print(f"\n✗ Actor loading failed:")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("All Tests Complete")
print("="*80)
