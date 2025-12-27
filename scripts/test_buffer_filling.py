#!/usr/bin/env python3
"""
Test the exact initialization sequence that PWM training uses.
Focus on buffer filling which may be where L40s hangs.
"""

import sys
import os
import time

# Add PWM to path
PWM_DIR = "/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM"
sys.path.insert(0, os.path.join(PWM_DIR, "src"))

import torch
print("="*60)
print("L40s Buffer Filling Test")
print("="*60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print("="*60)
sys.stdout.flush()

# Import dflex
print("\n1. Importing dflex...")
import dflex as df
from dflex.envs import AntEnv
print("✓ dflex imported")
sys.stdout.flush()

# Create environment with FULL SIZE (256 envs like actual training)
print("\n2. Creating AntEnv with num_envs=256 (FULL SIZE)...")
start_time = time.time()
env = AntEnv(
    render=False,
    device="cuda:0",
    num_envs=256,  # FULL SIZE
    stochastic_init=True,
    no_grad=False,
    episode_length=1000,
    MM_caching_frequency=16,
    early_termination=True,
    termination_height=0.27,
)
elapsed = time.time() - start_time
print(f"✓ Environment created in {elapsed:.2f}s")
print(f"  - num_envs: {env.num_envs}")
sys.stdout.flush()

# Reset environment
print("\n3. Resetting environment...")
start_time = time.time()
obs = env.reset()
elapsed = time.time() - start_time
print(f"✓ Environment reset in {elapsed:.2f}s")
sys.stdout.flush()

# Import Buffer
print("\n4. Creating Buffer with capacity=2M (FULL SIZE)...")
from pwm.utils.buffer import Buffer
buffer = Buffer(
    num_obs=env.num_obs,
    num_actions=env.num_actions,
    capacity=2_000_000,  # FULL SIZE
    device="cuda:0"
)
print(f"✓ Buffer created with capacity {buffer.capacity:,}")
sys.stdout.flush()

# CRITICAL TEST: Fill buffer to min_size (this is where training may hang!)
print("\n5. Filling buffer to min_size (5000 transitions)...")
print("   This is the CRITICAL step that may hang on L40s!")
print(f"   Starting at: {time.strftime('%H:%M:%S')}")
sys.stdout.flush()

min_size = 5000
transitions_collected = 0
start_time = time.time()
last_print = start_time

obs = env.reset()
while transitions_collected < min_size:
    # Random action
    action = torch.randn((env.num_envs, env.num_actions), device="cuda:0")
    
    # Step environment
    next_obs, reward, done, info = env.step(action)
    
    # Add to buffer
    buffer.add(obs, action, reward, next_obs, done)
    obs = next_obs
    
    transitions_collected += env.num_envs
    
    # Print progress every 5 seconds
    current_time = time.time()
    if current_time - last_print >= 5.0:
        elapsed = current_time - start_time
        rate = transitions_collected / elapsed
        remaining = (min_size - transitions_collected) / rate if rate > 0 else 0
        print(f"   Progress: {transitions_collected}/{min_size} ({100*transitions_collected/min_size:.1f}%) - "
              f"Rate: {rate:.1f} trans/s - ETA: {remaining:.1f}s")
        sys.stdout.flush()
        last_print = current_time

total_time = time.time() - start_time
print(f"\n✓ Buffer filling completed in {total_time:.2f}s!")
print(f"  - Rate: {min_size/total_time:.1f} transitions/second")
print(f"  - Buffer size: {len(buffer):,}")
sys.stdout.flush()

# Test sampling
print("\n6. Testing buffer sampling (batch_size=1024)...")
batch = buffer.sample(1024)
print(f"✓ Buffer sampling successful")
print(f"  - obs shape: {batch['obs'].shape}")
sys.stdout.flush()

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("L40s can successfully fill buffer and is ready for training!")
print("="*60)
sys.stdout.flush()
