#!/usr/bin/env python3
"""
Diagnostic script to identify where L40s training hangs.
Tests each initialization step separately with detailed logging.
"""

import sys
import os
import time
import torch
import numpy as np

# Add PWM to path
PWM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PWM_DIR, "src"))
print(f"Added to path: {os.path.join(PWM_DIR, 'src')}")
sys.stdout.flush()

def print_step(step_num, description):
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}")
    sys.stdout.flush()

def main():
    print_step(0, "Starting L40s Diagnostic")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    sys.stdout.flush()
    
    # Step 1: Import dflex
    print_step(1, "Importing dflex")
    try:
        import dflex as df
        print("✓ dflex imported successfully")
    except Exception as e:
        print(f"✗ Failed to import dflex: {e}")
        return
    sys.stdout.flush()
    
    # Step 2: Create simple environment
    print_step(2, "Creating dflex environment")
    try:
        from dflex.envs import AntEnv
        print("Creating AntEnv with num_envs=4 (small test)...")
        env = AntEnv(
            render=False,
            device="cuda:0",
            num_envs=4,
            stochastic_init=True,
            no_grad=False,
            episode_length=1000,
        )
        print(f"✓ Environment created")
        print(f"  - num_envs: {env.num_envs}")
        print(f"  - obs_dim: {env.num_obs}")
        print(f"  - act_dim: {env.num_actions}")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return
    sys.stdout.flush()
    
    # Step 3: Reset environment
    print_step(3, "Resetting environment")
    try:
        print("Calling env.reset()...")
        start_time = time.time()
        obs = env.reset()
        elapsed = time.time() - start_time
        print(f"✓ Environment reset successful in {elapsed:.2f}s")
        print(f"  - obs shape: {obs.shape}")
        print(f"  - obs dtype: {obs.dtype}")
        print(f"  - obs device: {obs.device}")
    except Exception as e:
        print(f"✗ Failed to reset environment: {e}")
        import traceback
        traceback.print_exc()
        return
    sys.stdout.flush()
    
    # Step 4: Take a few steps
    print_step(4, "Taking environment steps")
    try:
        action = torch.zeros((env.num_envs, env.num_actions), device="cuda:0")
        print("Taking 10 environment steps...")
        for i in range(10):
            start_time = time.time()
            obs, reward, done, info = env.step(action)
            elapsed = time.time() - start_time
            print(f"  Step {i+1}: {elapsed*1000:.1f}ms, reward={reward.mean().item():.3f}")
            sys.stdout.flush()
        print("✓ Environment steps successful")
    except Exception as e:
        print(f"✗ Failed environment step: {e}")
        import traceback
        traceback.print_exc()
        return
    sys.stdout.flush()
    
    # Step 5: Create models
    print_step(5, "Creating neural network models")
    try:
        from pwm.models.world_model import WorldModel
        print("Creating WorldModel...")
        wm = WorldModel(
            obs_dim=env.num_obs,
            action_dim=env.num_actions,
            units=[256, 256],
            encoder_units=[256, 256],
            latent_dim=128,
            num_bins=101,
            vmin=-10.0,
            vmax=10.0,
            task_dim=0,
            multitask=False,
        ).to("cuda:0")
        print(f"✓ WorldModel created: {sum(p.numel() for p in wm.parameters())} parameters")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return
    sys.stdout.flush()
    
    # Step 6: Forward pass
    print_step(6, "Testing model forward pass")
    try:
        print("Running forward pass...")
        with torch.no_grad():
            obs_tensor = torch.randn(32, env.num_obs, device="cuda:0")
            action_tensor = torch.randn(32, env.num_actions, device="cuda:0")
            start_time = time.time()
            latent = wm.encode_obs(obs_tensor)
            pred_obs_dist, pred_rew_dist = wm.forward(latent, action_tensor)
            elapsed = time.time() - start_time
        print(f"✓ Forward pass successful in {elapsed*1000:.1f}ms")
        print(f"  - latent shape: {latent.shape}")
    except Exception as e:
        print(f"✗ Failed forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    sys.stdout.flush()
    
    # Step 7: Create replay buffer
    print_step(7, "Creating replay buffer")
    try:
        from pwm.utils.replay_buffer import ReplayBuffer
        print("Creating ReplayBuffer with capacity=10000...")
        buffer = ReplayBuffer(
            obs_dim=env.num_obs,
            action_dim=env.num_actions,
            capacity=10000,
            device="cuda:0"
        )
        print(f"✓ ReplayBuffer created")
        print(f"  - capacity: {buffer.capacity}")
        print(f"  - device: {buffer.device}")
    except Exception as e:
        print(f"✗ Failed to create buffer: {e}")
        import traceback
        traceback.print_exc()
        return
    sys.stdout.flush()
    
    # Step 8: Collect data to buffer
    print_step(8, "Collecting data to buffer")
    try:
        print("Collecting 100 transitions...")
        obs = env.reset()
        for i in range(100):
            action = torch.randn((env.num_envs, env.num_actions), device="cuda:0")
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            if (i + 1) % 20 == 0:
                print(f"  Collected {i+1}/100 transitions")
                sys.stdout.flush()
        print(f"✓ Data collection successful")
        print(f"  - buffer size: {len(buffer)}")
    except Exception as e:
        print(f"✗ Failed data collection: {e}")
        import traceback
        traceback.print_exc()
        return
    sys.stdout.flush()
    
    # Step 9: Sample from buffer
    print_step(9, "Sampling from buffer")
    try:
        print("Sampling batch of 32...")
        batch = buffer.sample(32)
        print(f"✓ Buffer sampling successful")
        print(f"  - obs shape: {batch['obs'].shape}")
        print(f"  - action shape: {batch['action'].shape}")
    except Exception as e:
        print(f"✗ Failed buffer sampling: {e}")
        import traceback
        traceback.print_exc()
        return
    sys.stdout.flush()
    
    print_step(10, "ALL TESTS PASSED!")
    print("L40s environment is functional. Issue may be in training loop.")
    print("Check:")
    print("  1. Initial buffer filling (may take long time)")
    print("  2. WandB initialization")
    print("  3. Hydra configuration")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
