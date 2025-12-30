#!/usr/bin/env python3
"""
Evaluate trained PWM policies

Usage:
    # Evaluate single checkpoint
    python scripts/evaluate_policy.py \
        --checkpoint outputs/.../best_policy.pt \
        --env dflex_ant \
        --num-episodes 100 \
        --render
    
    # Compare baseline vs flow
    python scripts/evaluate_policy.py \
        --baseline outputs/.../pwm_5M_baseline/best_policy.pt \
        --flow outputs/.../pwm_5M_flow/best_policy.pt \
        --env dflex_ant \
        --num-episodes 100
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add PWM to path
PWM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PWM_DIR / "src"))

from flow_mbpo_pwm.algorithms.pwm import PWM


def load_checkpoint(checkpoint_path, env_name, device='cuda:0'):
    """Load PWM checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint_path = Path(checkpoint_path)
    
    # Load checkpoint to get actor weights
    ckpt_data = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint keys: {list(ckpt_data.keys())}")
    
    # Create environment using Hydra
    if env_name == 'dflex_ant':
        # Import dflex environment
        from dflex.envs import AntEnv
        
        environment = AntEnv(
            render=False,
            device=device,
            num_envs=1,  # Single env for evaluation
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
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    print(f"Environment: {environment.num_obs} obs, {environment.num_actions} actions")
    
    # Determine model type from path
    is_flow = 'flow' in str(checkpoint_path)
    model_size = '48M' if '48M' in str(checkpoint_path) else '5M'
    print(f"Model type: {'Flow' if is_flow else 'Baseline'}, size={model_size}")
    
    # Load models from checkpoint
    if 'actor' not in ckpt_data or 'world_model' not in ckpt_data:
        raise KeyError(f"Missing models in checkpoint. Keys: {list(ckpt_data.keys())}")
    
    actor_state = ckpt_data['actor']
    wm_state = ckpt_data['world_model']
    
    print(f"Loading models...")
    
    # Create actor network
    from flow_mbpo_pwm.models.actor import ActorStochasticMLP
    
    # Determine latent_dim from actor input
    first_weight = actor_state['mu_net.0.weight']
    latent_dim = first_weight.shape[1]  # Input dimension to actor
    
    actor = ActorStochasticMLP(
        obs_dim=latent_dim,
        action_dim=environment.num_actions,
        units=[400, 200, 100],
        activation_class=torch.nn.ELU,
    ).to(device)
    actor.load_state_dict(actor_state)
    actor.eval()
    
    print(f"  Actor: {latent_dim} -> {environment.num_actions}")
    
    # Load world model config from appropriate YAML
    config_dir = PWM_DIR / "scripts" / "cfg" / "alg"
    if is_flow:
        config_file = config_dir / f"pwm_{model_size}_flow.yaml"
    else:
        config_file = config_dir / f"pwm_{model_size}.yaml"
    
    alg_config = OmegaConf.load(config_file)
    wm_config_dict = OmegaConf.to_container(alg_config.world_model_config, resolve=True)
    
    # Instantiate world model
    wm = instantiate(
        alg_config.world_model_config,
        observation_dim=environment.num_obs,
        action_dim=environment.num_actions,
        latent_dim=latent_dim,
    ).to(device)
    
    wm.load_state_dict(wm_state)
    wm.eval()
    
    print(f"  World Model: {environment.num_obs} -> {latent_dim}")
    print("Models loaded successfully")
    
    # Create a wrapper that mimics PWM's act method
    class PolicyWrapper:
        def __init__(self, actor, world_model, device):
            self.actor = actor
            self.wm = world_model
            self.device = device
        
        def act(self, obs, deterministic=True):
            """Act method compatible with evaluation loop"""
            # Encode observation
            with torch.no_grad():
                z = self.wm.encode(obs, task=None)
                action = self.actor(z, deterministic=deterministic)
                action = torch.tanh(action)
            return action
    
    agent = PolicyWrapper(actor, wm, device)
    
    return agent, environment


def evaluate_policy(agent, env, num_episodes=100, render=False, verbose=True):
    """Evaluate policy performance"""
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # DFlex environments don't use 'done' flag the same way
        # Run for fixed episode length
        for step in range(1000):
            # Get action from policy (deterministic for evaluation)
            action = agent.act(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Accumulate metrics
            if isinstance(reward, torch.Tensor):
                episode_reward += reward.mean().item()
            else:
                episode_reward += reward
            
            episode_length += 1
            
            if render and ep < 5:  # Only render first 5 episodes
                env.render()
            
            # Check termination
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check if episode succeeded (task-specific heuristic)
        success = episode_reward > 100  # Ant task success threshold
        episode_successes.append(success)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}")
    
    # Compute statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': np.mean(episode_successes),
        'rewards': episode_rewards,
        'lengths': episode_lengths,
    }
    
    return results


def compare_policies(results_dict, save_path=None):
    """Compare multiple policies and create visualization"""
    
    # Create comparison table
    comparison = []
    for name, results in results_dict.items():
        comparison.append({
            'Policy': name,
            'Mean Reward': f"{results['mean_reward']:.2f} ± {results['std_reward']:.2f}",
            'Mean Length': f"{results['mean_length']:.1f} ± {results['std_length']:.1f}",
            'Success Rate': f"{results['success_rate']*100:.1f}%",
        })
    
    df = pd.DataFrame(comparison)
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Reward distribution
    ax = axes[0]
    for name, results in results_dict.items():
        ax.hist(results['rewards'], alpha=0.5, label=name, bins=30)
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Episode Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    ax = axes[1]
    data = [results['rewards'] for results in results_dict.values()]
    labels = list(results_dict.keys())
    ax.boxplot(data, labels=labels)
    ax.set_ylabel('Episode Reward')
    ax.set_title('Reward Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    else:
        plt.show()
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Evaluate PWM policies')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--baseline', type=str, help='Path to baseline checkpoint')
    parser.add_argument('--flow', type=str, help='Path to flow checkpoint')
    parser.add_argument('--env', type=str, default='dflex_ant', help='Environment name')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Prepare output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
    
    results_dict = {}
    
    # Evaluate single checkpoint
    if args.checkpoint:
        print(f"\nEvaluating checkpoint: {args.checkpoint}")
        agent, env = load_checkpoint(args.checkpoint, args.env, args.device)
        results = evaluate_policy(agent, env, args.num_episodes, args.render)
        results_dict['Policy'] = results
        
        print(f"\nResults:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
        print(f"  Success Rate: {results['success_rate']*100:.1f}%")
    
    # Compare baseline vs flow
    if args.baseline and args.flow:
        print(f"\n{'='*80}")
        print("COMPARING BASELINE vs FLOW")
        print('='*80)
        
        # Evaluate baseline
        print(f"\n1. Evaluating BASELINE: {args.baseline}")
        agent_baseline, env = load_checkpoint(args.baseline, args.env, args.device)
        results_baseline = evaluate_policy(agent_baseline, env, args.num_episodes, False)
        results_dict['Baseline'] = results_baseline
        
        # Evaluate flow
        print(f"\n2. Evaluating FLOW: {args.flow}")
        agent_flow, env = load_checkpoint(args.flow, args.env, args.device)
        results_flow = evaluate_policy(agent_flow, env, args.num_episodes, False)
        results_dict['Flow'] = results_flow
        
        # Compare
        comparison_df = compare_policies(
            results_dict, 
            save_path=output_dir / "comparison.png"
        )
        
        # Save results
        comparison_df.to_csv(output_dir / "comparison.csv", index=False)
        print(f"\nSaved comparison to: {output_dir}/comparison.csv")
        
        # Compute improvement
        improvement = (results_flow['mean_reward'] - results_baseline['mean_reward']) / abs(results_baseline['mean_reward']) * 100
        print(f"\n{'='*80}")
        print(f"FLOW IMPROVEMENT: {improvement:+.2f}%")
        print('='*80)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
