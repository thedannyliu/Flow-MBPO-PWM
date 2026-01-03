#!/usr/bin/env python3
"""
Simple evaluation script - loads actor and world model directly, runs eval loop.
"""
import argparse, sys, os, torch, numpy as np, pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate

PROJECT_ROOT = Path("/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import and create module aliases
import flow_mbpo_pwm
import flow_mbpo_pwm.models
import flow_mbpo_pwm.models.actor
import flow_mbpo_pwm.models.world_model
import flow_mbpo_pwm.models.flow_world_model

sys.modules['pwm'] = flow_mbpo_pwm  
sys.modules['pwm.models'] = flow_mbpo_pwm.models
sys.modules['pwm.models.actor'] = flow_mbpo_pwm.models.actor
sys.modules['pwm.models.world_model'] = flow_mbpo_pwm.models.world_model
sys.modules['pwm.models.flow_world_model'] = flow_mbpo_pwm.models.flow_world_model


def create_env(env_name, device='cuda:0', num_envs=128):
    if env_name == 'dflex_ant':
        from dflex.envs import AntEnv
        return AntEnv(render=False, device=device, num_envs=num_envs, stochastic_init=True,
                      no_grad=True, episode_length=1000, MM_caching_frequency=1,
                      early_termination=True, termination_height=0.27, action_penalty=0.0,
                      joint_vel_obs_scaling=0.1, up_rew_scale=0.1)
    elif env_name == 'dflex_anymal':
        from dflex.envs import AnymalEnv
        return AnymalEnv(render=False, device=device, num_envs=num_envs, stochastic_init=True,
                         no_grad=True, episode_length=1000, MM_caching_frequency=16, 
                         early_termination=True)
    elif env_name == 'dflex_humanoid':
        from dflex.envs import HumanoidEnv
        return HumanoidEnv(render=False, device=device, num_envs=num_envs, stochastic_init=True,
                           no_grad=True, episode_length=1000, MM_caching_frequency=48,
                           early_termination=True, termination_height=0.74, 
                           action_penalty=-0.002, joint_vel_obs_scaling=0.1,
                           termination_tolerance=0.1, height_rew_scale=10.0,
                           up_rew_scale=0.1, heading_rew_scale=1.0)
    raise ValueError(f"Unknown env: {env_name}")


def detect_env(config):
    env_target = str(config.get('env', {}).get('config', {}).get('_target_', ''))
    if 'AnymalEnv' in env_target:
        return 'dflex_anymal'
    elif 'HumanoidEnv' in env_target:
        return 'dflex_humanoid'
    elif 'AntEnv' in env_target:
        return 'dflex_ant'
    return None


def parse_variant(alg):
    use_flow = alg.get('use_flow_dynamics', False)
    integrator = alg.get('flow_integrator', 'heun')
    substeps = alg.get('flow_substeps', 4)
    actor_target = str(alg.get('actor_config', {}).get('_target_', ''))
    flow_policy = 'FlowODE' in actor_target or 'flow' in actor_target.lower()
    
    if use_flow and flow_policy:
        return f"FullFlow_K{substeps}"
    elif use_flow:
        return f"FlowWM_K{substeps}_{integrator}"
    elif flow_policy:
        return "FlowPolicy"
    return "Baseline"


@torch.no_grad()
def eval_policy(actor, wm, env, num_games=100, deterministic=True, device='cuda:0'):
    """Evaluate policy using real environment rewards."""
    num_envs = env.num_envs
    games_played = 0
    total_rewards = []
    total_lengths = []
    
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    
    obs = env.reset()
    
    while games_played < num_games:
        z = wm.encode(obs, task=None)
        action = actor(z, deterministic=deterministic)
        action = torch.tanh(action)
        obs, reward, done, info = env.step(action)
        
        if isinstance(reward, torch.Tensor) and reward.dim() > 1:
            reward = reward.squeeze(-1)
        
        episode_rewards += reward
        episode_lengths += 1
        
        done_mask = done if isinstance(done, torch.Tensor) else torch.tensor([done], device=device)
        if done_mask.any():
            done_indices = done_mask.nonzero(as_tuple=True)[0]
            for idx in done_indices:
                if games_played < num_games:
                    total_rewards.append(episode_rewards[idx].item())
                    total_lengths.append(episode_lengths[idx].item())
                    games_played += 1
                    if games_played % 20 == 0:
                        print(f"Evaluated {games_played}/{num_games}")
                    episode_rewards[idx] = 0
                    episode_lengths[idx] = 0
    
    return np.mean(total_rewards), np.std(total_rewards), np.mean(total_lengths)


def evaluate_checkpoint(ckpt_path, num_games=100, device='cuda:0'):
    ckpt_path = Path(ckpt_path)
    run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == 'logs' else ckpt_path.parent
    config_path = run_dir / '.hydra' / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    env_name = detect_env(cfg)
    if not env_name:
        raise ValueError("Could not detect environment")
    
    alg = cfg.alg
    latent_dim = alg.get('latent_dim', 64)
    
    print(f"Loading env: {env_name}")
    env = create_env(env_name, device)
    
    # Instantiate world model with all required args
    print("Creating world model...")
    wm = instantiate(
        alg.world_model_config,
        observation_dim=env.num_obs,
        action_dim=env.num_actions,
        latent_dim=latent_dim,
        _recursive_=True
    ).to(device)
    
    # Instantiate actor
    print("Creating actor...")
    actor = instantiate(
        alg.actor_config,
        obs_dim=latent_dim,
        action_dim=env.num_actions,
        _recursive_=True
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    wm.load_state_dict(ckpt['world_model'])
    actor.load_state_dict(ckpt['actor'])
    wm.eval()
    actor.eval()
    
    # Run evaluation
    print(f"Evaluating ({num_games} games)...")
    mean_reward, std_reward, mean_length = eval_policy(actor, wm, env, num_games, True, device)
    
    seed = cfg.get('general', {}).get('seed', 0)
    variant = parse_variant(alg)
    
    result = {
        'Task': env_name.replace('dflex_', '').capitalize(),
        'Variant': variant,
        'Seed': seed,
        'Integrator': alg.get('flow_integrator', 'N/A') if alg.get('use_flow_dynamics', False) else 'N/A',
        'Substeps': alg.get('flow_substeps', 'N/A') if alg.get('use_flow_dynamics', False) else 'N/A',
        'MeanReward': round(mean_reward, 2),
        'StdReward': round(std_reward, 2),
        'MeanLength': round(mean_length, 1),
        'NumGames': num_games,
    }
    
    print(f"\n{'='*50}")
    print(f"Result: {variant} s{seed} = {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"{'='*50}")
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-games', type=int, default=100)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='eval_result.csv')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    result = evaluate_checkpoint(args.checkpoint, args.num_games, args.device)
    pd.DataFrame([result]).to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
