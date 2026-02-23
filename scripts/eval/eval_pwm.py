#!/usr/bin/env python3
"""Simple evaluation script for PWM checkpoints."""
import argparse, sys, os, torch, numpy as np, pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def instantiate_eval_env(cfg, device='cuda:0', num_envs=128):
    """Instantiate env directly from the run Hydra config."""
    env_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.env.config, resolve=True)
    )
    env_cfg.device = device
    env_cfg.num_envs = num_envs
    if "no_grad" in env_cfg:
        env_cfg.no_grad = True

    return instantiate(env_cfg, logdir=str(PROJECT_ROOT / "logs" / "eval"))


def detect_env(config):
    env_target = str(config.get('env', {}).get('config', {}).get('_target_', ''))
    task_id = str(config.get('env', {}).get('config', {}).get('task_id', '')).strip()
    if 'create_mjlab_pwm_env' in env_target:
        return f"mjlab_{task_id}" if task_id else "mjlab"
    if 'AnymalEnv' in env_target:
        return 'dflex_anymal'
    elif 'HumanoidEnv' in env_target:
        return 'dflex_humanoid'
    elif 'AntEnv' in env_target:
        return 'dflex_ant'
    return env_target.split('.')[-1] if env_target else "unknown_env"


def parse_variant(alg):
    use_flow = alg.get('use_flow_dynamics', False)
    integrator = alg.get('flow_integrator', 'heun')
    substeps = alg.get('flow_substeps', 4)
    actor_target = str(alg.get('actor_config', {}).get('_target_', ''))
    # Check specifically for FlowActor class or FlowODE, avoiding package name matches
    flow_policy = 'FlowODE' in actor_target or 'FlowActor' in actor_target
    
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
    
    alg = cfg.alg
    latent_dim = alg.get('latent_dim', 64)
    
    print(f"Loading env from config: {env_name}")
    env = instantiate_eval_env(cfg, device=device, num_envs=128)

    obs_dim = getattr(env, "num_obs", env.observation_space.shape[0])
    act_dim = getattr(env, "num_actions", env.action_space.shape[0])
    
    # Instantiate world model with all required args
    print("Creating world model...")
    wm = instantiate(
        alg.world_model_config,
        observation_dim=obs_dim,
        action_dim=act_dim,
        latent_dim=latent_dim,
        _recursive_=True
    ).to(device)
    
    # Load checkpoint first to detect actor type from weights
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    actor_state = ckpt['actor']
    is_flow_actor = any('velocity_net' in k for k in actor_state.keys())
    
    # Instantiate actor based on checkpoint content, NOT config
    print(f"Creating actor... (Detected FlowActor: {is_flow_actor})")
    
    if is_flow_actor:
        from flow_mbpo_pwm.models.actor import FlowActor
        # For FlowActor, we need flow-specific args. 
        # Attempt to get them from config, or reasonable defaults if missing/mismatched.
        actor = FlowActor(
            obs_dim=latent_dim,
            action_dim=act_dim,
            # If config is mismatched (says Baseline), these might be missing.
            # We assume defaults or try to read from alg if available.
            flow_model_config=alg.get('flow_model_config', {'_target_': 'flow_mbpo_pwm.models.mlp.MLP', 'units': [400, 200, 100]}),
            t_embedding_dim=alg.get('t_embedding_dim', 32),
            use_flow_dynamics=True # Force true for evaluation if it is a FlowActor
        ).to(device)
    else:
        # Baseline Actor
        actor = instantiate(
            alg.actor_config,
            obs_dim=latent_dim,
            action_dim=act_dim,
            _recursive_=True
        ).to(device)

    wm.load_state_dict(ckpt['world_model'])
    actor.load_state_dict(actor_state)
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
