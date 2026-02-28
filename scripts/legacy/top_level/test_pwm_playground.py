"""Test PWM online training with MuJoCo Playground backend."""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mujoco_playground import registry
from mujoco_playground._src import wrapper_torch
from frank.pwm_env_adapter import PWMEnvAdapter


def create_pwm_env(task: str, num_envs: int, episode_length: int, seed: int = 42):
    """Create PWM-compatible environment from MuJoCo Playground.

    Args:
        task: Task name from MuJoCo Playground registry (e.g., "WalkerStand").
        num_envs: Number of parallel environments.
        episode_length: Maximum episode length (in action-repeated steps).
        seed: Random seed.

    Returns:
        PWMEnvAdapter wrapping the RSLRLBraxWrapper.
    """
    env_cfg = registry.get_default_config(task)
    env_cfg.episode_length = episode_length * 2  # Account for action repeat
    env_cfg.action_repeat = 2

    raw_env = registry.load(task, config=env_cfg)

    brax_wrapper = wrapper_torch.RSLRLBraxWrapper(
        raw_env, num_envs, seed, episode_length, action_repeat=2
    )

    return PWMEnvAdapter(brax_wrapper)


def test_pwm_training(num_envs: int = 64, max_epochs: int = 1000):
    """Test PWM training with playground environment.

    Args:
        num_envs: Number of parallel environments.
        max_epochs: Maximum training epochs.
    """
    cfg = OmegaConf.load(f"{_repo_root}/scripts/cfg/alg/pwm_5M.yaml")

    env = create_pwm_env(
        task="WalkerStand",
        num_envs=num_envs,
        episode_length=500,
    )

    print(f"\nEnvironment: WalkerStand")
    print(f"  num_envs: {num_envs}")
    print(f"  obs_dim: {env.observation_space.shape[0]}")
    print(f"  act_dim: {env.action_space.shape[0]}")
    print(f"  episode_length: {env.episode_length}")

    cfg.max_epochs = max_epochs
    cfg.horizon = 16
    cfg.device = "cuda"
    cfg.save_interval = 100

    cfg.ret_rms = False 
    cfg.detach = False

    pwm = instantiate(
        cfg,
        env=env,
        logdir="logs/pwm_playground_test",
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
    )

    pwm.train()

    best_reward = -pwm.best_policy_loss
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE")
    print(f"  Highest Reward: {best_reward:.2f}")
    print(f"  Total Steps: {pwm.step_count}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=1000)
    args = parser.parse_args()

    torch.set_default_device('cuda')
    test_pwm_training(args.num_envs, args.max_epochs)
