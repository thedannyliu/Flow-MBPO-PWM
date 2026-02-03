"""Train PWM online on DMControl environments via MuJoCo Playground."""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import json
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mujoco_playground import registry
from mujoco_playground._src import wrapper_torch
from frank.pwm_env_adapter import PWMEnvAdapter


# DMControl tasks available in MuJoCo Playground (CamelCase format)
DMCONTROL_TASKS = [
    "WalkerStand",
    "WalkerWalk",
    "WalkerRun",
    "CheetahRun",
    "HopperStand",
    "HopperHop",
    "CartpoleBalance",
    "CartpoleBalanceSparse",
    "CartpoleSwingup",
    "CartpoleSwingupSparse",
    "ReacherEasy",
    "ReacherHard",
    "FingerSpin",
    "FingerTurnEasy",
    "FingerTurnHard",
    "PendulumSwingup",
    "BallInCup",
    "FishSwim",
    "AcrobotSwingup",
    "HumanoidStand",
    "HumanoidWalk",
    "HumanoidRun",
    "PointMass",
    "SwimmerSwimmer6",
]


def task_to_filename(task: str) -> str:
    """Convert CamelCase task name to snake_case for filenames."""
    import re
    # Insert underscore before uppercase letters and convert to lowercase
    return re.sub(r'(?<!^)(?=[A-Z])', '_', task).lower()


def create_pwm_env(task: str, num_envs: int, episode_length: int, seed: int = 42):
    """Create PWM-compatible environment from MuJoCo Playground."""
    env_cfg = registry.get_default_config(task)
    env_cfg.episode_length = episode_length * 2  # Account for action repeat
    env_cfg.action_repeat = 2

    raw_env = registry.load(task, config=env_cfg)

    brax_wrapper = wrapper_torch.RSLRLBraxWrapper(
        raw_env, num_envs, seed, episode_length, action_repeat=2
    )

    return PWMEnvAdapter(brax_wrapper)


def train_pwm(task: str, num_envs: int = 32, max_epochs: int = 1000, seed: int = 42):
    """Train PWM on a single DMControl task.

    Args:
        task: Task name in CamelCase (e.g., "WalkerStand").
        num_envs: Number of parallel environments.
        max_epochs: Maximum training epochs.
        seed: Random seed.
    """
    cfg = OmegaConf.load(f"{_repo_root}/scripts/cfg/alg/pwm_5M.yaml")

    env = create_pwm_env(
        task=task,
        num_envs=num_envs,
        episode_length=500,
        seed=seed,
    )

    print(f"\n{'='*70}")
    print(f"TRAINING: {task}")
    print(f"{'='*70}")
    print(f"  num_envs: {num_envs}")
    print(f"  obs_dim: {env.observation_space.shape[0]}")
    print(f"  act_dim: {env.action_space.shape[0]}")
    print(f"  episode_length: {env.episode_length}")
    print(f"  max_epochs: {max_epochs}")
    print(f"  seed: {seed}")

    cfg.max_epochs = max_epochs
    cfg.horizon = 16
    cfg.device = "cuda"
    cfg.save_interval = 999999999  # effectively disable periodic checkpoints

    cfg.ret_rms = False
    cfg.detach = False

    task_filename = task_to_filename(task)
    logdir = f"logs/pwm_dmcontrol/{task_filename}/seed_{seed}"

    # Initialize wandb
    wandb.init(
        project="flow-mbpo-pwm-mjwarp",
        name=f"{task}_seed{seed}",
        config={
            "task": task,
            "num_envs": num_envs,
            "max_epochs": max_epochs,
            "seed": seed,
            "horizon": cfg.horizon,
        },
        dir=logdir,
    )

    pwm = instantiate(
        cfg,
        env=env,
        logdir=logdir,
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        log=True,
    )

    # Train and collect metrics
    pwm.train()

    best_reward = -pwm.best_policy_loss

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE: {task}")
    print(f"  Highest Reward: {best_reward:.2f}")
    print(f"  Total Steps: {pwm.step_count}")
    print(f"  Logdir: {logdir}")
    print(f"{'='*70}\n")

    # Save final summary as JSON for easy parsing
    summary = {
        "task": task,
        "best_reward": float(best_reward),
        "total_steps": int(pwm.step_count),
        "num_envs": num_envs,
        "max_epochs": max_epochs,
        "seed": seed,
    }

    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Close wandb run
    wandb.finish()

    return best_reward


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PWM on DMControl environments")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=DMCONTROL_TASKS,
        help="DMControl task name (CamelCase)"
    )
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_default_device('cuda')
    train_pwm(args.task, args.num_envs, args.max_epochs, args.seed)
