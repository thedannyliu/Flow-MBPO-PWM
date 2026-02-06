"""Train Flow-MBPO with DM Control tasks via MuJoCo Playground (MJX/GPU-accelerated)."""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import hydra
import wandb
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from mujoco_playground import registry
from mujoco_playground._src import wrapper_torch
from frank.pwm_env_adapter import PWMEnvAdapter
from flow_mbpo_pwm.utils.common import seeding

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)


def create_pwm_env(task: str, num_envs: int, episode_length: int, action_repeat: int, seed: int = 42):
    """Create PWM-compatible environment from MuJoCo Playground.

    Args:
        task: Task name from MuJoCo Playground registry (e.g., "WalkerStand").
        num_envs: Number of parallel GPU environments.
        episode_length: Maximum episode length (in action-repeated steps).
        action_repeat: Number of physics steps per action.
        seed: Random seed.

    Returns:
        PWMEnvAdapter wrapping the RSLRLBraxWrapper.
    """
    env_cfg = registry.get_default_config(task)
    env_cfg.episode_length = episode_length * action_repeat
    env_cfg.action_repeat = action_repeat

    raw_env = registry.load(task, config=env_cfg)

    brax_wrapper = wrapper_torch.RSLRLBraxWrapper(
        raw_env, num_envs, seed, episode_length, action_repeat=action_repeat
    )

    return PWMEnvAdapter(brax_wrapper)


def create_wandb_run(wandb_cfg, job_config, run_id=None):
    task_name = job_config["env"]["task"]
    try:
        alg_name = job_config["alg"]["_target_"].split(".")[-1]
    except:
        alg_name = job_config["alg"]["name"].upper()
    
    seed = job_config.get('general', {}).get('seed', 42)
    
    if hasattr(wandb_cfg, 'name') and wandb_cfg.name:
        name = wandb_cfg.name
    else:
        try:
            job_id = HydraConfig().get().job.num
            name = f"{alg_name}_{task_name}_sweep_{seed}"
        except:
            name = f"{alg_name}_{task_name}_s{seed}"
    
    notes = getattr(wandb_cfg, 'notes', '') if hasattr(wandb_cfg, 'notes') else ''
    
    print(f"Initializing WandB run: name='{name}', project='{wandb_cfg.project}'")
    
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


@hydra.main(config_path="cfg", config_name="config_dmcontrol_mjx.yaml", version_base="1.2")
def train(cfg: DictConfig):
    cfg_full = OmegaConf.to_container(cfg, resolve=True)

    if cfg.general.run_wandb:
        create_wandb_run(cfg.wandb, cfg_full)

    logdir = HydraConfig.get()["runtime"]["output_dir"]
    logdir = os.path.join(logdir, cfg.general.logdir)

    seeding(cfg.general.seed, False)

    env = create_pwm_env(
        task=cfg.env.task,
        num_envs=cfg.env.num_envs,
        episode_length=cfg.env.episode_length,
        action_repeat=cfg.env.action_repeat,
        seed=cfg.general.seed,
    )

    print(f"\nEnvironment: {cfg.env.task}")
    print(f"  num_envs: {cfg.env.num_envs}")
    print(f"  obs_dim: {env.observation_space.shape[0]}")
    print(f"  act_dim: {env.action_space.shape[0]}")
    print(f"  episode_length: {env.episode_length}")
    print(f"  action_repeat: {cfg.env.action_repeat}")

    pwm = instantiate(
        cfg.alg,
        env=env,
        logdir=logdir,
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
    )

    pwm.train()

    best_reward = -pwm.best_policy_loss
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE")
    print(f"  Task: {cfg.env.task}")
    print(f"  Highest Reward: {best_reward:.2f}")
    print(f"  Total Steps: {pwm.step_count}")
    print("=" * 70)

    if cfg.general.run_wandb:
        wandb.finish()


if __name__ == "__main__":
    torch.set_default_device('cuda')
    train()
