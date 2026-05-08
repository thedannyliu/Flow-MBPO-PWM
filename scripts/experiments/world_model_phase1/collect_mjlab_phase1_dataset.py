#!/usr/bin/env python3
"""
Collect a fixed offline dataset for Phase 1 world-model fitting experiments.

The output format is a torch-saved dict containing fixed-length windows built
from MJLab episodes. Each window follows the same alignment convention used by
the online PWM replay buffer:
  - row 0 contains the starting observation and ignored action/reward placeholders
  - rows 1..T contain the transition tuples that led to each next observation
`window_length` refers to the desired world-model horizon. Each saved sample
therefore contains `window_length + 1` rows so the first row is the conditioning
observation and the remaining rows correspond to the next `window_length`
transitions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tensordict import TensorDict

from flow_mbpo_pwm.algorithms.pwm import PWM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", required=True, help="Path to env yaml")
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--metadata-output", default=None, help="Optional metadata json path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--target-episodes", type=int, default=48)
    parser.add_argument("--episode-length", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument(
        "--action-mode",
        choices=["random_uniform", "zero", "teacher_policy", "mixed_episode"],
        default="random_uniform",
    )
    parser.add_argument("--teacher-alg-config", default=None, help="Alg yaml for teacher policy")
    parser.add_argument("--teacher-checkpoint", default=None, help="Checkpoint .pt for teacher policy")
    parser.add_argument("--teacher-deterministic", action="store_true")
    parser.add_argument("--mixed-teacher-prob", type=float, default=0.5)
    parser.add_argument("--wandb-project", default="flow-mbpo-phase1-wm-overfit")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="phase1_dataset_collection")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", default="phase1,dataset_collection,mjlab,world_model")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def _initial_tensordict(obs: torch.Tensor, action_dim: int, device: torch.device) -> TensorDict:
    return TensorDict(
        {
            "obs": obs.unsqueeze(0),
            "action": torch.full((1, action_dim), torch.nan, device=device),
            "reward": torch.full((1,), torch.nan, device=device),
            "term": torch.full((1,), torch.nan, dtype=torch.bool, device=device),
        },
        batch_size=(1,),
    )


def _step_tensordict(
    obs_before_reset: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    term: torch.Tensor,
) -> TensorDict:
    return TensorDict(
        {
            "obs": obs_before_reset.unsqueeze(0),
            "action": action.unsqueeze(0),
            "reward": reward.reshape(1),
            "term": term.reshape(1),
        },
        batch_size=(1,),
    )


def _build_teacher_agent(
    alg_config: str,
    checkpoint: str,
    obs_dim: int,
    act_dim: int,
    device: str,
    horizon: int,
    logdir: str,
) -> PWM:
    alg_cfg = OmegaConf.load(alg_config)
    alg_cfg.device = device
    alg_cfg.horizon = int(horizon)
    alg_cfg.max_epochs = 0
    agent = instantiate(
        alg_cfg,
        env=None,
        obs_dim=obs_dim,
        act_dim=act_dim,
        logdir=logdir,
        log=False,
    )
    agent.load(checkpoint, resume_training=False)
    return agent


@torch.no_grad()
def _teacher_actions(agent: PWM, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
    policy_obs = obs
    if agent.obs_rms is not None:
        policy_obs = agent.obs_rms.normalize(policy_obs)
    z = agent.wm.encode(policy_obs, task=None)
    actions = agent.actor(z, deterministic=deterministic)
    return torch.tanh(actions)


def _build_windows(
    episodes: List[TensorDict],
    episode_modes: List[str],
    horizon: int,
    stride: int,
    max_windows: int,
) -> Dict[str, torch.Tensor]:
    seq_len = horizon + 1
    obs_windows: List[torch.Tensor] = []
    action_windows: List[torch.Tensor] = []
    reward_windows: List[torch.Tensor] = []
    term_windows: List[torch.Tensor] = []
    source_episode: List[int] = []
    source_start: List[int] = []
    source_mode: List[int] = []
    mode_to_id = {
        "random_uniform": 0,
        "zero": 1,
        "teacher_policy": 2,
        "mixed_episode_teacher": 3,
        "mixed_episode_random": 4,
    }

    for ep_idx, episode in enumerate(episodes):
        ep_len = int(episode.batch_size[0])
        if ep_len < seq_len:
            continue
        for start in range(0, ep_len - seq_len + 1, stride):
            window = episode[start : start + seq_len]
            obs_windows.append(window["obs"])
            action_windows.append(window["action"])
            reward_windows.append(window["reward"])
            term_windows.append(window["term"])
            source_episode.append(ep_idx)
            source_start.append(start)
            source_mode.append(mode_to_id.get(episode_modes[ep_idx], -1))

    if not obs_windows:
        raise RuntimeError(
            f"No windows collected. episode_count={len(episodes)}, horizon={horizon}, stride={stride}"
        )

    if len(obs_windows) > max_windows:
        keep = torch.randperm(len(obs_windows))[:max_windows]
        keep = keep.sort().values
        obs_windows = [obs_windows[i] for i in keep]
        action_windows = [action_windows[i] for i in keep]
        reward_windows = [reward_windows[i] for i in keep]
        term_windows = [term_windows[i] for i in keep]
        source_episode = [source_episode[i] for i in keep]
        source_start = [source_start[i] for i in keep]
        source_mode = [source_mode[i] for i in keep]

    return {
        "obs": torch.stack(obs_windows, dim=0).cpu(),
        "action": torch.stack(action_windows, dim=0).cpu(),
        "reward": torch.stack(reward_windows, dim=0).cpu(),
        "term": torch.stack(term_windows, dim=0).cpu(),
        "source_episode": torch.tensor(source_episode, dtype=torch.long),
        "source_start": torch.tensor(source_start, dtype=torch.long),
        "source_mode": torch.tensor(source_mode, dtype=torch.long),
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_output) if args.metadata_output else output_path.with_suffix(".json")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    wandb_run = None
    if not args.disable_wandb:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name or f"phase1_dataset_go1_seed{args.seed}",
            job_type="dataset_collection",
            tags=tags,
            config=vars(args),
        )

    env_cfg = OmegaConf.load(args.env_config)
    env_cfg.config.device = args.device
    env_cfg.config.seed = args.seed
    env_cfg.config.num_envs = args.num_envs
    env_cfg.config.episode_length = args.episode_length

    env = instantiate(env_cfg.config, logdir=str(output_path.parent))
    action_dim = env.num_actions
    obs_dim = env.num_obs
    teacher_agent: Optional[PWM] = None
    if args.action_mode in {"teacher_policy", "mixed_episode"}:
        if not args.teacher_alg_config or not args.teacher_checkpoint:
            raise ValueError(
                "--teacher-alg-config and --teacher-checkpoint are required for teacher_policy/mixed_episode."
            )
        teacher_agent = _build_teacher_agent(
            alg_config=args.teacher_alg_config,
            checkpoint=args.teacher_checkpoint,
            obs_dim=obs_dim,
            act_dim=action_dim,
            device=args.device,
            horizon=args.window_length,
            logdir=str(output_path.parent),
        )
    obs = env.reset(grads=False)
    episodes_in_progress: List[List[TensorDict]] = [
        [_initial_tensordict(obs[i], action_dim, device)] for i in range(env.num_envs)
    ]
    completed: List[TensorDict] = []
    completed_modes: List[str] = []
    mixed_teacher_mask: Optional[torch.Tensor] = None
    if args.action_mode == "mixed_episode":
        mixed_teacher_mask = torch.rand(env.num_envs, generator=torch.Generator().manual_seed(args.seed)) < args.mixed_teacher_prob

    while len(completed) < args.target_episodes:
        if args.action_mode == "random_uniform":
            actions = torch.empty(env.num_envs, action_dim, device=device).uniform_(-1.0, 1.0)
        elif args.action_mode == "zero":
            actions = torch.zeros(env.num_envs, action_dim, device=device)
        elif args.action_mode == "teacher_policy":
            assert teacher_agent is not None
            actions = _teacher_actions(teacher_agent, obs, deterministic=args.teacher_deterministic)
        elif args.action_mode == "mixed_episode":
            assert teacher_agent is not None and mixed_teacher_mask is not None
            teacher_actions = _teacher_actions(teacher_agent, obs, deterministic=args.teacher_deterministic)
            random_actions = torch.empty(env.num_envs, action_dim, device=device).uniform_(-1.0, 1.0)
            actions = torch.where(mixed_teacher_mask.unsqueeze(-1).to(device), teacher_actions, random_actions)
        else:
            raise ValueError(f"Unsupported action mode: {args.action_mode}")

        obs_next, reward, done, info = env.step(actions)
        term = info["termination"]
        trunc = info["truncation"]
        obs_before_reset = info["obs_before_reset"]

        for env_idx in range(env.num_envs):
            episodes_in_progress[env_idx].append(
                _step_tensordict(
                    obs_before_reset[env_idx],
                    actions[env_idx],
                    reward[env_idx],
                    term[env_idx],
                )
            )

        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        for env_idx in done_ids.tolist():
            episode_td = torch.cat(episodes_in_progress[env_idx])
            completed.append(episode_td)
            if args.action_mode == "mixed_episode":
                assert mixed_teacher_mask is not None
                completed_modes.append(
                    "mixed_episode_teacher" if bool(mixed_teacher_mask[env_idx].item()) else "mixed_episode_random"
                )
            else:
                completed_modes.append(args.action_mode)
            episodes_in_progress[env_idx] = [_initial_tensordict(obs_next[env_idx], action_dim, device)]
            if args.action_mode == "mixed_episode":
                assert mixed_teacher_mask is not None
                mixed_teacher_mask[env_idx] = random.random() < args.mixed_teacher_prob

            if len(completed) >= args.target_episodes:
                break

    data = _build_windows(
        completed,
        episode_modes=completed_modes,
        horizon=args.window_length,
        stride=args.window_stride,
        max_windows=args.max_windows,
    )
    torch.save(data, output_path)

    metadata = {
        "env_config": str(args.env_config),
        "device": args.device,
        "seed": args.seed,
        "num_envs": int(args.num_envs),
        "target_episodes": int(args.target_episodes),
        "collected_episodes": int(len(completed)),
        "wm_horizon": int(args.window_length),
        "sequence_length": int(data["obs"].shape[1]),
        "window_stride": int(args.window_stride),
        "max_windows": int(args.max_windows),
        "num_windows": int(data["obs"].shape[0]),
        "obs_dim": int(data["obs"].shape[-1]),
        "act_dim": int(data["action"].shape[-1]),
        "action_mode": str(args.action_mode),
        "teacher_alg_config": args.teacher_alg_config,
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_deterministic": bool(args.teacher_deterministic),
        "mixed_teacher_prob": float(args.mixed_teacher_prob),
        "task_id_requested": str(env_cfg.config.task_id),
        "task_id_resolved": str(getattr(env, "resolved_task_id", env_cfg.config.task_id)),
        "strict_task_id_match": bool(env_cfg.config.get("strict_task_id_match", False)),
        "episode_mode_counts": {
            "random_uniform": int(sum(mode == "random_uniform" for mode in completed_modes)),
            "zero": int(sum(mode == "zero" for mode in completed_modes)),
            "teacher_policy": int(sum(mode == "teacher_policy" for mode in completed_modes)),
            "mixed_episode_teacher": int(sum(mode == "mixed_episode_teacher" for mode in completed_modes)),
            "mixed_episode_random": int(sum(mode == "mixed_episode_random" for mode in completed_modes)),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    if wandb_run is not None:
        wandb.log(
            {
                "dataset/collected_episodes": metadata["collected_episodes"],
                "dataset/num_windows": metadata["num_windows"],
                "dataset/wm_horizon": metadata["wm_horizon"],
                "dataset/sequence_length": metadata["sequence_length"],
                "dataset/obs_dim": metadata["obs_dim"],
                "dataset/act_dim": metadata["act_dim"],
                "dataset/action_mode_id_mean": float(data["source_mode"].float().mean().item()),
            }
        )
        wandb_run.summary.update(metadata)
        wandb_run.finish()

    print(f"Saved dataset to {output_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
