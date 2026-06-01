#!/usr/bin/env python3
"""Evaluate a single-task online RL checkpoint and log rich metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Force an offscreen backend by default for PACE-ICE headless nodes.
os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")


def instantiate_eval_env(cfg, device: str, num_envs: int, enable_render: bool = False):
    env_cfg_dict = OmegaConf.to_container(cfg.env.config, resolve=True)
    if not isinstance(env_cfg_dict, dict):
        raise TypeError("Expected resolved env config to be a dictionary.")

    if enable_render:
        target = str(env_cfg_dict.get("_target_", ""))
        if target.endswith("create_mjlab_pwm_env"):
            mjlab_kwargs = env_cfg_dict.get("mjlab_env_kwargs") or {}
            if not isinstance(mjlab_kwargs, dict):
                mjlab_kwargs = dict(mjlab_kwargs)
            mjlab_kwargs.setdefault("render_mode", "rgb_array")
            env_cfg_dict["mjlab_env_kwargs"] = mjlab_kwargs
        if target.endswith("create_gymnasium_mujoco_pwm_env"):
            env_cfg_dict["render_mode"] = "rgb_array"

    env_cfg = OmegaConf.create(env_cfg_dict)
    env_cfg.device = device
    env_cfg.num_envs = num_envs
    if "no_grad" in env_cfg:
        env_cfg.no_grad = True
    return instantiate(env_cfg, logdir=str(PROJECT_ROOT / "logs" / "eval"))


def iqm(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    sorted_vals = np.sort(values)
    lo = int(0.25 * len(sorted_vals))
    hi = int(0.75 * len(sorted_vals))
    if hi <= lo:
        return float(np.mean(sorted_vals))
    return float(np.mean(sorted_vals[lo:hi]))


def extract_success(info: Dict[str, torch.Tensor], idx: int) -> float:
    for key in ("success", "is_success", "task_success"):
        if key not in info:
            continue
        raw = info[key]
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw)
        flat = raw.reshape(-1)
        if idx < flat.shape[0]:
            return float(flat[idx].item())
    return float("nan")


def load_models(checkpoint_path: Path, cfg, device: str):
    alg = cfg.alg
    latent_dim = alg.get("latent_dim", 512)
    env_tmp = instantiate_eval_env(cfg, device=device, num_envs=1, enable_render=False)
    obs_dim = getattr(env_tmp, "num_obs", env_tmp.observation_space.shape[0])
    act_dim = getattr(env_tmp, "num_actions", env_tmp.action_space.shape[0])
    if hasattr(env_tmp, "close"):
        env_tmp.close()

    world_model = instantiate(
        alg.world_model_config,
        observation_dim=obs_dim,
        action_dim=act_dim,
        latent_dim=latent_dim,
        _recursive_=True,
    ).to(device)
    actor = instantiate(
        alg.actor_config,
        obs_dim=latent_dim,
        action_dim=act_dim,
        _recursive_=True,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    world_model.load_state_dict(checkpoint["world_model"])
    actor.load_state_dict(checkpoint["actor"])
    world_model.eval()
    actor.eval()
    return actor, world_model


@torch.no_grad()
def evaluate_policy(
    actor,
    world_model,
    env,
    num_games: int,
    deterministic: bool,
    gamma: float,
):
    num_envs = int(env.num_envs)
    games_played = 0
    device = next(world_model.parameters()).device

    ep_reward = torch.zeros(num_envs, device=device)
    ep_discounted_reward = torch.zeros_like(ep_reward)
    ep_len = torch.zeros(num_envs, dtype=torch.int64, device=ep_reward.device)
    ep_gamma = torch.ones(num_envs, device=ep_reward.device)

    episode_rows: List[Dict[str, float]] = []

    obs = env.reset()
    while games_played < num_games:
        z = world_model.encode(obs, task=None)
        action = actor(z, deterministic=deterministic)
        action = torch.tanh(action)
        obs, reward, done, info = env.step(action)

        reward = reward.reshape(-1).float()
        done = done.reshape(-1).bool()
        ep_reward += reward
        ep_discounted_reward += ep_gamma * reward
        ep_len += 1
        ep_gamma *= gamma

        done_indices = done.nonzero(as_tuple=False).squeeze(-1)
        for idx_t in done_indices:
            idx = int(idx_t.item())
            if games_played >= num_games:
                break
            episode_rows.append(
                {
                    "episode_id": games_played,
                    "return": float(ep_reward[idx].item()),
                    "discounted_return": float(ep_discounted_reward[idx].item()),
                    "length": float(ep_len[idx].item()),
                    "success": extract_success(info, idx),
                }
            )
            games_played += 1
            ep_reward[idx] = 0.0
            ep_discounted_reward[idx] = 0.0
            ep_len[idx] = 0
            ep_gamma[idx] = 1.0

    returns = np.array([row["return"] for row in episode_rows], dtype=np.float64)
    discounted_returns = np.array(
        [row["discounted_return"] for row in episode_rows], dtype=np.float64
    )
    lengths = np.array([row["length"] for row in episode_rows], dtype=np.float64)
    success = np.array([row["success"] for row in episode_rows], dtype=np.float64)
    finite_success = success[np.isfinite(success)]

    summary = {
        "num_episodes": float(len(episode_rows)),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_median": float(np.median(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "return_p25": float(np.percentile(returns, 25)),
        "return_p75": float(np.percentile(returns, 75)),
        "return_iqm": iqm(returns),
        "discounted_return_mean": float(np.mean(discounted_returns)),
        "episode_length_mean": float(np.mean(lengths)),
        "episode_length_std": float(np.std(lengths)),
        "success_rate": float(np.mean(finite_success)) if finite_success.size > 0 else float("nan"),
    }
    return summary, episode_rows


@torch.no_grad()
def collect_rollouts(
    actor,
    world_model,
    env,
    num_episodes: int,
    max_steps: int,
    deterministic: bool,
    save_video: bool,
    video_path: Path,
    video_fps: int,
    require_mp4: bool = False,
    allow_gif_fallback: bool = True,
):
    rollout_rows: List[Dict[str, float]] = []
    episode_summaries: List[Dict[str, float]] = []
    frames: List[np.ndarray] = []
    obs = env.reset()
    device = next(world_model.parameters()).device
    ep_return = torch.zeros(env.num_envs, device=device)
    ep_len = torch.zeros(env.num_envs, dtype=torch.int64, device=ep_return.device)
    episodes_done = 0

    def _to_numpy_frame(frame_candidate):
        if frame_candidate is None:
            return None
        if isinstance(frame_candidate, torch.Tensor):
            frame_candidate = frame_candidate.detach().cpu().numpy()
        if isinstance(frame_candidate, (list, tuple)) and len(frame_candidate) > 0:
            frame_candidate = frame_candidate[0]
        if isinstance(frame_candidate, dict):
            for key in ("rgb", "frame", "image"):
                if key in frame_candidate:
                    frame_candidate = frame_candidate[key]
                    break
        if not isinstance(frame_candidate, np.ndarray):
            return None
        return frame_candidate

    def _capture_frame():
        if not save_video:
            return
        render_fn = getattr(env, "render", None)
        if not callable(render_fn):
            return
        frame_candidate = None
        try:
            frame_candidate = render_fn(mode="rgb_array")
        except TypeError:
            try:
                frame_candidate = render_fn()
            except Exception:
                return
        except Exception:
            return
        frame = _to_numpy_frame(frame_candidate)
        if frame is not None:
            frames.append(frame)

    while episodes_done < num_episodes:
        _capture_frame()
        z = world_model.encode(obs, task=None)
        action = actor(z, deterministic=deterministic)
        action_tanh = torch.tanh(action)
        next_obs, reward, done, _ = env.step(action_tanh)
        reward = reward.reshape(-1).float()
        done = done.reshape(-1).bool()
        ep_return += reward
        ep_len += 1

        obs_l2 = torch.norm(obs, p=2, dim=-1)
        act_l2 = torch.norm(action_tanh, p=2, dim=-1)
        for idx in range(obs.shape[0]):
            rollout_rows.append(
                {
                    "episode_slot": float(episodes_done),
                    "env_index": float(idx),
                    "step": float(ep_len[idx].item()),
                    "reward": float(reward[idx].item()),
                    "obs_l2": float(obs_l2[idx].item()),
                    "action_l2": float(act_l2[idx].item()),
                    "done": float(done[idx].item()),
                }
            )

        done_indices = done.nonzero(as_tuple=False).squeeze(-1)
        for idx_t in done_indices:
            idx = int(idx_t.item())
            episode_summaries.append(
                {
                    "rollout_episode_id": float(episodes_done),
                    "env_index": float(idx),
                    "return": float(ep_return[idx].item()),
                    "length": float(ep_len[idx].item()),
                }
            )
            episodes_done += 1
            ep_return[idx] = 0.0
            ep_len[idx] = 0
            if episodes_done >= num_episodes:
                break

        obs = next_obs
        if int(ep_len.max().item()) >= max_steps:
            for idx in range(env.num_envs):
                episode_summaries.append(
                    {
                        "rollout_episode_id": float(episodes_done),
                        "env_index": float(idx),
                        "return": float(ep_return[idx].item()),
                        "length": float(ep_len[idx].item()),
                    }
                )
                episodes_done += 1
                ep_return[idx] = 0.0
                ep_len[idx] = 0
                if episodes_done >= num_episodes:
                    break

    if save_video and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        mp4_written = False
        try:
            import imageio.v3 as iio

            # Prefer ffmpeg backend explicitly for stable MP4 output on clusters.
            iio.imwrite(video_path, frames, fps=video_fps, plugin="ffmpeg")
            mp4_written = True
        except Exception as ffmpeg_exc:
            print(f"MP4 export with ffmpeg plugin failed: {ffmpeg_exc}")
            try:
                import imageio.v3 as iio

                iio.imwrite(video_path, frames, fps=video_fps)
                mp4_written = True
            except Exception as generic_exc:
                print(f"Generic MP4 export skipped: {generic_exc}")

        if not mp4_written and allow_gif_fallback:
            # Fallback for environments without working MP4 backend.
            try:
                import imageio.v2 as iio_v2

                gif_path = video_path.with_suffix(".gif")
                iio_v2.mimsave(str(gif_path), frames, format="GIF", fps=video_fps)
                print(f"Saved GIF rollout fallback: {gif_path}")
            except Exception as gif_exc:
                print(f"GIF video export skipped: {gif_exc}")

        if require_mp4 and not video_path.exists():
            raise RuntimeError(
                f"require_mp4=True but MP4 file was not produced at {video_path}. "
                "Install imageio-ffmpeg (or pyav) in the runtime environment."
            )

    return rollout_rows, episode_summaries


def maybe_log_to_wandb(
    args,
    summary: Dict[str, float],
    output_dir: Path,
    metadata: Dict[str, str],
) -> None:
    if not args.wandb_project:
        return

    try:
        import wandb  # Local import to keep script usable without wandb installed.
    except Exception as exc:
        print(f"WandB logging skipped: {exc}")
        return

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    for key in ("stage", "suite", "task", "method"):
        value = str(metadata.get(key, "")).strip()
        if value:
            tag = f"{key}_{value}"
            if tag not in tags:
                tags.append(tag)
    run_key = str(metadata.get("run_key", "")).strip()
    if run_key:
        seed_match = re.search(r"_s(\d+)_", run_key)
        if seed_match:
            seed_tag = f"seed_{seed_match.group(1)}"
            if seed_tag not in tags:
                tags.append(seed_tag)
    if "purpose_eval" not in tags:
        tags.append("purpose_eval")
    job_type = (args.wandb_job_type or "eval").strip() or "eval"
    job_tag = f"job_{job_type}"
    if job_tag not in tags:
        tags.append(job_tag)

    run_id = (args.wandb_id or "").strip()
    resume_mode = (args.wandb_resume or "").strip()
    init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group or None,
        name=args.wandb_name or None,
        tags=tags or None,
        job_type=job_type,
        config=metadata,
    )
    if run_id:
        init_kwargs["id"] = run_id
    if resume_mode:
        init_kwargs["resume"] = resume_mode
    run = wandb.init(**init_kwargs)
    wandb.log({f"eval/{k}": v for k, v in summary.items()})

    rollout_video_mp4 = output_dir / "rollout.mp4"
    rollout_video_gif = output_dir / "rollout.gif"
    rollout_video_path = rollout_video_mp4 if rollout_video_mp4.exists() else rollout_video_gif
    if args.wandb_log_video and rollout_video_path.exists():
        video_format = "mp4" if rollout_video_path.suffix.lower() == ".mp4" else "gif"
        try:
            wandb.log(
                {
                    "eval/rollout_video": wandb.Video(
                        str(rollout_video_path), fps=args.video_fps, format=video_format
                    )
                }
            )
        except Exception as exc:
            print(f"WandB video logging skipped: {exc}")

    if args.wandb_log_artifact:
        artifact_name = metadata.get("run_key") or metadata["run_name"]
        artifact = wandb.Artifact(
            name=f"eval-{artifact_name}",
            type="evaluation",
            metadata=summary,
        )
        for filename in (
            "eval_summary.json",
            "episode_metrics.csv",
            "rollout_steps.csv",
            "rollout_summary.csv",
        ):
            filepath = output_dir / filename
            if filepath.exists():
                artifact.add_file(str(filepath))
        # Video files can be large; upload them only when explicitly enabled.
        if args.wandb_log_video:
            for filename in ("rollout.mp4", "rollout.gif"):
                filepath = output_dir / filename
                if filepath.exists():
                    artifact.add_file(str(filepath))
        wandb.log_artifact(artifact)
    run.finish()


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_hydra_run_dir(checkpoint: Path) -> tuple[Path, Path]:
    for parent in checkpoint.parents:
        config_path = parent / ".hydra" / "config.yaml"
        if config_path.exists():
            return parent, config_path
    raise FileNotFoundError(
        f"Missing Hydra config in parents of checkpoint path {checkpoint}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate single-task online checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Hydra config to use when the checkpoint is not inside a Hydra run directory.",
    )
    parser.add_argument("--num-games", type=int, default=40)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions during evaluation/rollout.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-episodes", type=int, default=3)
    parser.add_argument("--rollout-max-steps", type=int, default=1000)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--require-mp4",
        action="store_true",
        help="Fail evaluation if MP4 video cannot be generated.",
    )
    parser.add_argument(
        "--allow-gif-fallback",
        action="store_true",
        help="Allow GIF fallback when MP4 backend is unavailable.",
    )
    parser.add_argument(
        "--wandb-log-video",
        dest="wandb_log_video",
        action="store_true",
        help="Upload rollout video to W&B (default: enabled).",
    )
    parser.add_argument(
        "--no-wandb-log-video",
        dest="wandb_log_video",
        action="store_false",
        help="Disable rollout video upload to W&B.",
    )
    parser.add_argument(
        "--wandb-log-artifact",
        dest="wandb_log_artifact",
        action="store_true",
        help="Upload eval artifact files to W&B (default: enabled).",
    )
    parser.add_argument(
        "--no-wandb-log-artifact",
        dest="wandb_log_artifact",
        action="store_false",
        help="Disable eval artifact upload to W&B.",
    )
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-job-type", default="eval")
    parser.add_argument("--wandb-id", default="")
    parser.add_argument("--wandb-resume", default="")
    parser.set_defaults(
        wandb_log_video=True,
        wandb_log_artifact=True,
        allow_gif_fallback=True,
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if args.config_path:
        config_path = args.config_path.resolve()
        run_dir = config_path.parent.parent if config_path.parent.name == ".hydra" else config_path.parent
    else:
        run_dir, config_path = find_hydra_run_dir(checkpoint)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Hydra config at {config_path}")

    cfg = OmegaConf.load(config_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actor, world_model = load_models(checkpoint, cfg, args.device)
    env = instantiate_eval_env(cfg, device=args.device, num_envs=args.num_envs, enable_render=False)
    summary, episode_rows = evaluate_policy(
        actor=actor,
        world_model=world_model,
        env=env,
        num_games=args.num_games,
        deterministic=not args.stochastic,
        gamma=float(cfg.alg.get("gamma", 0.99)),
    )
    if hasattr(env, "close"):
        env.close()

    rollout_rows: List[Dict[str, float]] = []
    rollout_summary: List[Dict[str, float]] = []
    if args.rollout_episodes > 0:
        rollout_env = instantiate_eval_env(
            cfg,
            device=args.device,
            num_envs=1,
            enable_render=args.save_video,
        )
        rollout_rows, rollout_summary = collect_rollouts(
            actor=actor,
            world_model=world_model,
            env=rollout_env,
            num_episodes=args.rollout_episodes,
            max_steps=args.rollout_max_steps,
            deterministic=not args.stochastic,
            save_video=args.save_video,
            video_path=args.output_dir / "rollout.mp4",
            video_fps=args.video_fps,
            require_mp4=args.require_mp4,
            allow_gif_fallback=args.allow_gif_fallback,
        )
        if hasattr(rollout_env, "close"):
            rollout_env.close()

    summary["checkpoint"] = str(checkpoint)
    summary["env_target"] = str(cfg.env.config.get("_target_", ""))
    summary["task_id"] = str(cfg.env.config.get("task_id", ""))
    summary["requested_task_id"] = str(getattr(env, "requested_task_id", summary["task_id"]))
    summary["resolved_task_id"] = str(getattr(env, "resolved_task_id", summary["task_id"]))
    summary["seed"] = float(cfg.general.get("seed", 0))
    summary["num_games_requested"] = float(args.num_games)
    summary["num_envs_eval"] = float(args.num_envs)

    with (args.output_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_csv(
        args.output_dir / "episode_metrics.csv",
        episode_rows,
        fieldnames=["episode_id", "return", "discounted_return", "length", "success"],
    )
    if rollout_rows:
        write_csv(
            args.output_dir / "rollout_steps.csv",
            rollout_rows,
            fieldnames=["episode_slot", "env_index", "step", "reward", "obs_l2", "action_l2", "done"],
        )
    if rollout_summary:
        write_csv(
            args.output_dir / "rollout_summary.csv",
            rollout_summary,
            fieldnames=["rollout_episode_id", "env_index", "return", "length"],
        )

    metadata = {
        "checkpoint": str(checkpoint),
        "run_name": run_dir.name,
        "config_path": str(config_path),
        "run_key": str(cfg.get("experiment", {}).get("run_key", "")),
        "stage": str(cfg.get("experiment", {}).get("stage", "")),
        "suite": str(cfg.get("experiment", {}).get("suite", "")),
        "task": str(cfg.get("experiment", {}).get("task", "")),
        "method": str(cfg.get("experiment", {}).get("method", "")),
    }
    maybe_log_to_wandb(args, summary, args.output_dir, metadata)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
