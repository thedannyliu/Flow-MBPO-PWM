#!/usr/bin/env python3
"""Render real-environment rollout videos for completed MJLab-QS policies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.mjlab_qs.collect_mjlab_qs_native_episodes import (
    patch_headless_display_dependency,
    patch_mujoco_compatibility,
    split_obs,
    tensor_from_actor_obs,
)
from scripts.experiments.mjlab_qs.run_offline_pwm_policy_extraction import (
    Actor,
    FlowActor,
    load_data,
    norm,
    parse_units,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rollout-episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--require-mp4", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--wandb-name", default="")
    p.add_argument("--disable-wandb", action="store_true")
    return p.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_actor(ckpt: dict[str, Any], state_dim: int, command_dim: int, action_dim: int, device: torch.device):
    ckpt_args = ckpt["args"]
    if ckpt_args.get("policy_type", "mlp") == "mlp":
        actor = Actor(
            state_dim,
            command_dim,
            action_dim,
            units=parse_units(ckpt_args.get("actor_units", "400,200,100")),
            init_logstd=float(ckpt_args.get("init_logstd", -1.0)),
            min_logstd=float(ckpt_args.get("min_logstd", -1.427)),
        )
    else:
        actor = FlowActor(
            state_dim,
            command_dim,
            action_dim,
            units=parse_units(ckpt_args.get("actor_units", "400,200,100")),
            flow_substeps=int(ckpt_args.get("flow_policy_substeps", 2)),
            flow_integrator=ckpt_args.get("flow_policy_integrator", "heun"),
        )
    actor.load_state_dict(ckpt["actor"])
    actor.to(device)
    actor.eval()
    return actor


def build_render_env(ckpt_args: dict[str, Any], device: str):
    patch_mujoco_compatibility()
    patch_headless_display_dependency()
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    task_id = ckpt_args.get("task_id", "Mjlab-Velocity-Flat-Unitree-G1")
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(ckpt_args.get("seed", 0)) + 20000
    if hasattr(env_cfg, "episode_length_s") and hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "mujoco"):
        env_dt = float(env_cfg.sim.mujoco.timestep) * float(env_cfg.decimation)
        env_cfg.episode_length_s = float(ckpt_args.get("episode_length", 1000)) * env_dt
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    obs_td = wrapped.get_observations()
    obs_groups = list(agent_cfg.obs_groups["actor"])
    return wrapped, base_env, obs_td, obs_groups


def frame_from_render(env, base_env):
    for candidate_env in (env, base_env):
        render_fn = getattr(candidate_env, "render", None)
        if not callable(render_fn):
            continue
        try:
            frame = render_fn(mode="rgb_array")
        except TypeError:
            frame = render_fn()
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if isinstance(frame, (list, tuple)) and frame:
            frame = frame[0]
        if isinstance(frame, dict):
            for key in ("rgb", "frame", "image"):
                if key in frame:
                    frame = frame[key]
                    break
        if hasattr(frame, "shape"):
            return frame
    return None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect_rollout(actor, ckpt_args: dict[str, Any], nrm: dict[str, torch.Tensor], args: argparse.Namespace):
    device = torch.device(args.device)
    env, base_env, obs_td, obs_groups = build_render_env(ckpt_args, args.device)
    frames = []
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    ep_return = torch.zeros(1, device=device)
    ep_len = torch.zeros(1, dtype=torch.long, device=device)
    episodes_done = 0
    command_dim = int(ckpt_args.get("command_dim", 3))
    command_position = ckpt_args.get("command_position", "tail")
    try:
        while episodes_done < args.rollout_episodes:
            frame = frame_from_render(env, base_env)
            if frame is not None:
                frames.append(frame)
            obs = tensor_from_actor_obs(obs_td, obs_groups)
            phys, cmd = split_obs(obs, command_dim, command_position)
            z = norm(phys.float(), nrm["phys_obs_mean"], nrm["phys_obs_std"])
            c = cmd.float()
            if c.shape[-1] and "command_mean" in nrm:
                c = norm(c, nrm["command_mean"], nrm["command_std"])
            action = actor(z, c, deterministic=True).clamp(-1.0, 1.0)
            next_obs_td, reward, done, _extras = env.step(action)
            reward = reward.to(device).float().reshape(-1)
            done = done.to(device).bool().reshape(-1)
            ep_return += reward[:1]
            ep_len += 1
            step_rows.append(
                {
                    "episode_slot": episodes_done,
                    "step": int(ep_len[0].item()),
                    "reward": float(reward[0].item()),
                    "action_l2": float(action[0].pow(2).mean().sqrt().item()),
                    "done": int(done[0].item()),
                }
            )
            if bool(done[0].item()) or int(ep_len[0].item()) >= args.max_steps:
                episode_rows.append(
                    {
                        "episode": episodes_done,
                        "return": float(ep_return[0].item()),
                        "length": int(ep_len[0].item()),
                        "terminated": int(done[0].item()),
                    }
                )
                episodes_done += 1
                ep_return.zero_()
                ep_len.zero_()
            obs_td = next_obs_td
    finally:
        env.close()
    return frames, step_rows, episode_rows


def write_video(frames: list[Any], path: Path, fps: int, require_mp4: bool) -> None:
    if not frames:
        raise RuntimeError("No render frames were captured; cannot write rollout video.")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, frames, fps=fps, plugin="ffmpeg")
    except Exception as exc:
        if require_mp4:
            raise RuntimeError(f"MP4 export failed for {path}: {exc}") from exc
        gif_path = path.with_suffix(".gif")
        import imageio.v2 as iio_v2

        iio_v2.mimsave(str(gif_path), frames, format="GIF", fps=fps)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.policy_checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    data, _metadata, nrm, _train_idx = load_data(argparse.Namespace(**ckpt_args), device)
    actor = build_actor(
        ckpt,
        state_dim=int(data["phys_obs"].shape[-1]),
        command_dim=int(data["command"].shape[-1]),
        action_dim=int(data["policy_action"].shape[-1]),
        device=device,
    )
    t0 = time.time()
    run = None
    if args.wandb_project and not args.disable_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group or ckpt_args.get("wandb_group", ""),
            name=args.wandb_name or f"{ckpt_args.get('wm_method')}_{ckpt_args.get('policy_type')}_seed{ckpt_args.get('seed')}_rollout",
            job_type="policy_rollout_video",
            config={
                "policy_checkpoint": args.policy_checkpoint,
                "rollout_episodes": args.rollout_episodes,
                "max_steps": args.max_steps,
                "video_fps": args.video_fps,
                "git_sha": git_sha(),
                **ckpt_args,
            },
        )
    frames, step_rows, episode_rows = collect_rollout(actor, ckpt_args, nrm, args)
    video_path = output_dir / "rollout.mp4"
    write_video(frames, video_path, args.video_fps, args.require_mp4)
    write_csv(output_dir / "rollout_steps.csv", step_rows, ["episode_slot", "step", "reward", "action_l2", "done"])
    write_csv(output_dir / "rollout_summary.csv", episode_rows, ["episode", "return", "length", "terminated"])
    returns = torch.tensor([row["return"] for row in episode_rows], dtype=torch.float32)
    lengths = torch.tensor([row["length"] for row in episode_rows], dtype=torch.float32)
    summary = {
        "policy_checkpoint": args.policy_checkpoint,
        "video": str(video_path),
        "num_frames": len(frames),
        "num_episodes": len(episode_rows),
        "return_mean": float(returns.mean().item()) if returns.numel() else None,
        "return_std": float(returns.std(unbiased=False).item()) if returns.numel() else None,
        "episode_length_mean": float(lengths.mean().item()) if lengths.numel() else None,
        "wall_clock_seconds": time.time() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if run is not None:
        import wandb

        run.log(
            {
                "rollout/return_mean": summary["return_mean"],
                "rollout/episode_length_mean": summary["episode_length_mean"],
                "rollout/video": wandb.Video(str(video_path), fps=args.video_fps, format="mp4"),
            }
        )
        run.summary.update(summary)
        run.finish()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
