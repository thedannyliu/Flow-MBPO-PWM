#!/usr/bin/env python3
"""Run the original PWM algorithm on MJLab-QS windows through a thin adapter.

The upstream PWM entrypoints only support DFlex single-task environments or
TD-MPC2 multitask datasets.  MJLab-QS stores fixed H-step windows with
`phys_obs`, `command`, `policy_action`, and `reward`, so the original scripts
cannot consume it directly.

This runner keeps the original PWM implementation intact and adapts only the
data/evaluation boundary:

* original `pwm.algorithms.pwm.PWM`
* original stochastic MLP actor, critic ensemble, SimNorm world model
* original `compute_wm_loss`, `update`, TD(lambda), ret RMS, LR schedule
* MJLab-QS windows are packed as `obs = [normalized phys_obs, normalized command]`

This is therefore an "original PWM algorithm adapter", not a byte-identical
execution of upstream `train_dflex.py` or `train_multitask.py`.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import wandb
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PWM_SRC = PROJECT_ROOT / "baselines" / "PWM" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PWM_SRC) not in sys.path:
    sys.path.insert(0, str(PWM_SRC))

from pwm.algorithms.pwm import PWM  # noqa: E402

from scripts.experiments.mjlab_qs.collect_mjlab_qs_native_episodes import (  # noqa: E402
    patch_headless_display_dependency,
    patch_mujoco_compatibility,
    split_obs,
    tensor_from_actor_obs,
)
from scripts.experiments.mjlab_qs.run_phaseA_wm_feasibility import (  # noqa: E402
    load_norm,
    norm,
    sample_train_indices,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task-id", default="Mjlab-Velocity-Flat-Unitree-G1")
    p.add_argument("--pretrain-iters", type=int, default=50000)
    p.add_argument("--policy-iters", type=int, default=15000)
    p.add_argument("--wm-batch-size", type=int, default=256)
    p.add_argument("--policy-batch-size", type=int, default=64)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--actor-lr", type=float, default=5e-4)
    p.add_argument("--critic-lr", type=float, default=5e-4)
    p.add_argument("--model-lr", type=float, default=3e-4)
    p.add_argument("--critic-iterations", type=int, default=8)
    p.add_argument("--critic-batches", type=int, default=4)
    p.add_argument("--num-critics", type=int, default=3)
    p.add_argument("--latent-dim", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--pretrain-log-every", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--eval-num-envs", type=int, default=16)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--command-dim", type=int, default=3)
    p.add_argument("--command-position", choices=["tail", "head", "none"], default="tail")
    p.add_argument("--obs-mode", choices=["normalized", "raw"], default="normalized")
    p.add_argument("--reward-mode", choices=["normalized", "raw"], default="normalized")
    p.add_argument("--rew-rms", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ret-rms", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-real-eval", action="store_true")
    p.add_argument("--wandb-project", default="flow-mbpo-mjlab-original-pwm-adapter")
    p.add_argument("--wandb-group", default="g1_original_pwm_adapter")
    p.add_argument("--wandb-name", default="")
    p.add_argument("--disable-wandb", action="store_true")
    return p.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_data(args: argparse.Namespace, device: torch.device):
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    nrm = load_norm(Path(args.normalization), device)
    train_idx = (data["split_id"] == 0).nonzero(as_tuple=False).squeeze(-1)
    val_idx = (data["split_id"] == 1).nonzero(as_tuple=False).squeeze(-1)
    test_idx = (data["split_id"] == 2).nonzero(as_tuple=False).squeeze(-1)
    return data, metadata, nrm, train_idx, val_idx, test_idx


def pack_obs(
    phys: torch.Tensor,
    command: torch.Tensor,
    nrm: Dict[str, torch.Tensor],
    obs_mode: str,
) -> torch.Tensor:
    if obs_mode == "normalized":
        phys = norm(phys.float(), nrm["phys_obs_mean"], nrm["phys_obs_std"])
        if command.shape[-1] and "command_mean" in nrm:
            command = norm(command.float(), nrm["command_mean"], nrm["command_std"])
        else:
            command = command.float()
    else:
        phys = phys.float()
        command = command.float()
    return torch.cat([phys, command], dim=-1)


def pack_reward(reward: torch.Tensor, nrm: Dict[str, torch.Tensor], reward_mode: str) -> torch.Tensor:
    reward = reward.float()
    if reward_mode == "normalized":
        reward = norm(reward[..., None], nrm["reward_mean"], nrm["reward_std"]).squeeze(-1)
    return reward


def batch_windows(
    data: Dict[str, torch.Tensor],
    ids: torch.Tensor,
    device: torch.device,
    nrm: Dict[str, torch.Tensor],
    args: argparse.Namespace,
):
    phys = data["phys_obs"][ids].to(device)
    command = data["command"][ids].to(device)
    # MJLab-QS stores observations as H+1 endpoints and commands/actions/rewards
    # as H transition-aligned values.  Original PWM expects an observation at
    # every endpoint, so carry the last command to the final endpoint.
    command_obs = torch.cat([command, command[:, -1:].clone()], dim=1)
    obs = pack_obs(phys, command_obs, nrm, args.obs_mode).permute(1, 0, 2).contiguous()
    act = data["policy_action"][ids].to(device).float().permute(1, 0, 2).contiguous()
    rew = pack_reward(data["reward"][ids].to(device), nrm, args.reward_mode)
    rew = rew.permute(1, 0).unsqueeze(-1).contiguous()
    return obs, act, rew


def build_pwm_agent(args: argparse.Namespace, obs_dim: int, action_dim: int) -> PWM:
    actor_config = OmegaConf.create(
        {
            "_target_": "pwm.models.actor.ActorStochasticMLP",
            "units": [400, 200, 100],
            "activation_class": "nn.Mish",
            "init_gain": 1.0,
            "init_logstd": -1.0,
            "min_logstd": -1.427,
        }
    )
    critic_config = OmegaConf.create(
        {
            "_target_": "pwm.models.critic.CriticMLP",
            "units": [400, 200],
            "activation_class": "nn.Mish",
        }
    )
    world_model_config = OmegaConf.create(
        {
            "_target_": "pwm.models.world_model.WorldModel",
            "units": [512, 512],
            "encoder_units": [256],
            "num_bins": None,
            "vmin": None,
            "vmax": None,
            "multitask": False,
            "task_dim": 0,
            "encoder": {
                "last_layer": "normedlinear",
                "last_layer_kwargs": {
                    "act": {"_target_": "pwm.models.mlp.SimNorm", "simnorm_dim": 8}
                },
            },
            "dynamics": {
                "last_layer": "normedlinear",
                "last_layer_kwargs": {
                    "act": {"_target_": "pwm.models.mlp.SimNorm", "simnorm_dim": 8}
                },
            },
            "reward": {"last_layer": "linear", "last_layer_kwargs": {}},
        }
    )
    return PWM(
        env=None,
        actor_config=actor_config,
        critic_config=critic_config,
        world_model_config=world_model_config,
        horizon=args.horizon,
        max_epochs=args.policy_iters,
        logdir=args.output_dir,
        latent_dim=args.latent_dim,
        obs_dim=obs_dim,
        act_dim=action_dim,
        actor_grad_norm=1.0,
        critic_grad_norm=100.0,
        num_critics=args.num_critics,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        model_lr=args.model_lr,
        gamma=args.gamma,
        lam=args.lam,
        obs_rms=False,
        rew_rms=args.rew_rms,
        ret_rms=args.ret_rms,
        critic_iterations=args.critic_iterations,
        critic_batches=args.critic_batches,
        wm_batch_size=args.wm_batch_size,
        wm_iterations=8,
        wm_grad_norm=20.0,
        wm_buffer_size=1_000_000,
        save_interval=max(1, args.policy_iters + 1),
        device=args.device,
        log=False,
        detach=True,
    )


@torch.no_grad()
def eval_wm(agent: PWM, data, idx: torch.Tensor, nrm, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    if idx.numel() == 0:
        return {"wm_loss": float("nan"), "dyn_loss": float("nan"), "rew_loss": float("nan")}
    ids = idx[torch.randint(0, idx.numel(), (min(args.wm_batch_size, idx.numel()),))]
    obs, act, rew = batch_windows(data, ids, device, nrm, args)
    loss, dyn, rew_loss = agent.compute_wm_loss(obs, act, rew, task=None)
    return {"wm_loss": float(loss.detach().item()), "dyn_loss": float(dyn.detach().item()), "rew_loss": float(rew_loss)}


def build_eval_env(args: argparse.Namespace):
    patch_mujoco_compatibility()
    patch_headless_display_dependency()
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    env_cfg = load_env_cfg(args.task_id, play=True)
    agent_cfg = load_rl_cfg(args.task_id)
    env_cfg.scene.num_envs = int(args.eval_num_envs)
    env_cfg.seed = int(args.seed) + 20000
    if hasattr(env_cfg, "episode_length_s") and hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "mujoco"):
        env_dt = float(env_cfg.sim.mujoco.timestep) * float(env_cfg.decimation)
        env_cfg.episode_length_s = float(args.episode_length) * env_dt
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    obs_td = wrapped.get_observations()
    obs_groups = list(agent_cfg.obs_groups["actor"])
    return wrapped, obs_td, obs_groups


@torch.no_grad()
def real_env_eval(agent: PWM, args: argparse.Namespace, nrm: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, float | str]:
    env, obs_td, obs_groups = build_eval_env(args)
    returns = torch.zeros(args.eval_num_envs, device=device)
    lengths = torch.zeros(args.eval_num_envs, device=device)
    done_returns: List[float] = []
    done_lengths: List[float] = []
    while len(done_returns) < args.eval_episodes:
        actor_obs = tensor_from_actor_obs(obs_td, obs_groups)
        phys, cmd = split_obs(actor_obs, args.command_dim, args.command_position)
        packed = pack_obs(phys.to(device), cmd.to(device), nrm, args.obs_mode)
        z = agent.wm.encode(packed, task=None)
        action = torch.tanh(agent.actor(z, deterministic=True)).clamp(-1.0, 1.0)
        next_obs_td, reward, done, _extras = env.step(action)
        reward = reward.to(device).float().reshape(-1)
        done = done.to(device).bool().reshape(-1)
        returns = returns + reward
        lengths = lengths + 1.0
        for idx in done.nonzero(as_tuple=False).reshape(-1).tolist():
            done_returns.append(float(returns[idx].item()))
            done_lengths.append(float(lengths[idx].item()))
            returns[idx] = 0.0
            lengths[idx] = 0.0
            if len(done_returns) >= args.eval_episodes:
                break
        obs_td = next_obs_td
    env.close()
    ret = torch.tensor(done_returns[: args.eval_episodes])
    lens = torch.tensor(done_lengths[: args.eval_episodes])
    return {
        "task_id": args.task_id,
        "resolved_task_id": args.task_id,
        "return_mean": float(ret.mean().item()),
        "return_std": float(ret.std(unbiased=False).item()),
        "episode_length_mean": float(lens.float().mean().item()),
        "episode_length_std": float(lens.float().std(unbiased=False).item()),
        "num_episodes": float(args.eval_episodes),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, metadata, nrm, train_idx, val_idx, test_idx = load_data(args, device)
    obs_dim = int(data["phys_obs"].shape[-1]) + int(data["command"].shape[-1])
    action_dim = int(data["policy_action"].shape[-1])
    agent = build_pwm_agent(args, obs_dim, action_dim)

    run = None
    if not args.disable_wandb:
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name or f"original_pwm_adapter_seed{args.seed}",
            job_type="original_pwm_adapter",
            config={**vars(args), "dataset_metadata": metadata, "git_sha": git_sha()},
        )

    t0 = time.time()
    for it in range(1, args.pretrain_iters + 1):
        ids = sample_train_indices(data, train_idx, args.wm_batch_size)
        obs, act, rew = batch_windows(data, ids, device, nrm, args)
        if agent.rew_rms:
            agent.rew_rms.update(rew.reshape(-1, 1))
            rew = agent.rew_rms.normalize(rew)
        agent.wm_optimizer.zero_grad(set_to_none=True)
        loss, dyn_loss, rew_loss = agent.compute_wm_loss(obs, act, rew, task=None)
        loss.backward()
        wm_grad = torch.nn.utils.clip_grad_norm_(agent.wm.parameters(), agent.wm_grad_norm)
        agent.wm_optimizer.step()
        if it == 1 or it == args.pretrain_iters or it % args.pretrain_log_every == 0:
            val_metrics = eval_wm(agent, data, val_idx, nrm, args, device)
            metrics = {
                "pretrain/iter": it,
                "pretrain/wm_loss": float(loss.detach().item()),
                "pretrain/dynamics_loss": float(dyn_loss.detach().item()),
                "pretrain/reward_loss": float(rew_loss),
                "pretrain/wm_grad_norm": float(wm_grad.detach().item()),
                "val/wm_loss": val_metrics["wm_loss"],
                "val/dynamics_loss": val_metrics["dyn_loss"],
                "val/reward_loss": val_metrics["rew_loss"],
                "wall_clock_seconds": time.time() - t0,
            }
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if run is not None:
                wandb.log(metrics, step=it)
    agent.wm_bootstrapped = True
    agent.save("pretrained_original_pwm_adapter")

    best_imagined_return = -float("inf")
    best_payload = None
    for it in range(1, args.policy_iters + 1):
        agent.update_lrs(it)
        ids = sample_train_indices(data, train_idx, args.policy_batch_size)
        obs, act, rew = batch_windows(data, ids, device, nrm, args)
        if agent.rew_rms:
            rew = agent.rew_rms.normalize(rew)
        metrics = agent.update(obs, act, rew, task=None, finetune_wm=False)
        actor_loss = float(metrics["actor_loss"])
        imagined_return = -actor_loss
        if imagined_return > best_imagined_return:
            best_imagined_return = imagined_return
            best_payload = {"iter": it, "imagined_return": imagined_return, **metrics}
            agent.save("best_policy_extraction")
        if it == 1 or it == args.policy_iters or it % args.eval_every == 0:
            log_metrics = {
                "policy/iter": it,
                "policy/imagined_return_proxy": imagined_return,
                "policy/actor_loss": actor_loss,
                "policy/value_loss": float(metrics["value_loss"]),
                "policy/actor_grad_norm": float(metrics["actor_grad_norm"]),
                "policy/critic_grad_norm": float(metrics["critic_grad_norm"]),
                "wall_clock_seconds": time.time() - t0,
            }
            print(json.dumps(log_metrics, sort_keys=True), flush=True)
            if run is not None:
                wandb.log(log_metrics, step=args.pretrain_iters + it)
    agent.save("final_policy_extraction")

    test_metrics = eval_wm(agent, data, test_idx, nrm, args, device)
    if args.skip_real_eval:
        eval_summary = {"skipped": True, "reason": "--skip-real-eval"}
    else:
        eval_summary = real_env_eval(agent, args, nrm, device)
    summary = {
        "runner": "original_pwm_algorithm_adapter",
        "seed": args.seed,
        "pretrain_iters": args.pretrain_iters,
        "policy_iters": args.policy_iters,
        "obs_mode": args.obs_mode,
        "reward_mode": args.reward_mode,
        "rew_rms": args.rew_rms,
        "ret_rms": args.ret_rms,
        "best_imagined_return_proxy": best_imagined_return,
        "best_iter": best_payload["iter"] if best_payload else None,
        "test/wm_loss": test_metrics["wm_loss"],
        "test/dynamics_loss": test_metrics["dyn_loss"],
        "test/reward_loss": test_metrics["rew_loss"],
        "wall_clock_seconds": time.time() - t0,
        **{f"eval/{k}": v for k, v in eval_summary.items()},
    }
    (output_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if run is not None:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
