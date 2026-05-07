#!/usr/bin/env python3
"""Frozen-WM policy extraction for the MJLab-QS offline PWM pipeline.

This runner consumes the state-space WM checkpoints produced by
`run_phaseA_wm_feasibility.py`, freezes the world model, trains an MLP policy
and critic using differentiable imagined rollouts, then evaluates the extracted
policy in the real MJLab environment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.mjlab_qs.collect_mjlab_qs_native_episodes import (
    patch_headless_display_dependency,
    patch_mujoco_compatibility,
    split_obs,
    tensor_from_actor_obs,
)
from flow_mbpo_pwm.models.actor import ActorStochasticMLP
from flow_mbpo_pwm.models.flow_actor import ActorFlowODE
from flow_mbpo_pwm.models.critic import CriticMLP
from scripts.experiments.mjlab_qs.run_phaseA_wm_feasibility import (
    FlowWM,
    MLPWM,
    load_norm,
    norm,
    rollout_losses,
    sample_train_indices,
    train_loss,
)


class Actor(nn.Module):
    """PWM-style stochastic MLP actor.

    Original PWM uses a stochastic MLP actor with hidden layers [400, 200, 100],
    Mish activations, init_logstd=-1.0, min_logstd=-1.427, followed by tanh.
    """

    def __init__(
        self,
        state_dim: int,
        command_dim: int,
        action_dim: int,
        units: List[int],
        init_logstd: float,
        min_logstd: float,
    ):
        super().__init__()
        self.net = ActorStochasticMLP(
            obs_dim=state_dim + command_dim,
            action_dim=action_dim,
            units=units,
            activation_class=nn.Mish,
            init_gain=1.0,
            init_logstd=init_logstd,
            min_logstd=min_logstd,
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        raw = self.net(torch.cat([z, c], dim=-1), deterministic=deterministic)
        return torch.tanh(raw)

    def std_mean(self) -> torch.Tensor:
        return self.net.get_logstd().exp().mean()


class FlowActor(nn.Module):
    """ODE-based flow policy used as the policy-side 2x2 variant.

    The actor still optimizes the same PWM-style imagined return objective.
    This is not FPO/PPO-ratio training; it is an expressive differentiable
    policy class inside the PWM/FoG policy extraction loop.
    """

    def __init__(
        self,
        state_dim: int,
        command_dim: int,
        action_dim: int,
        units: List[int],
        flow_substeps: int,
        flow_integrator: str,
    ):
        super().__init__()
        self.net = ActorFlowODE(
            obs_dim=state_dim + command_dim,
            action_dim=action_dim,
            units=units,
            activation_class=nn.Mish,
            init_gain=1.0,
            flow_substeps=flow_substeps,
            flow_integrator=flow_integrator,
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        raw = self.net(torch.cat([z, c], dim=-1), deterministic=deterministic)
        return torch.tanh(raw)

    def std_mean(self) -> torch.Tensor:
        return self.net.get_logstd().exp().mean()


class Critic(nn.Module):
    def __init__(self, state_dim: int, command_dim: int, units: List[int]):
        super().__init__()
        self.net = CriticMLP(
            obs_dim=state_dim + command_dim,
            units=units,
            activation_class=nn.Mish,
            init_gain=2.0**0.5,
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, c], dim=-1)).squeeze(-1)


class CriticEnsemble(nn.Module):
    def __init__(self, state_dim: int, command_dim: int, units: List[int], num_critics: int):
        super().__init__()
        self.critics = nn.ModuleList([Critic(state_dim, command_dim, units) for _ in range(num_critics)])

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.stack([critic(z, c) for critic in self.critics], dim=0)

    def min_value(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.forward(z, c).min(dim=0).values


class RunningScalarRMS:
    def __init__(self, device: torch.device):
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = torch.zeros((), device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1)
        if x.numel() == 0:
            return
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = torch.tensor(float(x.numel()), device=x.device)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total.clamp_min(1.0)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total.clamp_min(1.0)
        self.mean = new_mean
        self.var = m_2 / total.clamp_min(1.0)
        self.count = total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--wm-method", choices=["mlp_ref", "flow_ref", "flow_endpoint"], required=True)
    p.add_argument("--policy-type", choices=["mlp", "flow"], default="mlp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task-id", default="Mjlab-Velocity-Flat-Unitree-G1")
    p.add_argument("--policy-iters", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--actor-lr", type=float, default=5e-4)
    p.add_argument("--critic-lr", type=float, default=5e-4)
    p.add_argument("--actor-units", default="400,200,100")
    p.add_argument("--critic-units", default="400,200")
    p.add_argument("--num-critics", type=int, default=3)
    p.add_argument("--critic-iterations", type=int, default=8)
    p.add_argument("--critic-batches", type=int, default=4)
    p.add_argument("--actor-grad-norm", type=float, default=1.0)
    p.add_argument("--critic-grad-norm", type=float, default=100.0)
    p.add_argument("--init-logstd", type=float, default=-1.0)
    p.add_argument("--min-logstd", type=float, default=-1.427)
    p.add_argument("--flow-policy-substeps", type=int, default=2)
    p.add_argument("--flow-policy-integrator", choices=["euler", "heun"], default="heun")
    p.add_argument("--ret-rms", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wm-hidden", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--eval-num-envs", type=int, default=16)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--command-dim", type=int, default=3)
    p.add_argument("--command-position", choices=["tail", "head", "none"], default="tail")
    p.add_argument("--action-l2", type=float, default=1.0e-4)
    p.add_argument("--skip-real-eval", action="store_true")
    p.add_argument("--online-finetune-rounds", type=int, default=0)
    p.add_argument("--online-collect-windows", type=int, default=256)
    p.add_argument("--online-wm-iters", type=int, default=1000)
    p.add_argument("--online-policy-iters", type=int, default=3000)
    p.add_argument("--online-wm-lr", type=float, default=3e-4)
    p.add_argument("--wandb-project", default="flow-mbpo-mjlab-offline-pwm-policy-extraction")
    p.add_argument("--wandb-group", default="g1_frozen_wm_policy_extraction")
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
    return data, metadata, nrm, train_idx


def build_wm(args: argparse.Namespace, data: Dict[str, torch.Tensor], device: torch.device, frozen: bool = True) -> nn.Module:
    state_dim = int(data["phys_obs"].shape[-1])
    action_dim = int(data["policy_action"].shape[-1])
    command_dim = int(data["command"].shape[-1])
    ckpt = torch.load(args.wm_checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    hidden = int(ckpt_args.get("hidden", args.wm_hidden))
    substeps = int(ckpt_args.get("flow_substeps", 4))
    if args.wm_method == "mlp_ref":
        wm: nn.Module = MLPWM(state_dim, action_dim, command_dim, hidden=hidden)
    else:
        wm = FlowWM(state_dim, action_dim, command_dim, hidden=hidden, substeps=substeps)
    wm.load_state_dict(ckpt["model"])
    wm.to(device)
    if frozen:
        wm.eval()
        for p in wm.parameters():
            p.requires_grad_(False)
    return wm


def batch_windows(data: Dict[str, torch.Tensor], ids: torch.Tensor, device: torch.device, nrm: Dict[str, torch.Tensor]):
    z = norm(data["phys_obs"][ids].to(device).float(), nrm["phys_obs_mean"], nrm["phys_obs_std"])
    c = data["command"][ids].to(device).float()
    if c.shape[-1] and "command_mean" in nrm:
        c = norm(c, nrm["command_mean"], nrm["command_std"])
    return z, c


def parse_units(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def imagine_rollout(
    wm: nn.Module,
    actor: nn.Module,
    critic: CriticEnsemble,
    z0: torch.Tensor,
    c_seq: torch.Tensor,
    horizon: int,
    gamma: float,
    lam: float,
    action_l2: float,
    deterministic_actor: bool = False,
):
    h = min(horizon, c_seq.shape[1])
    z = z0
    states: List[torch.Tensor] = []
    commands: List[torch.Tensor] = []
    next_states: List[torch.Tensor] = []
    next_commands: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    rewards: List[torch.Tensor] = []
    for t in range(h):
        c = c_seq[:, t]
        a = actor(z, c, deterministic=deterministic_actor)
        r = wm.reward(z, a, c) - float(action_l2) * a.pow(2).mean(dim=-1)
        states.append(z)
        commands.append(c)
        actions.append(a)
        rewards.append(r)
        z = wm.next(z, a)
        next_states.append(z)
        next_commands.append(c_seq[:, min(t + 1, h - 1)])
    terminal_c = c_seq[:, h - 1]
    terminal_v = critic.min_value(z, terminal_c)
    next_values = [critic.min_value(ns, nc) for ns, nc in zip(next_states, next_commands)]
    values = [critic(s, c) for s, c in zip(states, commands)]
    ret = terminal_v
    lambda_targets: List[torch.Tensor] = []
    for t in reversed(range(h)):
        bootstrap = next_values[t]
        ret = rewards[t] + gamma * ((1.0 - lam) * bootstrap + lam * ret)
        lambda_targets.append(ret)
    lambda_targets.reverse()
    stacked_targets = torch.stack(lambda_targets, dim=1)
    stacked_values = torch.stack(values, dim=2)
    discounted_return = torch.zeros_like(rewards[0])
    disc = 1.0
    for r in rewards:
        discounted_return = discounted_return + disc * r
        disc *= gamma
    discounted_return = discounted_return + disc * terminal_v
    action_norm = torch.stack([a.pow(2).mean(dim=-1).sqrt() for a in actions], dim=1).mean()
    states_t = torch.stack(states, dim=1)
    commands_t = torch.stack(commands, dim=1)
    return discounted_return, stacked_values, stacked_targets, states_t, commands_t, action_norm


def train_actor_critic_steps(
    wm: nn.Module,
    actor: nn.Module,
    critic: CriticEnsemble,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    ret_rms: RunningScalarRMS | None,
    data: Dict[str, torch.Tensor],
    train_idx: torch.Tensor,
    nrm: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    start_iter: int,
    num_iters: int,
    run,
    t0: float,
    metric_prefix: str = "train",
) -> Tuple[float, Dict[str, float] | None]:
    best_return = -math.inf
    best_payload = None
    for local_it in range(1, num_iters + 1):
        it = start_iter + local_it
        ids = sample_train_indices(data, train_idx, args.batch_size)
        z_seq, c_seq = batch_windows(data, ids, device, nrm)
        imagined_return, _values, _targets, _states, _commands, action_norm = imagine_rollout(
            wm, actor, critic, z_seq[:, 0], c_seq, args.horizon, args.gamma, args.lam, args.action_l2
        )

        for p in critic.parameters():
            p.requires_grad_(False)
        actor_objective = imagined_return
        if ret_rms is not None:
            ret_rms.update(actor_objective)
            actor_objective = actor_objective / torch.sqrt(ret_rms.var + 1e-5)
        actor_loss = -actor_objective.mean()
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), args.actor_grad_norm)
        actor_opt.step()
        for p in critic.parameters():
            p.requires_grad_(True)

        for p in actor.parameters():
            p.requires_grad_(False)
        z_seq2, c_seq2 = batch_windows(data, ids, device, nrm)
        _imagined_return2, values2, targets2, states2, commands2, _ = imagine_rollout(
            wm, actor, critic, z_seq2[:, 0], c_seq2, args.horizon, args.gamma, args.lam, args.action_l2
        )
        flat_states = states2.detach().reshape(-1, int(data["phys_obs"].shape[-1]))
        flat_commands = commands2.detach().reshape(-1, int(data["command"].shape[-1]))
        flat_targets = targets2.detach().reshape(-1)
        total = flat_targets.numel()
        critic_batch_size = max(1, total // max(1, args.critic_batches))
        critic_loss = torch.zeros((), device=device)
        critic_grad = torch.zeros((), device=device)
        for _ in range(args.critic_iterations):
            perm = torch.randperm(total, device=device)
            for start in range(0, total, critic_batch_size):
                mb = perm[start : start + critic_batch_size]
                pred = critic(flat_states[mb], flat_commands[mb])
                target = flat_targets[mb].unsqueeze(0).expand_as(pred)
                loss_v = F.mse_loss(pred, target)
                critic_opt.zero_grad(set_to_none=True)
                loss_v.backward()
                critic_grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), args.critic_grad_norm)
                critic_opt.step()
                critic_loss = loss_v.detach()
        for p in actor.parameters():
            p.requires_grad_(True)

        if local_it % args.eval_every == 0 or local_it == 1 or local_it == num_iters:
            metrics = {
                "iter": it,
                f"{metric_prefix}/imagined_return": float(imagined_return.detach().mean().item()),
                f"{metric_prefix}/actor_loss": float(actor_loss.detach().item()),
                f"{metric_prefix}/critic_loss": float(critic_loss.detach().item()),
                f"{metric_prefix}/action_norm": float(action_norm.detach().item()),
                f"{metric_prefix}/actor_std": float(actor.std_mean().detach().item()),
                f"{metric_prefix}/actor_grad_norm": float(actor_grad.detach().item()),
                f"{metric_prefix}/critic_grad_norm": float(critic_grad.detach().item()),
                "wall_clock_seconds": time.time() - t0,
            }
            if ret_rms is not None:
                metrics[f"{metric_prefix}/ret_rms_mean"] = float(ret_rms.mean.detach().item())
                metrics[f"{metric_prefix}/ret_rms_var"] = float(ret_rms.var.detach().item())
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if run is not None:
                wandb.log(metrics, step=it)
            if metrics[f"{metric_prefix}/imagined_return"] > best_return:
                best_return = metrics[f"{metric_prefix}/imagined_return"]
                best_payload = {"iter": it, **metrics}
    return best_return, best_payload


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
    env_cfg.seed = int(args.seed) + 10000
    if hasattr(env_cfg, "episode_length_s") and hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "mujoco"):
        env_dt = float(env_cfg.sim.mujoco.timestep) * float(env_cfg.decimation)
        env_cfg.episode_length_s = float(args.episode_length) * env_dt
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    obs_td = wrapped.get_observations()
    obs_groups = list(agent_cfg.obs_groups["actor"])
    return wrapped, obs_td, obs_groups


@torch.no_grad()
def real_env_eval(actor: nn.Module, args: argparse.Namespace, nrm: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, float | str]:
    env, obs_td, obs_groups = build_eval_env(args)
    returns = torch.zeros(args.eval_num_envs, device=device)
    lengths = torch.zeros(args.eval_num_envs, device=device)
    done_returns: List[float] = []
    done_lengths: List[float] = []
    while len(done_returns) < args.eval_episodes:
        obs = tensor_from_actor_obs(obs_td, obs_groups)
        phys, cmd = split_obs(obs, args.command_dim, args.command_position)
        z = norm(phys.float(), nrm["phys_obs_mean"], nrm["phys_obs_std"])
        c = cmd.float()
        if c.shape[-1] and "command_mean" in nrm:
            c = norm(c, nrm["command_mean"], nrm["command_std"])
        action = actor(z, c, deterministic=True).clamp(-1.0, 1.0)
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


def collect_online_windows(
    actor: nn.Module,
    args: argparse.Namespace,
    nrm: Dict[str, torch.Tensor],
    device: torch.device,
    num_windows: int,
) -> Dict[str, torch.Tensor]:
    """Collect real MJLab windows with the current policy for WM finetuning.

    This is a lightweight PWM-style online replay approximation for the
    state-space MJLab-QS runner. It stores normalized physical states so the
    same fixed-latent WM losses can be reused.
    """

    env, obs_td, obs_groups = build_eval_env(args)
    per_env = [[] for _ in range(args.eval_num_envs)]
    windows = []
    while len(windows) < num_windows:
        obs = tensor_from_actor_obs(obs_td, obs_groups)
        phys, cmd = split_obs(obs, args.command_dim, args.command_position)
        z = norm(phys.float(), nrm["phys_obs_mean"], nrm["phys_obs_std"])
        c_raw = cmd.float()
        c = c_raw
        if c.shape[-1] and "command_mean" in nrm:
            c = norm(c, nrm["command_mean"], nrm["command_std"])
        action = actor(z, c, deterministic=False).clamp(-1.0, 1.0)
        next_obs_td, reward, done, _extras = env.step(action)
        reward_n = norm(reward.to(device).float().reshape(-1, 1), nrm["reward_mean"], nrm["reward_std"]).squeeze(-1)
        done = done.to(device).bool().reshape(-1)
        next_obs = tensor_from_actor_obs(next_obs_td, obs_groups)
        next_phys, _next_cmd = split_obs(next_obs, args.command_dim, args.command_position)
        next_z = norm(next_phys.float(), nrm["phys_obs_mean"], nrm["phys_obs_std"])
        for env_i in range(args.eval_num_envs):
            per_env[env_i].append(
                (
                    z[env_i].detach().cpu(),
                    action[env_i].detach().cpu(),
                    reward_n[env_i].detach().cpu(),
                    c[env_i].detach().cpu(),
                    done[env_i].detach().cpu(),
                    next_z[env_i].detach().cpu(),
                )
            )
            if len(per_env[env_i]) >= args.horizon:
                seq = per_env[env_i][-args.horizon :]
                z_seq = [item[0] for item in seq] + [seq[-1][5]]
                windows.append(
                    {
                        "phys_obs": torch.stack(z_seq, dim=0),
                        "policy_action": torch.stack([item[1] for item in seq], dim=0),
                        "reward": torch.stack([item[2] for item in seq], dim=0),
                        "command": torch.stack([item[3] for item in seq], dim=0),
                        "done": torch.stack([item[4] for item in seq], dim=0),
                    }
                )
                if len(windows) >= num_windows:
                    break
            if bool(done[env_i].item()):
                per_env[env_i].clear()
        obs_td = next_obs_td
    env.close()
    out = {}
    for key in ["phys_obs", "policy_action", "reward", "command", "done"]:
        out[key] = torch.stack([w[key] for w in windows], dim=0)
    return out


def finetune_wm_on_online_windows(
    wm: nn.Module,
    method: str,
    online_data: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    run,
    global_step_offset: int,
) -> None:
    for p in wm.parameters():
        p.requires_grad_(True)
    wm.train()
    opt = torch.optim.Adam(wm.parameters(), lr=args.online_wm_lr)
    n = online_data["phys_obs"].shape[0]
    for it in range(1, args.online_wm_iters + 1):
        ids = torch.randint(0, n, (min(args.batch_size, n),), device="cpu")
        z = online_data["phys_obs"][ids].to(device).float()
        a = online_data["policy_action"][ids].to(device).float()
        r = online_data["reward"][ids].to(device).float()
        c = online_data["command"][ids].to(device).float()
        done = online_data["done"][ids].to(device).bool()
        loss = train_loss(wm, method, z, a, r, c, done, gamma=args.gamma)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it == 1 or it == args.online_wm_iters or it % max(1, args.online_wm_iters // 5) == 0:
            with torch.no_grad():
                dyn, rew, _, dyn_agg, rew_agg = rollout_losses(wm, z, a, r, c, done, gamma=args.gamma)
            metrics = {
                "online_wm/iter": it,
                "online_wm/loss": float(loss.detach().item()),
                "online_wm/rollout_dyn_mse_H16": float(dyn_agg.detach().item()),
                "online_wm/reward_mse": float(rew_agg.detach().item()),
            }
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if run is not None:
                wandb.log(metrics, step=global_step_offset + it)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, metadata, nrm, train_idx = load_data(args, device)
    wm = build_wm(args, data, device, frozen=True)
    state_dim = int(data["phys_obs"].shape[-1])
    action_dim = int(data["policy_action"].shape[-1])
    command_dim = int(data["command"].shape[-1])
    if args.policy_type == "mlp":
        actor: nn.Module = Actor(
            state_dim,
            command_dim,
            action_dim,
            units=parse_units(args.actor_units),
            init_logstd=args.init_logstd,
            min_logstd=args.min_logstd,
        ).to(device)
    else:
        actor = FlowActor(
            state_dim,
            command_dim,
            action_dim,
            units=parse_units(args.actor_units),
            flow_substeps=args.flow_policy_substeps,
            flow_integrator=args.flow_policy_integrator,
        ).to(device)
    critic = CriticEnsemble(
        state_dim,
        command_dim,
        units=parse_units(args.critic_units),
        num_critics=args.num_critics,
    ).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    ret_rms = RunningScalarRMS(device) if args.ret_rms else None

    run = None
    if not args.disable_wandb:
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name or f"{args.wm_method}_seed{args.seed}",
            job_type="policy_extraction",
            config={**vars(args), "dataset_metadata": metadata, "git_sha": git_sha()},
        )

    t0 = time.time()
    best_return, best_payload = train_actor_critic_steps(
        wm,
        actor,
        critic,
        actor_opt,
        critic_opt,
        ret_rms,
        data,
        train_idx,
        nrm,
        args,
        device,
        start_iter=0,
        num_iters=args.policy_iters,
        run=run,
        t0=t0,
        metric_prefix="train",
    )
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "args": vars(args),
            "best": best_payload,
        },
        output_dir / "best_policy_extraction.pt",
    )

    total_policy_iters = args.policy_iters
    for round_id in range(1, args.online_finetune_rounds + 1):
        online_data = collect_online_windows(actor, args, nrm, device, args.online_collect_windows)
        torch.save(online_data, output_dir / f"online_round_{round_id}_windows.pt")
        if run is not None:
            wandb.log({"online/round": round_id, "online/collected_windows": args.online_collect_windows}, step=total_policy_iters)
        finetune_wm_on_online_windows(
            wm,
            args.wm_method,
            online_data,
            args,
            device,
            run,
            global_step_offset=total_policy_iters,
        )
        round_best, round_payload = train_actor_critic_steps(
            wm,
            actor,
            critic,
            actor_opt,
            critic_opt,
            ret_rms,
            data,
            train_idx,
            nrm,
            args,
            device,
            start_iter=total_policy_iters,
            num_iters=args.online_policy_iters,
            run=run,
            t0=t0,
            metric_prefix=f"online_round_{round_id}",
        )
        total_policy_iters += args.online_policy_iters
        if round_best > best_return:
            best_return = round_best
            best_payload = round_payload

    torch.save(
        {"actor": actor.state_dict(), "critic": critic.state_dict(), "args": vars(args)},
        output_dir / "final_policy_extraction.pt",
    )
    if args.skip_real_eval:
        eval_summary = {"skipped": True, "reason": "--skip-real-eval"}
    else:
        eval_summary = real_env_eval(actor, args, nrm, device)
    summary = {
        "wm_method": args.wm_method,
        "policy_type": args.policy_type,
        "seed": args.seed,
        "online_finetune_rounds": args.online_finetune_rounds,
        "best_imagined_return": best_return,
        "best_iter": best_payload["iter"] if best_payload else None,
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
