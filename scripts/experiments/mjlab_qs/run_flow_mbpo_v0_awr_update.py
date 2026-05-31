#!/usr/bin/env python3
"""Run a minimal Flow-MBPO v0 AWR-style policy update on mixed replay."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.experiments.mjlab_qs.render_policy_rollout import build_actor, command_line, git_branch, git_sha  # noqa: E402
from scripts.experiments.mjlab_qs.run_offline_pwm_policy_extraction import real_env_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--policy-checkpoint", required=True)
    p.add_argument("--synthetic-replay", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--update-iters", type=int, default=1000)
    p.add_argument("--real-batch-size", type=int, default=128)
    p.add_argument("--synthetic-batch-size", type=int, default=128)
    p.add_argument("--actor-lr", type=float, default=1.0e-5)
    p.add_argument("--adv-temperature", type=float, default=1.0)
    p.add_argument("--weight-clip", type=float, default=20.0)
    p.add_argument("--bc-anchor-weight", type=float, default=0.1)
    p.add_argument(
        "--action-deviation-weight",
        type=float,
        default=0.0,
        help="KL-like MSE penalty to keep current actor actions near the frozen BC/reference actor.",
    )
    p.add_argument(
        "--support-action-penalty-weight",
        type=float,
        default=0.0,
        help="Penalty for current actor actions whose normalized (state, command, action) is out of real-data support.",
    )
    p.add_argument("--support-max-rows", type=int, default=20000)
    p.add_argument("--support-probe-rows", type=int, default=4096)
    p.add_argument("--support-threshold", type=float, default=-1.0)
    p.add_argument("--support-threshold-quantile", type=float, default=0.90)
    p.add_argument("--support-state-weight", type=float, default=1.0)
    p.add_argument("--support-command-weight", type=float, default=1.0)
    p.add_argument("--support-action-weight", type=float, default=1.0)
    p.add_argument("--grad-norm", type=float, default=1.0)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--quality-filter", default="expert,expert_noisy")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--real-eval-every", type=int, default=0)
    p.add_argument("--real-eval-episodes", type=int, default=8)
    p.add_argument("--real-eval-num-envs", type=int, default=16)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--task-id", default="Mjlab-Velocity-Flat-Unitree-G1")
    p.add_argument("--command-dim", type=int, default=3)
    p.add_argument("--command-position", choices=["tail", "head", "none"], default="tail")
    p.add_argument("--enable-wandb", action="store_true")
    p.add_argument("--wandb-project", default="flow-mbpo-mjlab-flow-mbpo-v0-awr")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--wandb-name", default="")
    return p.parse_args()


def load_norm(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in raw.items() if isinstance(v, list)}


def norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std.clamp_min(1e-6)


def summarize_tensor(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().reshape(-1).cpu()
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "p90": math.nan, "max": math.nan}
    return {
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "p90": float(torch.quantile(finite, 0.90).item()),
        "max": float(finite.max().item()),
    }


def isin(values: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(values.shape, dtype=torch.bool)
    for candidate in candidates.tolist():
        mask |= values == int(candidate)
    return mask


def select_real_indices(data: dict[str, torch.Tensor], metadata: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    split_map = metadata["split_id_map"]
    quality_map = metadata["quality_id_map"]
    split_id = int(split_map[args.split])
    quality_names = [item.strip() for item in args.quality_filter.split(",") if item.strip()]
    quality_ids = torch.tensor([int(quality_map[name]) for name in quality_names], dtype=torch.long)
    mask = data["split_id"].long() == split_id
    mask = mask & isin(data["quality_bin_id"].long(), quality_ids)
    indices = mask.nonzero(as_tuple=False).reshape(-1)
    if indices.numel() == 0:
        raise ValueError(f"No real windows match split={args.split!r}, quality_filter={quality_names!r}")
    return indices


def sample_real_batch(
    data: dict[str, torch.Tensor],
    indices: torch.Tensor,
    nrm: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pick = indices[torch.randint(indices.numel(), (batch_size,))]
    phys = data["phys_obs"][pick, 0].to(device).float()
    command = data["command"][pick, 0].to(device).float()
    action = data["policy_action"][pick, 0].to(device).float()
    reward = data["reward"][pick, 0].to(device).float()
    z = norm(phys, nrm["phys_obs_mean"], nrm["phys_obs_std"])
    if command.shape[-1] and "command_mean" in nrm:
        command = norm(command, nrm["command_mean"], nrm["command_std"])
    return z, command, action, reward


def sample_synthetic_batch(
    replay: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(replay["reward_conservative"].shape[0])
    pick = torch.randint(n, (batch_size,))
    z = replay["state"][pick].to(device).float()
    command = replay["command"][pick].to(device).float()
    action = replay["action"][pick].to(device).float()
    reward = replay["reward_conservative"][pick].to(device).float()
    done = replay["done"][pick].to(device).bool()
    return z, command, action, reward, done


def advantage_weights(reward: torch.Tensor, temperature: float, weight_clip: float) -> torch.Tensor:
    centered = reward - reward.mean()
    scaled = centered / max(float(temperature), 1.0e-6)
    weights = torch.exp(scaled.clamp(min=-20.0, max=20.0))
    return weights.clamp(max=float(weight_clip))


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_row = (pred - target).pow(2).mean(dim=-1)
    return (per_row * weights).sum() / weights.sum().clamp_min(1.0e-8)


def support_features(
    state: torch.Tensor,
    command: torch.Tensor,
    action: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    parts = [
        state.float() * float(args.support_state_weight),
        command.float() * float(args.support_command_weight),
        action.float() * float(args.support_action_weight),
    ]
    return torch.cat(parts, dim=-1).contiguous()


def real_support_features(
    data: dict[str, torch.Tensor],
    indices: torch.Tensor,
    nrm: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    phys = data["phys_obs"][indices, 0].to(device).float()
    command = data["command"][indices, 0].to(device).float()
    action = data["policy_action"][indices, 0].to(device).float()
    z = norm(phys, nrm["phys_obs_mean"], nrm["phys_obs_std"])
    if command.shape[-1] and "command_mean" in nrm:
        command = norm(command, nrm["command_mean"], nrm["command_std"])
    return support_features(z, command, action, args)


def nearest_l2_per_dim(query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    if query.shape[-1] != support.shape[-1]:
        raise ValueError(f"Support feature mismatch: query={query.shape[-1]}, support={support.shape[-1]}")
    denom = math.sqrt(float(query.shape[-1]))
    return torch.cdist(query, support, p=2).min(dim=1).values / denom


def build_support_state(
    data: dict[str, torch.Tensor],
    real_indices: torch.Tensor,
    nrm: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any] | None:
    if float(args.support_action_penalty_weight) <= 0.0:
        return None
    generator = torch.Generator().manual_seed(int(args.seed))
    perm = real_indices[torch.randperm(real_indices.numel(), generator=generator)]
    support_n = min(int(args.support_max_rows), int(perm.numel()))
    probe_n = min(int(args.support_probe_rows), max(0, int(perm.numel()) - support_n))
    support_indices = perm[:support_n]
    probe_indices = perm[support_n : support_n + probe_n]
    if support_indices.numel() == 0 or probe_indices.numel() == 0:
        raise ValueError("Need non-empty support and disjoint probe sets; reduce --support-max-rows if needed")
    support = real_support_features(data, support_indices, nrm, args, device)
    probe = real_support_features(data, probe_indices, nrm, args, device)
    with torch.no_grad():
        probe_distance = nearest_l2_per_dim(probe, support)
        if float(args.support_threshold) >= 0.0:
            threshold = torch.tensor(float(args.support_threshold), dtype=torch.float32, device=device)
        else:
            q = min(max(float(args.support_threshold_quantile), 0.0), 1.0)
            threshold = torch.quantile(probe_distance[torch.isfinite(probe_distance)], q)
    return {
        "support": support,
        "threshold": threshold.detach(),
        "probe_distance": probe_distance.detach().cpu(),
        "support_rows": int(support_indices.numel()),
        "probe_rows": int(probe_indices.numel()),
        "args": args,
    }


def support_action_penalty(
    state: torch.Tensor,
    command: torch.Tensor,
    action: torch.Tensor,
    support_state: dict[str, Any] | None,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if support_state is None:
        zero = action.new_zeros(())
        return zero, action.new_zeros(action.shape[0])
    feat = support_features(state, command, action, support_state["args"])
    distance = nearest_l2_per_dim(feat, support_state["support"])
    penalty = F.relu(distance - support_state["threshold"])
    if weights is None:
        loss = penalty.mean()
    else:
        loss = (penalty * weights).sum() / weights.sum().clamp_min(1.0e-8)
    return loss, distance.detach()


def eval_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        task_id=args.task_id,
        eval_num_envs=int(args.real_eval_num_envs),
        eval_episodes=int(args.real_eval_episodes),
        episode_length=int(args.episode_length),
        command_dim=int(args.command_dim),
        command_position=args.command_position,
        device=args.device,
        seed=int(args.seed),
    )


@torch.no_grad()
def real_eval_snapshot(
    actor,
    args: argparse.Namespace,
    nrm: dict[str, torch.Tensor],
    device: torch.device,
    iteration: int,
) -> dict[str, float | str]:
    was_training = actor.training
    actor.eval()
    summary = real_env_eval(actor, eval_namespace(args), nrm, device)
    if was_training:
        actor.train()
    out: dict[str, float | str] = {f"real_eval/{key}": value for key, value in summary.items()}
    out["real_eval/iter"] = float(iteration)
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA device requested ({args.device}) but torch.cuda.is_available() is false")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_paths = [output_dir / "summary.json", output_dir / "final_policy_extraction.pt"]
    if all(path.exists() for path in complete_paths):
        print(f"flow-mbpo v0 awr update already complete; skipping {output_dir}", flush=True)
        return
    lock_file = (output_dir / ".flow_mbpo_v0_awr_update.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"flow-mbpo v0 awr update already running; skipping {output_dir}", flush=True)
        return
    if all(path.exists() for path in complete_paths):
        print(f"flow-mbpo v0 awr update already complete; skipping {output_dir}", flush=True)
        return
    t0 = time.time()

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    nrm = load_norm(Path(args.normalization), device)
    replay = torch.load(args.synthetic_replay, map_location="cpu", weights_only=False)
    if not isinstance(replay, dict):
        raise TypeError(f"{args.synthetic_replay} must contain a dict")

    state_dim = int(data["phys_obs"].shape[-1])
    action_dim = int(data["policy_action"].shape[-1])
    command_dim = int(data["command"].shape[-1])
    policy_ckpt = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=False)
    actor = build_actor(policy_ckpt, state_dim, command_dim, action_dim, device)
    actor.train()
    reference_actor = None
    if float(args.action_deviation_weight) > 0.0:
        reference_actor = build_actor(policy_ckpt, state_dim, command_dim, action_dim, device)
        reference_actor.eval()
        for param in reference_actor.parameters():
            param.requires_grad_(False)
    real_indices = select_real_indices(data, metadata, args)
    support_state = build_support_state(data, real_indices, nrm, args, device)
    opt = torch.optim.Adam(actor.parameters(), lr=float(args.actor_lr))
    ckpt_args = dict(policy_ckpt.get("args", {}))
    ckpt_args.update(
        {
            "flow_mbpo_v0_update": "awr",
            "flow_mbpo_v0_synthetic_replay": args.synthetic_replay,
            "flow_mbpo_v0_policy_checkpoint": args.policy_checkpoint,
            "flow_mbpo_v0_update_iters": int(args.update_iters),
            "flow_mbpo_v0_action_deviation_weight": float(args.action_deviation_weight),
            "flow_mbpo_v0_support_action_penalty_weight": float(args.support_action_penalty_weight),
            "flow_mbpo_v0_support_threshold": float(support_state["threshold"].item()) if support_state is not None else None,
            "dataset": args.dataset,
            "metadata": args.metadata,
            "normalization": args.normalization,
            "seed": int(args.seed),
        }
    )
    run = None
    config = vars(args) | {
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
        "real_train_windows": int(real_indices.numel()),
        "synthetic_transitions": int(replay["reward_conservative"].shape[0]),
        "support_action_penalty_enabled": support_state is not None,
        "support_rows": int(support_state["support_rows"]) if support_state is not None else 0,
        "support_probe_rows": int(support_state["probe_rows"]) if support_state is not None else 0,
        "support_threshold": float(support_state["threshold"].item()) if support_state is not None else None,
    }
    if args.enable_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group or "flow_mbpo_v0_awr",
            name=args.wandb_name or f"seed{args.seed}_awr_update",
            job_type="flow_mbpo_v0_awr_update",
            config=config,
        )

    best_loss = float("inf")
    best_loss_actor = None
    best_real_return = -float("inf")
    best_real_eval: dict[str, float | str] | None = None
    best_real_actor = None
    last_metrics: dict[str, float] = {}
    real_eval_snapshot_paths: list[str] = []
    for it in range(1, int(args.update_iters) + 1):
        rz, rc, ra, rr = sample_real_batch(data, real_indices, nrm, int(args.real_batch_size), device)
        sz, sc, sa, sr, sd = sample_synthetic_batch(replay, int(args.synthetic_batch_size), device)
        real_pred = actor(rz, rc, deterministic=True).clamp(-1.0, 1.0)
        synth_pred = actor(sz, sc, deterministic=True).clamp(-1.0, 1.0)
        real_weights = advantage_weights(rr, args.adv_temperature, args.weight_clip)
        synth_weights = advantage_weights(sr, args.adv_temperature, args.weight_clip)
        synth_weights = synth_weights * (~sd).float()
        real_loss = weighted_mse(real_pred, ra, real_weights)
        synth_loss = weighted_mse(synth_pred, sa, synth_weights)
        anchor_loss = F.mse_loss(real_pred, ra)
        if reference_actor is not None:
            with torch.no_grad():
                ref_real = reference_actor(rz, rc, deterministic=True).clamp(-1.0, 1.0)
                ref_synth = reference_actor(sz, sc, deterministic=True).clamp(-1.0, 1.0)
            action_deviation_loss = 0.5 * (
                F.mse_loss(real_pred, ref_real) + weighted_mse(synth_pred, ref_synth, (~sd).float())
            )
        else:
            action_deviation_loss = real_pred.new_zeros(())
        real_support_loss, real_support_distance = support_action_penalty(rz, rc, real_pred, support_state)
        synth_support_loss, synth_support_distance = support_action_penalty(sz, sc, synth_pred, support_state, (~sd).float())
        support_action_loss = 0.5 * (real_support_loss + synth_support_loss)
        loss = (
            real_loss
            + synth_loss
            + float(args.bc_anchor_weight) * anchor_loss
            + float(args.action_deviation_weight) * action_deviation_loss
            + float(args.support_action_penalty_weight) * support_action_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), float(args.grad_norm))
        opt.step()
        loss_value = float(loss.detach().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_loss_actor = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        if it == 1 or it == int(args.update_iters) or it % int(args.log_every) == 0:
            last_metrics = {
                "awr/iter": float(it),
                "awr/loss": loss_value,
                "awr/real_loss": float(real_loss.detach().item()),
                "awr/synthetic_loss": float(synth_loss.detach().item()),
                "awr/bc_anchor_loss": float(anchor_loss.detach().item()),
                "awr/action_deviation_loss": float(action_deviation_loss.detach().item()),
                "awr/support_action_loss": float(support_action_loss.detach().item()),
                "awr/real_support_distance_mean": float(real_support_distance.detach().mean().item()),
                "awr/real_support_distance_p90": float(torch.quantile(real_support_distance.detach(), 0.90).item()),
                "awr/synthetic_support_distance_mean": float(synth_support_distance.detach().mean().item()),
                "awr/synthetic_support_distance_p90": float(torch.quantile(synth_support_distance.detach(), 0.90).item()),
                "awr/real_reward_mean": float(rr.detach().mean().item()),
                "awr/synthetic_reward_mean": float(sr.detach().mean().item()),
                "awr/real_weight_mean": float(real_weights.detach().mean().item()),
                "awr/synthetic_weight_mean": float(synth_weights.detach().mean().item()),
                "awr/synthetic_done_fraction": float(sd.float().mean().item()),
                "awr/grad_norm": float(grad.detach().item()),
                "wall_clock_seconds": time.time() - t0,
            }
            print(json.dumps(last_metrics, sort_keys=True), flush=True)
            if run is not None:
                run.log(last_metrics, step=it)
        if int(args.real_eval_every) > 0 and (it == int(args.update_iters) or it % int(args.real_eval_every) == 0):
            eval_metrics = real_eval_snapshot(actor, args, nrm, device, it)
            print(json.dumps(eval_metrics, sort_keys=True), flush=True)
            if run is not None:
                run.log(eval_metrics, step=it)
            snapshot_path = output_dir / "real_eval_snapshots" / f"iter_{it:06d}_policy_extraction.pt"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "actor": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
                    "args": ckpt_args,
                    "checkpoint_kind": "real_eval_snapshot",
                    "is_true_best_snapshot": False,
                    "real_eval": eval_metrics,
                    "iteration": int(it),
                },
                snapshot_path,
            )
            real_eval_snapshot_paths.append(str(snapshot_path))
            real_return = float(eval_metrics["real_eval/return_mean"])
            if real_return > best_real_return:
                best_real_return = real_return
                best_real_eval = eval_metrics
                best_real_actor = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}

    final_checkpoint = output_dir / "final_policy_extraction.pt"
    best_checkpoint = output_dir / "best_policy_extraction.pt"
    best_training_checkpoint = output_dir / "best_training_loss_policy_extraction.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "args": ckpt_args,
            "checkpoint_kind": "final",
        },
        final_checkpoint,
    )
    torch.save(
        {
            "actor": best_loss_actor if best_loss_actor is not None else actor.state_dict(),
            "args": ckpt_args,
            "checkpoint_kind": "best_training_loss",
            "is_true_best_snapshot": False,
            "best_training_loss": best_loss,
        },
        best_training_checkpoint,
    )
    best_payload: dict[str, Any]
    if best_real_actor is not None:
        best_payload = {
            "actor": best_real_actor,
            "args": ckpt_args,
            "checkpoint_kind": "best_real_eval",
            "is_true_best_snapshot": True,
            "best_real_eval": best_real_eval,
        }
    else:
        best_payload = {
            "actor": best_loss_actor if best_loss_actor is not None else actor.state_dict(),
            "args": ckpt_args,
            "checkpoint_kind": "best_training_loss",
            "is_true_best_snapshot": False,
            "note": "Best is selected by AWR training loss only; enable --real-eval-every for true best.",
        }
    torch.save(
        best_payload,
        best_checkpoint,
    )
    summary: dict[str, Any] = config | {
        "output_dir": str(output_dir),
        "final_checkpoint": str(final_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "best_training_checkpoint": str(best_training_checkpoint),
        "real_eval_snapshot_checkpoints": real_eval_snapshot_paths,
        "best_training_loss": best_loss,
        "best_real_return": best_real_return if math.isfinite(best_real_return) else None,
        "best_real_eval": best_real_eval,
        "best_is_true_snapshot": best_real_actor is not None,
        "last_metrics": last_metrics,
        "synthetic_reward_conservative": summarize_tensor(replay["reward_conservative"]),
        "synthetic_done_fraction": float(replay["done"].float().mean().item()),
        "support_probe_distance": summarize_tensor(support_state["probe_distance"]) if support_state is not None else None,
        "wall_clock_seconds": time.time() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if run is not None:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
