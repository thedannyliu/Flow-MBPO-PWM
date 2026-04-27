#!/usr/bin/env python3
"""Flow-push sidecar: train Flow until it matches MLP train rollout loss.

This is an existence/upper-bound diagnostic, not the equal-update fair result.
For each dataset/seed, the script:
  1. trains an MLP WM for a fixed update budget;
  2. records MLP train/val/test masked H16 rollout dynamics loss;
  3. trains a Flow WM for a longer budget and early-stops if its train H16
     rollout dynamics loss reaches the MLP target within tolerance;
  4. records iterations, wall-clock, and compute ratio.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict

import torch
import wandb

from run_phaseA_wm_feasibility import (
    FlowWM,
    MLPWM,
    batch,
    evaluate,
    load_norm,
    sample_train_indices,
    train_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mlp-train-iters", type=int, default=50000)
    p.add_argument("--flow-max-iters", type=int, default=300000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--flow-substeps", type=int, default=4)
    p.add_argument("--rollout-gamma", type=float, default=0.99)
    p.add_argument("--match-tolerance", type=float, default=0.05)
    p.add_argument("--wandb-project", default="flow-mbpo-mjlab-phaseA-train-loss-match")
    p.add_argument("--wandb-group", default="a25_flow_train_match")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--disable-wandb", action="store_true")
    return p.parse_args()


def split_indices(data: Dict[str, torch.Tensor]):
    return (
        (data["split_id"] == 0).nonzero(as_tuple=False).squeeze(-1),
        (data["split_id"] == 1).nonzero(as_tuple=False).squeeze(-1),
        (data["split_id"] == 2).nonzero(as_tuple=False).squeeze(-1),
    )


def train_steps(
    model: torch.nn.Module,
    method: str,
    data: Dict[str, torch.Tensor],
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    device: torch.device,
    nrm: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    max_iters: int,
    run,
    phase: str,
    target: float | None = None,
) -> tuple[int, float, Dict[str, float], bool]:
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.time()
    matched = False
    last_metrics: Dict[str, float] = {}
    for it in range(max_iters + 1):
        if it > 0:
            ids = sample_train_indices(data, train_idx, args.batch_size)
            z, a, r, c, done = batch(data, ids, device, nrm)
            loss = train_loss(model, method, z, a, r, c, done, gamma=args.rollout_gamma)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()
        if it % args.eval_every == 0 or it == max_iters:
            train_m = evaluate(model, method, data, train_idx[: min(train_idx.numel(), 4096)], device, nrm, args.eval_batch_size, gamma=args.rollout_gamma)
            val_m = evaluate(model, method, data, val_idx, device, nrm, args.eval_batch_size, gamma=args.rollout_gamma)
            elapsed = time.time() - t0
            last_metrics = {
                f"{phase}/train/{k}": v for k, v in train_m.items()
            }
            last_metrics.update({f"{phase}/val/{k}": v for k, v in val_m.items()})
            last_metrics[f"{phase}/iter"] = it
            last_metrics[f"{phase}/wall_clock_seconds"] = elapsed
            if target is not None:
                threshold = target * (1.0 + args.match_tolerance)
                last_metrics[f"{phase}/target_train_rollout_dyn_mse_H16"] = target
                last_metrics[f"{phase}/match_threshold"] = threshold
                last_metrics[f"{phase}/train_loss_ratio_to_mlp"] = train_m["rollout_dyn_mse_H16"] / max(target, 1e-12)
                if train_m["rollout_dyn_mse_H16"] <= threshold:
                    matched = True
            if run is not None:
                run.log(last_metrics, step=it if phase == "flow" else it)
            print(json.dumps(last_metrics, sort_keys=True), flush=True)
            if matched:
                return it, elapsed, last_metrics, True
    return max_iters, time.time() - t0, last_metrics, matched


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    metadata = json.loads(Path(args.metadata).read_text())
    nrm = load_norm(Path(args.normalization), device)
    train_idx, val_idx, test_idx = split_indices(data)

    state_dim = int(data["phys_obs"].shape[-1])
    action_dim = int(data["policy_action"].shape[-1])
    command_dim = int(data["command"].shape[-1])

    run = None
    if not args.disable_wandb:
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name or f"train_loss_match_seed{args.seed}",
            job_type="flow_push_train_loss_match",
            config={**vars(args), "dataset_metadata": metadata},
        )

    mlp = MLPWM(state_dim, action_dim, command_dim, args.hidden).to(device)
    mlp_iters, mlp_time, mlp_last, _ = train_steps(
        mlp, "mlp_ref", data, train_idx, val_idx, device, nrm, args, args.mlp_train_iters, run, "mlp"
    )
    mlp_train_target = float(mlp_last["mlp/train/rollout_dyn_mse_H16"])
    mlp_test = evaluate(mlp, "mlp_ref", data, test_idx, device, nrm, args.eval_batch_size, gamma=args.rollout_gamma)
    torch.save({"model": mlp.state_dict(), "args": vars(args), "train_target": mlp_train_target}, output_dir / "mlp_ref.pt")

    flow = FlowWM(state_dim, action_dim, command_dim, args.hidden, substeps=args.flow_substeps).to(device)
    flow_iters, flow_time, flow_last, matched = train_steps(
        flow,
        "flow_ref",
        data,
        train_idx,
        val_idx,
        device,
        nrm,
        args,
        args.flow_max_iters,
        run,
        "flow",
        target=mlp_train_target,
    )
    flow_test = evaluate(flow, "flow_ref", data, test_idx, device, nrm, args.eval_batch_size, gamma=args.rollout_gamma)
    torch.save({"model": flow.state_dict(), "args": vars(args), "matched": matched}, output_dir / "flow_ref_train_match.pt")

    summary = {
        "seed": args.seed,
        "mlp_train_iters": mlp_iters,
        "mlp_wall_clock_seconds": mlp_time,
        "mlp_train_rollout_dyn_mse_H16": mlp_train_target,
        "mlp_test": mlp_test,
        "flow_max_iters": args.flow_max_iters,
        "flow_iters_to_stop": flow_iters,
        "flow_wall_clock_seconds": flow_time,
        "flow_matched_mlp_train_loss": matched,
        "match_tolerance": args.match_tolerance,
        "flow_train_rollout_dyn_mse_H16": float(flow_last.get("flow/train/rollout_dyn_mse_H16", math.nan)),
        "flow_train_loss_ratio_to_mlp": float(flow_last.get("flow/train_loss_ratio_to_mlp", math.nan)),
        "flow_test": flow_test,
        "extra_flow_iters_vs_mlp": flow_iters - mlp_iters,
        "flow_to_mlp_wall_clock_ratio": flow_time / max(mlp_time, 1e-12),
    }
    (output_dir / "train_match_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    if run is not None:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
