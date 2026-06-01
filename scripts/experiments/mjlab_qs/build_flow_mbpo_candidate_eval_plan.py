#!/usr/bin/env python3
"""Build eval/render commands for Flow-MBPO AWR candidate checkpoints."""

from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--awr-dir", required=True)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--rollout-dir", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-sh", required=True)
    p.add_argument("--output-eval-manifest", default="")
    p.add_argument("--output-rollout-manifest", default="")
    p.add_argument("--python-bin", default="python")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--eval-num-envs", type=int, default=16)
    p.add_argument("--rollout-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--wandb-project-eval", default="")
    p.add_argument("--wandb-project-rollout", default="")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--eval-baseline-return", type=float, default=45.8491)
    p.add_argument("--eval-baseline-length", type=float, default=594.97)
    p.add_argument("--eval-baseline-fall", type=float, default=0.625)
    p.add_argument("--rollout-baseline-return", type=float, default=54.1283)
    p.add_argument("--rollout-baseline-length", type=float, default=688.40)
    p.add_argument("--rollout-baseline-fall", type=float, default=0.400)
    return p.parse_args()


def candidate_paths(awr_dir: Path) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    named = [
        ("final", awr_dir / "final_policy_extraction.pt"),
        ("best", awr_dir / "best_policy_extraction.pt"),
        ("best_training_loss", awr_dir / "best_training_loss_policy_extraction.pt"),
    ]
    for name, path in named:
        if path.exists():
            candidates.append((name, path))
    snapshot_dir = awr_dir / "real_eval_snapshots"
    for path in sorted(snapshot_dir.glob("*_policy_extraction.pt")):
        name = path.name.removesuffix("_policy_extraction.pt")
        candidates.append((name, path))
    return candidates


def command(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    awr_dir = Path(args.awr_dir)
    eval_dir = Path(args.eval_dir)
    rollout_dir = Path(args.rollout_dir)
    rows: list[dict[str, str]] = []
    for name, ckpt in candidate_paths(awr_dir):
        eval_out = eval_dir / name
        rollout_out = rollout_dir / name
        eval_cmd = command(
            [
                args.python_bin,
                "scripts/experiments/mjlab_qs/eval_policy_checkpoint.py",
                "--policy-checkpoint",
                ckpt,
                "--output-dir",
                eval_out,
                "--checkpoint-kind",
                name,
                "--device",
                args.device,
                "--eval-episodes",
                args.eval_episodes,
                "--eval-num-envs",
                args.eval_num_envs,
                "--max-steps",
                args.max_steps,
                "--baseline-return",
                args.eval_baseline_return,
                "--baseline-length",
                args.eval_baseline_length,
                "--baseline-fall",
                args.eval_baseline_fall,
            ]
        )
        rollout_cmd = command(
            [
                args.python_bin,
                "scripts/experiments/mjlab_qs/render_policy_rollout.py",
                "--policy-checkpoint",
                ckpt,
                "--output-dir",
                rollout_out,
                "--checkpoint-kind",
                name,
                "--device",
                args.device,
                "--rollout-episodes",
                args.rollout_episodes,
                "--max-steps",
                args.max_steps,
                "--baseline-return",
                args.rollout_baseline_return,
                "--baseline-length",
                args.rollout_baseline_length,
                "--baseline-fall",
                args.rollout_baseline_fall,
            ]
        )
        if args.wandb_project_eval:
            eval_cmd += (
                f" --wandb-project {shlex.quote(args.wandb_project_eval)}"
                f" --wandb-group {shlex.quote(args.wandb_group)}"
                f" --wandb-name {shlex.quote(args.wandb_group + '_' + name + '_eval40')}"
            )
        if args.wandb_project_rollout:
            rollout_cmd += (
                f" --wandb-project {shlex.quote(args.wandb_project_rollout)}"
                f" --wandb-group {shlex.quote(args.wandb_group)}"
                f" --wandb-name {shlex.quote(args.wandb_group + '_' + name + '_rollout1000_ep10')}"
            )
        rows.append(
            {
                "candidate": name,
                "checkpoint": str(ckpt),
                "eval_output_dir": str(eval_out),
                "rollout_output_dir": str(rollout_out),
                "eval_baseline_return": str(args.eval_baseline_return),
                "eval_baseline_length": str(args.eval_baseline_length),
                "eval_baseline_fall": str(args.eval_baseline_fall),
                "rollout_baseline_return": str(args.rollout_baseline_return),
                "rollout_baseline_length": str(args.rollout_baseline_length),
                "rollout_baseline_fall": str(args.rollout_baseline_fall),
                "eval_command": eval_cmd,
                "rollout_command": rollout_cmd,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "candidate",
        "checkpoint",
        "eval_output_dir",
        "rollout_output_dir",
        "eval_baseline_return",
        "eval_baseline_length",
        "eval_baseline_fall",
        "rollout_baseline_return",
        "rollout_baseline_length",
        "rollout_baseline_fall",
        "eval_command",
        "rollout_command",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_eval_manifest(path: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    fields = [
        "stage",
        "candidate",
        "policy_checkpoint",
        "checkpoint_kind",
        "eval_output_dir",
        "eval_episodes",
        "eval_num_envs",
        "eval_max_steps",
        "eval_baseline_return",
        "eval_baseline_length",
        "eval_baseline_fall",
        "wandb_project",
        "wandb_group",
        "wandb_name",
    ]
    stage = args.wandb_group or Path(args.eval_dir).name
    manifest_rows = []
    for row in rows:
        candidate = row["candidate"]
        manifest_rows.append(
            {
                "stage": stage,
                "candidate": candidate,
                "policy_checkpoint": row["checkpoint"],
                "checkpoint_kind": candidate,
                "eval_output_dir": row["eval_output_dir"],
                "eval_episodes": str(args.eval_episodes),
                "eval_num_envs": str(args.eval_num_envs),
                "eval_max_steps": str(args.max_steps),
                "eval_baseline_return": row["eval_baseline_return"],
                "eval_baseline_length": row["eval_baseline_length"],
                "eval_baseline_fall": row["eval_baseline_fall"],
                "wandb_project": args.wandb_project_eval,
                "wandb_group": args.wandb_group,
                "wandb_name": f"{args.wandb_group}_{candidate}_eval{args.eval_episodes}" if args.wandb_group else "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)


def write_rollout_manifest(path: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    fields = [
        "stage",
        "candidate",
        "policy_checkpoint",
        "checkpoint_kind",
        "rollout_output_dir",
        "rollout_episodes",
        "rollout_max_steps",
        "rollout_baseline_return",
        "rollout_baseline_length",
        "rollout_baseline_fall",
        "wandb_project",
        "wandb_group",
        "wandb_name",
    ]
    stage = args.wandb_group or Path(args.rollout_dir).name
    manifest_rows = []
    for row in rows:
        candidate = row["candidate"]
        manifest_rows.append(
            {
                "stage": stage,
                "candidate": candidate,
                "policy_checkpoint": row["checkpoint"],
                "checkpoint_kind": candidate,
                "rollout_output_dir": row["rollout_output_dir"],
                "rollout_episodes": str(args.rollout_episodes),
                "rollout_max_steps": str(args.max_steps),
                "rollout_baseline_return": row["rollout_baseline_return"],
                "rollout_baseline_length": row["rollout_baseline_length"],
                "rollout_baseline_fall": row["rollout_baseline_fall"],
                "wandb_project": args.wandb_project_rollout,
                "wandb_group": args.wandb_group,
                "wandb_name": (
                    f"{args.wandb_group}_{candidate}_rollout{args.max_steps}_ep{args.rollout_episodes}"
                    if args.wandb_group
                    else ""
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)


def write_shell(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export MUJOCO_GL=${MUJOCO_GL:-egl}",
        "export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}",
        "export EGL_PLATFORM=${EGL_PLATFORM:-surfaceless}",
        "export PYTHONPATH=$PWD/src:$PWD:${PYTHONPATH:-}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"# candidate: {row['candidate']}",
                row["eval_command"],
                row["rollout_command"],
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_csv(Path(args.output_csv), rows)
    write_shell(Path(args.output_sh), rows)
    if args.output_eval_manifest:
        write_eval_manifest(Path(args.output_eval_manifest), rows, args)
    if args.output_rollout_manifest:
        write_rollout_manifest(Path(args.output_rollout_manifest), rows, args)
    print(f"wrote {len(rows)} candidate rows to {args.output_csv} and {args.output_sh}")


if __name__ == "__main__":
    main()
