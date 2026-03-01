#!/usr/bin/env python3
"""Run one manifest row: train -> eval -> rollout summary."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def load_row(manifest_path: Path, row_index: int) -> Dict[str, str]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index={row_index} is out of range [0, {len(rows)-1}]")
    return rows[row_index]


def split_overrides(raw: str) -> List[str]:
    if not raw:
        return []
    return [token.strip() for token in raw.split(";") if token.strip()]


def hydra_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def run_command(
    cmd: List[str],
    cwd: Path,
    *,
    env_overrides: Dict[str, str] | None = None,
    allow_failure: bool = False,
) -> int:
    print("Running command:")
    print("  " + " ".join(shlex.quote(token) for token in cmd))
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
        print(
            "  with env overrides: "
            + ", ".join(f"{k}={v}" for k, v in sorted(env_overrides.items()))
        )
    result = subprocess.run(cmd, cwd=str(cwd), check=not allow_failure, env=env)
    return int(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one single-task-online manifest entry.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    row = load_row(manifest_path, args.row_index)

    run_key = row["run_key"]
    stage = row["stage"]
    suite = row["suite"]
    task_key = row["task_key"]
    method_key = row["method_key"]
    seed = int(row["seed"])
    hparam_profile = row["hparam_profile"]

    output_dir = (
        project_root
        / "scripts"
        / "outputs"
        / "single_task_online"
        / stage
        / suite
        / task_key
        / method_key
        / f"seed_{seed}"
        / hparam_profile
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"

    wandb_tag_list = [
        "single_task_online",
        "online_rl",
        "from_scratch",
        f"stage_{stage}",
        f"suite_{suite}",
        f"task_{task_key}",
        f"method_{method_key}",
        f"profile_{hparam_profile}",
        f"seed_{seed}",
    ]

    resume_if_exists = os.environ.get("RESUME_IF_EXISTS", "1") != "0"
    latest_ckpt = logs_dir / "latest_checkpoint.pt"
    final_ckpt = logs_dir / "final_policy.pt"
    best_ckpt = logs_dir / "best_policy.pt"
    checkpoint_path = ""
    resume_training = False
    checkpoint_with_buffer = False

    if resume_if_exists:
        for candidate in (latest_ckpt, final_ckpt, best_ckpt):
            if candidate.exists():
                checkpoint_path = str(candidate)
                resume_training = True
                checkpoint_with_buffer = candidate.with_suffix(".buffer").exists()
                print(
                    f"Resuming run {run_key} from checkpoint={checkpoint_path} "
                    f"(with_buffer={checkpoint_with_buffer})"
                )
                break

    train_cmd = [
        args.python_bin,
        "scripts/train_dflex.py",
        f"env={row['env']}",
        f"alg={row['alg']}",
        f"general.seed={seed}",
        "general.train=true",
        f"general.checkpoint={checkpoint_path}",
        "general.pretrain=",
        f"general.resume_training={'true' if resume_training else 'false'}",
        f"general.checkpoint_with_buffer={'true' if checkpoint_with_buffer else 'false'}",
        "general.run_wandb=true",
        f"general.eval_runs={row['eval_runs']}",
        f"alg.max_epochs={row['max_epochs']}",
        f"env.config.num_envs={row['num_envs']}",
        f"hydra.run.dir={output_dir}",
        f"++wandb.project={row['wandb_project']}",
        f"++wandb.group={row['wandb_group']}",
        f"++wandb.job_type=train",
        f"++wandb.name={run_key}",
        f"++wandb.notes={hydra_quote(row['notes'])}",
        f"++experiment.run_key={run_key}",
        f"++experiment.stage={stage}",
        f"++experiment.suite={suite}",
        f"++experiment.task={task_key}",
        f"++experiment.method={method_key}",
        f"++experiment.hparam_profile={hparam_profile}",
    ]
    train_cmd.extend(split_overrides(row.get("overrides", "")))

    run_command(train_cmd, cwd=project_root)

    if args.skip_eval or os.environ.get("SKIP_EVAL", "0") == "1":
        print("Skipping evaluation as requested.")
        return

    best_ckpt = output_dir / "logs" / "best_policy.pt"
    final_ckpt = output_dir / "logs" / "final_policy.pt"
    checkpoint = best_ckpt if best_ckpt.exists() else final_ckpt
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Expected checkpoint not found. Missing both {best_ckpt} and {final_ckpt}"
        )

    eval_output_dir = output_dir / "eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    eval_cmd = [
        args.python_bin,
        "scripts/eval/eval_online_single_task.py",
        "--checkpoint",
        str(checkpoint),
        "--num-games",
        row["eval_runs"],
        "--num-envs",
        "16",
        "--device",
        "cuda:0",
        "--output-dir",
        str(eval_output_dir),
        "--rollout-episodes",
        row["rollout_episodes"],
        "--rollout-max-steps",
        row["rollout_max_steps"],
        "--wandb-project",
        row["wandb_project"],
        "--wandb-group",
        row["wandb_group"],
        "--wandb-name",
        f"{run_key}_eval",
        "--wandb-tags",
        ",".join(wandb_tag_list + ["job_eval"]),
    ]
    eval_env = {
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
        "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM", "egl"),
        "EGL_PLATFORM": os.environ.get("EGL_PLATFORM", "surfaceless"),
    }
    enable_video = os.environ.get("ENABLE_ROLLOUT_VIDEO", "1") != "0"
    strict_video = os.environ.get("STRICT_EVAL_VIDEO", "0") == "1"
    require_mp4 = os.environ.get("REQUIRE_EVAL_MP4", "0") == "1"
    # Default off: rollout videos are large and should not be uploaded unless requested.
    wandb_log_video = os.environ.get("WANDB_LOG_EVAL_VIDEO", "0") != "0"
    wandb_log_artifact = os.environ.get("WANDB_LOG_EVAL_ARTIFACT", "1") != "0"
    eval_cmd_with_video = list(eval_cmd)
    if not wandb_log_video:
        eval_cmd.append("--no-wandb-log-video")
    if not wandb_log_artifact:
        eval_cmd.append("--no-wandb-log-artifact")
    if not wandb_log_video:
        eval_cmd_with_video.append("--no-wandb-log-video")
    if not wandb_log_artifact:
        eval_cmd_with_video.append("--no-wandb-log-artifact")
    if enable_video:
        eval_cmd_with_video.append("--save-video")
    if require_mp4:
        eval_cmd_with_video.append("--require-mp4")

    if enable_video:
        rc = run_command(
            eval_cmd_with_video,
            cwd=project_root,
            env_overrides=eval_env,
            allow_failure=True,
        )
        if rc != 0:
            if strict_video:
                raise subprocess.CalledProcessError(rc, eval_cmd_with_video)
            print(
                "Evaluation with video failed. Retrying without --save-video "
                f"(run_key={run_key}, rc={rc})."
            )
            rc = run_command(
                eval_cmd,
                cwd=project_root,
                env_overrides=eval_env,
                allow_failure=True,
            )
            if rc != 0:
                raise subprocess.CalledProcessError(rc, eval_cmd)
        if require_mp4:
            video_exists = (eval_output_dir / "rollout.mp4").exists()
        else:
            video_exists = (eval_output_dir / "rollout.mp4").exists() or (
                eval_output_dir / "rollout.gif"
            ).exists()
        if strict_video and not video_exists:
            raise RuntimeError(
                f"STRICT_EVAL_VIDEO=1 but required rollout video was not found "
                f"(run={run_key}, require_mp4={require_mp4})."
            )
        if not video_exists:
            print(
                f"Warning: evaluation completed but no rollout video artifact was produced "
                f"(run_key={run_key})."
            )
    else:
        run_command(eval_cmd, cwd=project_root, env_overrides=eval_env)
    print(f"Completed training + evaluation for {run_key}")


if __name__ == "__main__":
    main()
