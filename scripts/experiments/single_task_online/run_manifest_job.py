#!/usr/bin/env python3
"""Run one manifest row: train -> eval -> rollout summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def has_hydra_override(overrides: List[str], key: str) -> bool:
    return any(token.startswith(f"{key}=") for token in overrides)


TASK_FALLBACK_OVERRIDES: Dict[str, str] = {
    # Keep horizon aligned with short-trajectory tasks during unstable phases.
    "hopper": "alg.horizon=7",
    "ant": "alg.horizon=14",
    "anymal": "alg.horizon=7",
    "humanoid": "alg.horizon=13",
    "snu_humanoid": "alg.horizon=13",
    "leap_left_grasp_asymmetric": "alg.horizon=13",
    "tracking_rough_unitree_g1": "alg.horizon=1",
}


def hydra_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def detect_gpu_type() -> str:
    """Auto-detect GPU type from CUDA device properties or Slurm GRES."""
    # Try Slurm GRES first (e.g. gpu:h100:1)
    gres = os.environ.get("SLURM_JOB_GPUS", "") or os.environ.get("SLURM_GRES", "")
    for keyword in ("h100", "h200", "l40s", "a100", "v100", "a40", "rtx"):
        if keyword in gres.lower():
            return keyword.upper()
    # Try Slurm node name pattern
    node = os.environ.get("SLURMD_NODENAME", "")
    if node:
        # ICE node naming may not encode GPU type; fall through to CUDA
        pass
    # Try CUDA device name
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            for keyword, label in [
                ("h100", "H100"), ("h200", "H200"), ("l40s", "L40S"),
                ("a100", "A100"), ("v100", "V100"), ("a40", "A40"),
            ]:
                if keyword in gpu_name:
                    return label
            return gpu_name.replace(" ", "_")[:20]
    except Exception:
        pass
    return "unknown"


def stable_wandb_run_id(project: str, run_key: str, job_type: str) -> str:
    key = f"{project}::{job_type}::{run_key}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def derive_method_families(method_key: str) -> tuple[str, str]:
    if method_key == "mlpwm_mlppolicy":
        return "mlp", "mlp"
    if method_key == "flowwm_mlppolicy":
        return "flow", "mlp"
    if method_key == "mlpwm_flowpolicy":
        return "mlp", "flow"
    if method_key == "flowwm_flowpolicy":
        return "flow", "flow"
    if "flowwm" in method_key:
        return "flow", "unknown"
    if "flowpolicy" in method_key:
        return "unknown", "flow"
    return "unknown", "unknown"


def derive_stage_metadata(stage: str, suite: str, method_key: str) -> dict[str, str]:
    wm_family, policy_family = derive_method_families(method_key)
    strict_stage = "strict" in stage
    original_pwm_aligned_stage = stage in {
        "confirm_fair_small",
        "confirm_fair_small_mjlab3",
        "confirm_fair_small_mjlab_extra3",
        "confirm_fair_small_mjlab6_strict",
        "confirm_mjlab_curriculum4_strict_default",
        "confirm_fair_small_tracking_rough_strict",
        "confirm_tracking_rough_strict",
        "confirm_mjlab_curriculum4_strict_default_h100shadow",
    }
    if original_pwm_aligned_stage:
        alignment_status = "original_pwm_aligned"
    elif strict_stage:
        alignment_status = "strict_exploratory_aligned"
    else:
        alignment_status = "exploratory_or_historical"

    if stage.startswith("confirm"):
        comparison_role = "fair_compare" if original_pwm_aligned_stage else "historical_confirm"
    else:
        comparison_role = "strict_exploratory" if strict_stage else "exploratory"

    task_resolution_policy = "strict_exact_task" if suite == "mjlab" and strict_stage else "default_resolution"
    env_family = "mjlab" if suite.startswith("mjlab") or suite == "mjlab" else "gym_mujoco"

    return {
        "alignment_status": alignment_status,
        "comparison_role": comparison_role,
        "task_resolution_policy": task_resolution_policy,
        "env_family": env_family,
        "wm_family": wm_family,
        "policy_family": policy_family,
    }


def build_wandb_tags(
    *,
    stage: str,
    suite: str,
    task_key: str,
    method_key: str,
    hparam_profile: str,
    seed: int,
    gpu_type: str,
    slurm_partition: str,
) -> list[str]:
    metadata = derive_stage_metadata(stage, suite, method_key)
    tags = [
        "single_task_online",
        "online_rl",
        "from_scratch",
        f"stage_{stage}",
        f"suite_{suite}",
        f"task_{task_key}",
        f"method_{method_key}",
        f"profile_{hparam_profile}",
        f"seed_{seed}",
        f"gpu_{gpu_type}",
        f"alignment_{metadata['alignment_status']}",
        f"comparison_{metadata['comparison_role']}",
        f"envfamily_{metadata['env_family']}",
        f"wm_{metadata['wm_family']}",
        f"policy_{metadata['policy_family']}",
        f"taskmatch_{metadata['task_resolution_policy']}",
    ]
    if slurm_partition:
        tags.append(f"partition_{slurm_partition}")
    return tags


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


def checkpoint_is_loadable(path: Path) -> bool:
    try:
        import torch
        try:
            torch.load(path, map_location="cpu")
            return True
        except Exception as exc:
            # PyTorch 2.6+ may fail on older full-object checkpoints when
            # weights_only=True is implicitly used. Retry with weights_only=False
            # before declaring the checkpoint corrupt.
            msg = str(exc)
            if "Weights only load failed" in msg:
                try:
                    torch.load(path, map_location="cpu", weights_only=False)
                    return True
                except Exception as exc2:
                    print(f"Warning: invalid checkpoint file {path}: {exc2}")
                    return False
            print(f"Warning: invalid checkpoint file {path}: {exc}")
            return False
    except Exception as exc:
        print(f"Warning: could not validate checkpoint {path}: {exc}")
        return False


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

    # Detect GPU type and Slurm metadata for detailed WandB annotation
    gpu_type = detect_gpu_type()
    slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", ""))
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    slurm_node = os.environ.get("SLURMD_NODENAME", "")
    slurm_partition = os.environ.get("SLURM_JOB_PARTITION", "")

    wandb_base_tags = build_wandb_tags(
        stage=stage,
        suite=suite,
        task_key=task_key,
        method_key=method_key,
        hparam_profile=hparam_profile,
        seed=seed,
        gpu_type=gpu_type,
        slurm_partition=slurm_partition,
    )
    wandb_tag_list = list(wandb_base_tags)
    wandb_tag_list.extend(["purpose_eval", "job_eval"])
    stage_metadata = derive_stage_metadata(stage, suite, method_key)
    wandb_train_id = stable_wandb_run_id(row["wandb_project"], run_key, "train")
    wandb_eval_id = stable_wandb_run_id(row["wandb_project"], run_key, "eval")

    print(f"[run_manifest_job] GPU={gpu_type} node={slurm_node} "
          f"job={slurm_job_id}_{slurm_task_id} run_key={run_key}")

    resume_if_exists = os.environ.get("RESUME_IF_EXISTS", "1") != "0"
    latest_ckpt = logs_dir / "latest_checkpoint.pt"
    final_ckpt = logs_dir / "final_policy.pt"
    best_ckpt = logs_dir / "best_policy.pt"
    checkpoint_path = ""
    resume_training = False
    checkpoint_with_buffer = False
    skip_train_if_final = os.environ.get("SKIP_TRAIN_IF_FINAL", "1") != "0"
    force_reeval_if_final = os.environ.get("FORCE_REEVAL_IF_FINAL", "0") == "1"
    validate_checkpoints = os.environ.get("VALIDATE_CHECKPOINTS", "1") != "0"
    skip_train = False
    ckpt_valid_cache: Dict[Path, bool] = {}

    def is_ckpt_valid(path: Path) -> bool:
        if not validate_checkpoints:
            return True
        if path not in ckpt_valid_cache:
            ckpt_valid_cache[path] = checkpoint_is_loadable(path)
        return ckpt_valid_cache[path]

    if resume_if_exists:
        for candidate in (latest_ckpt, final_ckpt, best_ckpt):
            if candidate.exists():
                if not is_ckpt_valid(candidate):
                    continue
                checkpoint_path = str(candidate)
                resume_training = True
                checkpoint_with_buffer = candidate.with_suffix(".buffer").exists()
                print(
                    f"Resuming run {run_key} from checkpoint={checkpoint_path} "
                    f"(with_buffer={checkpoint_with_buffer})"
                )
                break
    if skip_train_if_final and final_ckpt.exists():
        if is_ckpt_valid(final_ckpt):
            skip_train = True
            checkpoint_path = str(final_ckpt)
            resume_training = False
            checkpoint_with_buffer = False
            print(
                f"Detected completed training for {run_key} "
                f"(final checkpoint: {final_ckpt}). Skipping train step."
            )
        else:
            print(
                f"Final checkpoint exists but is invalid for {run_key}: {final_ckpt}. "
                "Will continue with resume/start-over path."
            )

    row_overrides = split_overrides(row.get("overrides", ""))
    if stage == "smoke" and not has_hydra_override(row_overrides, "alg.horizon"):
        smoke_force_horizon = os.environ.get("SMOKE_FORCE_HORIZON", "1").strip()
        if smoke_force_horizon:
            row_overrides.append(f"alg.horizon={smoke_force_horizon}")
    if not has_hydra_override(row_overrides, "alg.horizon"):
        fallback_horizon = TASK_FALLBACK_OVERRIDES.get(task_key, "")
        if fallback_horizon:
            row_overrides.append(fallback_horizon)
            print(f"Applied fallback task override for {task_key}: {fallback_horizon}")

    max_epochs = row["max_epochs"]
    if stage == "smoke":
        smoke_max_epochs = os.environ.get("SMOKE_MAX_EPOCHS_OVERRIDE", "").strip()
        if smoke_max_epochs:
            max_epochs = smoke_max_epochs

    notes_text = row['notes']
    enriched_notes = f"{notes_text} | GPU={gpu_type} node={slurm_node} job={slurm_job_id}"

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
        f"alg.max_epochs={max_epochs}",
        f"env.config.num_envs={row['num_envs']}",
        f"hydra.run.dir={output_dir}",
        f"++wandb.project={row['wandb_project']}",
        f"++wandb.group={row['wandb_group']}",
        f"++wandb.job_type=train",
        f"++wandb.name={run_key}",
        f"++wandb.id={wandb_train_id}",
        "++wandb.resume=allow",
        f"++wandb.notes={hydra_quote(enriched_notes)}",
        f"++experiment.run_key={run_key}",
        f"++experiment.stage={stage}",
        f"++experiment.suite={suite}",
        f"++experiment.task={task_key}",
        f"++experiment.method={method_key}",
        f"++experiment.hparam_profile={hparam_profile}",
        f"++experiment.gpu_type={gpu_type}",
        f"++experiment.alignment_status={stage_metadata['alignment_status']}",
        f"++experiment.comparison_role={stage_metadata['comparison_role']}",
        f"++experiment.env_family={stage_metadata['env_family']}",
        f"++experiment.wm_family={stage_metadata['wm_family']}",
        f"++experiment.policy_family={stage_metadata['policy_family']}",
        f"++experiment.task_resolution_policy={stage_metadata['task_resolution_policy']}",
        f"++wandb.tags={hydra_quote(','.join(wandb_base_tags + ['purpose_train', 'job_train']))}",
        f"++experiment.slurm_job_id={slurm_job_id}",
        f"++experiment.slurm_node={slurm_node}",
    ]
    train_cmd.extend(row_overrides)

    if not skip_train:
        run_command(train_cmd, cwd=project_root)

    if args.skip_eval or os.environ.get("SKIP_EVAL", "0") == "1":
        print("Skipping evaluation as requested.")
        return

    eval_output_dir = output_dir / "eval"
    eval_summary_path = eval_output_dir / "eval_summary.json"
    if skip_train and eval_summary_path.exists() and not force_reeval_if_final:
        print(
            f"Detected existing evaluation for completed run {run_key} "
            f"({eval_summary_path}). Skipping eval step."
        )
        return

    best_ckpt = output_dir / "logs" / "best_policy.pt"
    final_ckpt = output_dir / "logs" / "final_policy.pt"
    checkpoint = best_ckpt if best_ckpt.exists() else final_ckpt
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Expected checkpoint not found. Missing both {best_ckpt} and {final_ckpt}"
        )

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
        "--wandb-job-type",
        "eval",
        "--wandb-id",
        wandb_eval_id,
        "--wandb-resume",
        "allow",
        "--wandb-tags",
        ",".join(wandb_tag_list),
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
