#!/usr/bin/env python3
"""Train MJLab-native RSL-RL/PPO collectors for MJLab-QS.

This runner intentionally stays outside the Flow/PWM comparison. It trains
neutral MJLab-native policies that can later be audited as data collectors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import types


def patch_mujoco_compatibility() -> None:
    """Patch MuJoCo enum drift before importing MJLab / MuJoCo-Warp."""
    try:
        import mujoco  # type: ignore

        enable_bits = getattr(mujoco, "mjtEnableBit", None)
        if enable_bits is not None and not hasattr(enable_bits, "mjENBL_MULTICCD"):
            setattr(enable_bits, "mjENBL_MULTICCD", 0)
    except Exception:
        return


def patch_headless_display_dependency() -> None:
    """Provide a tiny IPython.display shim for mediapy in headless jobs."""
    try:
        import IPython.display  # type: ignore  # noqa: F401

        return
    except Exception:
        pass

    ipython_mod = types.ModuleType("IPython")
    display_mod = types.ModuleType("IPython.display")

    class HTML:
        def __init__(self, data=None, *args, **kwargs):
            self.data = data

    def display(*args, **kwargs):
        return None

    display_mod.HTML = HTML
    display_mod.display = display
    ipython_mod.display = display_mod
    sys.modules.setdefault("IPython", ipython_mod)
    sys.modules.setdefault("IPython.display", display_mod)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--method", required=True, choices=["rslrl_ppo_default", "rslrl_ppo_conservative"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--max-iterations", type=int, default=-1)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--wandb-project", default="flow-mbpo-mjlab-native-collector")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--logger", choices=["wandb", "tensorboard"], default="wandb")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def apply_method(cfg, method: str) -> None:
    """Apply collector method variant in-place."""
    if method == "rslrl_ppo_default":
        return

    if method == "rslrl_ppo_conservative":
        # Keep MJLab-native PPO, but reduce optimizer aggressiveness and remove
        # exogenous perturbation/randomization events for a more canonical flat
        # velocity collector.
        cfg.agent.algorithm.learning_rate = 3.0e-4
        cfg.agent.algorithm.desired_kl = 0.008
        cfg.agent.algorithm.entropy_coef = 0.005
        cfg.agent.actor.obs_normalization = True
        cfg.agent.critic.obs_normalization = True
        for event_name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
            cfg.env.events.pop(event_name, None)
        return

    raise ValueError(f"Unsupported method: {method}")


def set_num_envs(env_cfg, num_envs: int) -> None:
    env_cfg.scene.num_envs = int(num_envs)
    terrain = getattr(env_cfg.scene, "terrain", None)
    if terrain is not None:
        terrain.num_envs = int(num_envs)
        generator = getattr(terrain, "terrain_generator", None)
        if generator is not None and hasattr(generator, "num_envs"):
            generator.num_envs = int(num_envs)


def main() -> None:
    args = parse_args()
    patch_mujoco_compatibility()
    patch_headless_display_dependency()

    import mjlab.tasks  # noqa: F401  # Populate task registry.
    from mjlab.scripts.train import TrainConfig, run_train

    cfg = TrainConfig.from_task(args.task_id)
    cfg.agent.seed = int(args.seed)
    cfg.env.seed = int(args.seed)
    cfg.agent.logger = args.logger
    cfg.agent.wandb_project = args.wandb_project
    cfg.agent.run_name = args.run_name or f"{args.method}_seed{args.seed}"
    cfg.agent.experiment_name = "mjlab_qs_native_collectors"
    cfg.agent.save_interval = int(args.save_interval)
    if args.max_iterations > 0:
        cfg.agent.max_iterations = int(args.max_iterations)
    set_num_envs(cfg.env, args.num_envs)
    apply_method(cfg, args.method)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and any(output_dir.glob("model_*.pt")):
        cfg.agent.resume = True
        cfg.agent.load_run = output_dir.name
        cfg.agent.load_checkpoint = "model_.*.pt"

    print(
        "[mjlab-native-collector] "
        f"task_id={args.task_id} method={args.method} seed={args.seed} "
        f"num_envs={args.num_envs} max_iterations={cfg.agent.max_iterations} "
        f"output_dir={output_dir} resume={cfg.agent.resume}"
    )
    run_train(args.task_id, cfg, output_dir)


if __name__ == "__main__":
    main()
