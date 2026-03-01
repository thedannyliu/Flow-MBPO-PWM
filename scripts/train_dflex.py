# Fix torch.onnx import error by mocking the missing import
import sys
from unittest.mock import MagicMock

# Create a mock for the problematic module
mock_exporter = MagicMock()
mock_exporter.DiagnosticOptions = MagicMock
mock_exporter.ExportOutput = MagicMock
sys.modules['torch.onnx._internal.exporter'] = mock_exporter

import hydra, os, wandb, yaml
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from flow_mbpo_pwm.utils import hydra_utils
from flow_mbpo_pwm.utils.common import seeding
from hydra.utils import instantiate

try:
    from IPython.core import ultratb

    # Keep rich tracebacks when IPython is available; avoid interactive pdb on cluster jobs.
    sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=0)
except Exception:
    pass


def _resolve_dflex_cuda_arch() -> str:
    """
    Resolve CUDA arch used by dflex kernel JIT.

    dflex hardcodes `compute_50`, which fails on modern H100/H200 toolchains.
    We derive the active GPU capability at runtime and fall back to sm_90.
    """
    arch_override = os.environ.get("DFLEX_CUDA_ARCH", "").strip()
    if arch_override:
        return arch_override.replace(".", "").replace("sm_", "")

    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}{minor}"
    except Exception:
        pass
    return "90"


def _patch_dflex_load_inline() -> None:
    """
    Monkey-patch torch load_inline to rewrite dflex's legacy CUDA flags.

    This patch is intentionally narrow: only dflex's `kernels` extension is
    affected, leaving other JIT extensions untouched.
    """
    try:
        from torch.utils import cpp_extension
    except Exception:
        return

    if getattr(cpp_extension.load_inline, "_flow_mbpo_dflex_patch", False):
        return

    original_load_inline = cpp_extension.load_inline

    def _patched_load_inline(*args, **kwargs):
        name = args[0] if args else kwargs.get("name", "")
        if name == "kernels":
            cuda_flags = list(kwargs.get("extra_cuda_cflags", []) or [])
            if any("compute_50" in str(flag) for flag in cuda_flags):
                arch = _resolve_dflex_cuda_arch()
                rewritten_flags = [
                    f"-gencode=arch=compute_{arch},code=sm_{arch}",
                    f"-gencode=arch=compute_{arch},code=compute_{arch}",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ]
                print(
                    f"[dflex patch] Rewriting CUDA flags for kernels: "
                    f"{cuda_flags} -> {rewritten_flags}"
                )
                kwargs["extra_cuda_cflags"] = rewritten_flags

            # dflex adds Windows-only `-Z` flag on Linux; remove it.
            extra_cflags = list(kwargs.get("extra_cflags", []) or [])
            if "-Z" in extra_cflags:
                kwargs["extra_cflags"] = [flag for flag in extra_cflags if flag != "-Z"]

        return original_load_inline(*args, **kwargs)

    _patched_load_inline._flow_mbpo_dflex_patch = True
    cpp_extension.load_inline = _patched_load_inline


def _normalize_wandb_tags(tags):
    if tags is None:
        return None
    if isinstance(tags, str):
        if not tags.strip():
            return None
        return [tag.strip() for tag in tags.split(",") if tag.strip()]
    if isinstance(tags, (list, tuple)):
        normalized = [str(tag).strip() for tag in tags if str(tag).strip()]
        return normalized or None
    return [str(tags)]


def create_wandb_run(wandb_cfg, job_config, run_id=None):
    """Create a WandB run with proper naming.
    
    Supports name and notes from config override (e.g., ++wandb.name=XXX).
    Auto-generates tags from experiment.* config fields for clear experiment
    identification in the WandB dashboard.
    """
    env_name = job_config["env"]["config"]["_target_"].split(".")[-1]
    try:
        alg_name = job_config["alg"]["_target_"].split(".")[-1]
    except:
        alg_name = job_config["alg"]["name"].upper()
    
    # Get seed for naming
    seed = job_config.get('general', {}).get('seed', 42)
    
    # Use wandb.name from config if provided, otherwise generate default
    if hasattr(wandb_cfg, 'name') and wandb_cfg.name:
        name = wandb_cfg.name
    else:
        try:
            # Multirun config
            job_id = HydraConfig().get().job.num
            name = f"{alg_name}_{env_name}_sweep_{seed}"
        except:
            # Normal (singular) run config - include seed for uniqueness
            name = f"{alg_name}_{env_name}_s{seed}"
    
    # Get notes from config
    notes = getattr(wandb_cfg, 'notes', '') if hasattr(wandb_cfg, 'notes') else ''
    job_type = getattr(wandb_cfg, "job_type", "train")

    # Auto-build tags from experiment.* config fields for clear WandB identification.
    # This avoids Hydra override issues with list-valued tags.
    experiment = job_config.get("experiment", {})
    tags = []
    for key in ("stage", "suite", "task", "method", "hparam_profile", "gpu_type"):
        val = experiment.get(key, "")
        if val:
            tags.append(f"{key}_{val}")
    seed_val = experiment.get("run_key", "")
    if seed_val:
        # Extract seed from run_key (e.g., smoke_gym_hopper_mlpwm_mlppolicy_s0_default)
        import re
        seed_match = re.search(r'_s(\d+)_', seed_val)
        if seed_match:
            tags.append(f"seed_{seed_match.group(1)}")
    # Add base project tags
    tags.extend(["single_task_online", "online_rl", "from_scratch"])
    # Merge with any explicit tags from config (if provided)
    explicit_tags = _normalize_wandb_tags(getattr(wandb_cfg, "tags", None))
    if explicit_tags:
        tags.extend(t for t in explicit_tags if t not in tags)

    # Record scheduler/runtime metadata in config for reproducibility.
    job_config.setdefault("runtime", {})
    job_config["runtime"]["slurm"] = {
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "node_name": os.environ.get("SLURMD_NODENAME", ""),
        "cluster_name": os.environ.get("SLURM_CLUSTER_NAME", ""),
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
    }
    
    print(f"Initializing WandB run: name='{name}', project='{wandb_cfg.project}', tags={tags}")
    
    return wandb.init(
        project=wandb_cfg.project,
        config=job_config,
        group=wandb_cfg.group if wandb_cfg.group else None,
        entity=wandb_cfg.entity if wandb_cfg.entity else None,
        tags=tags,
        job_type=job_type,
        name=name,
        notes=notes,
        id=run_id,
        resume=run_id is not None,
    )


@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    _patch_dflex_load_inline()
    cfg_full = OmegaConf.to_container(cfg, resolve=True)

    if cfg.general.run_wandb:
        create_wandb_run(cfg.wandb, cfg_full)

    # patch code to make jobs log in the correct directory when doing multirun
    logdir = HydraConfig.get()["runtime"]["output_dir"]
    logdir = os.path.join(logdir, cfg.general.logdir)

    seeding(cfg.general.seed, False)

    if "SHAC" in cfg.alg._target_ or "AHAC" in cfg.alg._target_:
        cfg.env.config.no_grad = False
    else:
        cfg.env.config.no_grad = True
    print(f"Running sim with no_grad={cfg.env.config.no_grad}")

    env = instantiate(cfg.env.config, logdir=logdir)
    print("num_envs = ", env.num_envs)
    print("num_actions = ", env.num_actions)
    print("num_obs = ", env.num_obs)

    agent = instantiate(
        cfg.alg,
        env=env,
        obs_dim=env.num_obs,
        act_dim=env.num_actions,
        logdir=logdir,
        log=cfg.general.run_wandb,
    )

    if cfg.general.checkpoint:
        agent.load(
            cfg.general.checkpoint, 
            buffer=cfg.general.checkpoint_with_buffer,
            resume_training=cfg.general.resume_training
        )
        agent.wm_bootstrapped = True

    if cfg.general.pretrain:
        actually_train = True if not cfg.general.checkpoint else False
        agent.pretrain_wm(
            cfg.general.pretrain, cfg.general.pretrain_steps, actually_train
        )

    # Quick self-check: test eval() before starting full training
    if cfg.general.train and cfg.alg.max_epochs > 100:
        print("\n" + "=" * 80)
        print("Running quick self-check: testing eval() function...")
        print("=" * 80)
        try:
            # Test eval with 2 games - this catches the reward shape bug
            test_loss, test_disc_loss, test_len = agent.eval(num_games=2, deterministic=True)
            print(f"✓ Self-check PASSED: eval() works correctly")
            print(f"  Test results: loss={test_loss:.2f}, disc_loss={test_disc_loss:.2f}, len={test_len:.1f}")
        except Exception as e:
            print(f"✗ Self-check FAILED: {e}")
            print("\nABORTING training to save compute time!")
            print("Please fix the bug before resubmitting.")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Self-check failed, eval() is broken: {e}")
        
        print("Self-check completed, starting full training...")
        print("=" * 80 + "\n")
    
    if cfg.general.train:
        agent.train()

    # evaluate the final policy's performance
    loss, discounted_loss, ep_len = agent.eval(cfg.general.eval_runs)
    print(
        f"mean episode loss = {loss:.2f}, mean discounted loss = {discounted_loss:.2f}, mean episode length = {ep_len:.2f}"
    )

    if cfg.general.run_wandb:
        final_eval_metrics = {
            "eval/final_episode_loss": float(loss),
            "eval/final_discounted_loss": float(discounted_loss),
            "eval/final_episode_length": float(ep_len),
            "eval/final_mean_reward_proxy": float(-loss),
            "eval/final_num_games": float(cfg.general.eval_runs),
        }
        if hasattr(agent, "step_count"):
            wandb.log(final_eval_metrics, step=agent.step_count)
        else:
            wandb.log(final_eval_metrics)

    if cfg.general.run_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
