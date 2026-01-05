import os
import time
from glob import glob
from pathlib import Path

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb

from flow_mbpo_pwm.utils.common import seeding
from common import TASK_SET


def _infer_task_set(data_dir: str):
    if "mt80" in str(data_dir).lower():
        return TASK_SET["mt80"], 96
    return TASK_SET["mt30"], 64


def _load_first_dataset_file(data_dir: str):
    fps = sorted(glob(str(Path(os.path.join(data_dir, "*.pt")))))
    if len(fps) == 0:
        raise ValueError(f"No data found at {data_dir} (expected *.pt files)")
    td = torch.load(fps[0], weights_only=False)
    return fps, td


def _extract_scalar_task_ids(task_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert TD-MPC2 task field to a 1D int64 tensor [B].
    Handles common encodings:
      - scalar per step: [T,B] or [B]
      - one-hot per step: [T,B,K] or [B,K]
    """
    if task_tensor.ndim == 0:
        return task_tensor.view(1).long()
    if task_tensor.ndim == 1:
        return task_tensor.long()
    if task_tensor.ndim >= 2 and task_tensor.shape[-1] > 1:
        return task_tensor.argmax(dim=-1).long().reshape(-1)
    return task_tensor.reshape(-1).long()


@hydra.main(config_path="cfg", config_name="pretrain_mt30_wm.yaml", version_base="1.2")
def pretrain(cfg):
    assert torch.cuda.is_available()
    if not cfg.general.data_dir:
        raise ValueError("Missing `general.data_dir` (TD-MPC2 multitask dataset path)")

    # Patch code to make jobs log in the correct directory when doing multirun
    hydra_out = HydraConfig.get()["runtime"]["output_dir"]
    logdir = os.path.join(hydra_out, cfg.general.logdir)
    os.makedirs(logdir, exist_ok=True)

    seeding(cfg.general.seed)

    task_set, task_dim = _infer_task_set(cfg.general.data_dir)
    cfg.tasks = task_set
    cfg.alg.world_model_config.tasks = task_set
    cfg.alg.world_model_config.task_dim = task_dim

    # Load dataset files (TD-MPC2 tensordict format)
    fps, td0 = _load_first_dataset_file(cfg.general.data_dir)
    obs_dim = td0["obs"].shape[-1]
    act_dim = td0["action"].shape[-1]

    # Some WM implementations expect `action_dims` for multitask setup (masks).
    cfg.alg.world_model_config.action_dims = [act_dim] * len(task_set)

    # TD-MPC2 episode lengths are fixed for MT30/MT80.
    episode_length = 101 if "mt80" in str(cfg.general.data_dir).lower() else 501
    cfg.episode_length = episode_length

    cfg.buffer.buffer_size = 24000 * episode_length
    buffer = instantiate(cfg.buffer)

    print(f"Found {len(fps)} files in {cfg.general.data_dir}")
    for fp in fps:
        print(f"Loading {fp}", flush=True)
        td = torch.load(fp, weights_only=False)
        if td.shape[1] != episode_length:
            raise ValueError(
                f"Expected episode length {episode_length} but got {td.shape[1]} in {fp}"
            )
        buffer.add_batch(td)

    if buffer.num_eps == 0:
        raise ValueError("Buffer is empty after loading data; check dataset path.")

    # Initialize WandB if enabled
    if cfg.general.run_wandb:
        wandb.init(
            project=cfg.wandb.get("project", "WM-Pretrain"),
            group=cfg.wandb.get("group", "wm_pretrain"),
            name=cfg.wandb.get("name", f"{cfg.general.out_name}_s{cfg.general.seed}"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Instantiate PWM agent (we will only train WM here)
    agent = instantiate(
        cfg.alg,
        obs_dim=obs_dim,
        act_dim=act_dim,
        env=None,
        logdir=logdir,
        max_epochs=0,
    )

    start_time = time.time()
    wm_pretrain_iters = int(cfg.general.wm_pretrain_iters)
    log_every = int(cfg.general.log_every)
    eval_every = int(cfg.general.get("eval_every", 1000))  # Validation frequency

    # Best/last checkpoint tracking
    best_wm_loss = float("inf")
    best_ckpt_path = os.path.join(logdir, f"{cfg.general.out_name}_best.pt")
    last_ckpt_path = os.path.join(logdir, f"{cfg.general.out_name}_last.pt")

    pbar = tqdm(range(wm_pretrain_iters), desc="WM pretrain", dynamic_ncols=True)
    for i in pbar:
        obs, act, rew, task = buffer.sample_with_task()
        task_ids = _extract_scalar_task_ids(task).to(agent.device)

        agent.wm_optimizer.zero_grad(set_to_none=True)
        wm_loss, dyn_loss, rew_loss = agent.compute_wm_loss(obs, act, rew, task_ids)
        wm_loss.backward()
        wm_grad_norm = torch.nn.utils.clip_grad_norm_(agent.wm.parameters(), agent.wm_grad_norm)
        agent.wm_optimizer.step()

        # Convert to Python floats for logging
        wm_loss_val = float(wm_loss.detach().cpu())
        dyn_loss_val = float(dyn_loss.detach().cpu()) if torch.is_tensor(dyn_loss) else float(dyn_loss)
        rew_loss_val = float(rew_loss.detach().cpu()) if torch.is_tensor(rew_loss) else float(rew_loss)
        grad_norm_val = float(wm_grad_norm.detach().cpu()) if torch.is_tensor(wm_grad_norm) else float(wm_grad_norm)

        if i % log_every == 0:
            elapsed = time.time() - start_time
            pbar.set_postfix(
                wm_loss=wm_loss_val,
                dyn=dyn_loss_val,
                rew=rew_loss_val,
                gn=grad_norm_val,
                sec=int(elapsed),
            )

            # Log to WandB
            if cfg.general.run_wandb:
                wandb.log({
                    "train/wm_loss": wm_loss_val,
                    "train/dynamics_loss": dyn_loss_val,
                    "train/reward_loss": rew_loss_val,
                    "train/grad_norm": grad_norm_val,
                    "train/iter": i,
                    "train/elapsed_sec": elapsed,
                    "train/lr": agent.wm_optimizer.param_groups[0]["lr"],
                }, step=i)

        # Validation / Best checkpoint evaluation
        if (i + 1) % eval_every == 0:
            # Use training loss as proxy for "best" (no separate val set in offline pretraining)
            # In practice, we track best training loss to save the most converged model
            if wm_loss_val < best_wm_loss:
                best_wm_loss = wm_loss_val
                _save_checkpoint(agent, best_ckpt_path, i + 1, obs_dim, act_dim, cfg, task_set, is_best=True)
                print(f"\n[Best] Saved best WM checkpoint (loss={best_wm_loss:.4f}): {best_ckpt_path}", flush=True)
                if cfg.general.run_wandb:
                    wandb.log({"train/best_wm_loss": best_wm_loss}, step=i)

    # Save last checkpoint
    _save_checkpoint(agent, last_ckpt_path, wm_pretrain_iters, obs_dim, act_dim, cfg, task_set, is_best=False)
    print(f"\n[Last] Saved final WM checkpoint: {last_ckpt_path}", flush=True)

    if cfg.general.run_wandb:
        wandb.log({
            "final/best_wm_loss": best_wm_loss,
            "final/total_iters": wm_pretrain_iters,
            "final/total_time_sec": time.time() - start_time,
        })
        wandb.finish()

    print(f"\nWM Pretraining complete. Best loss: {best_wm_loss:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")


def _save_checkpoint(agent, path, iter_num, obs_dim, act_dim, cfg, task_set, is_best=False):
    """Save WM checkpoint with metadata."""
    torch.save(
        {
            "world_model": agent.wm.state_dict(),
            "world_model_opt": agent.wm_optimizer.state_dict(),
            "iter": iter_num,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "horizon": int(cfg.horizon),
            "task_set": list(task_set),
            "is_best": is_best,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


if __name__ == "__main__":
    pretrain()


