"""
Enhanced training monitoring with progress bars, ETA, and detailed logging.
"""

import time
from datetime import timedelta
from typing import Optional, Dict, Any
import torch
import numpy as np
from tqdm import tqdm


class TrainingMonitor:
    """
    Monitors training progress with ETA estimation and detailed metrics tracking.
    """
    
    def __init__(
        self, 
        total_epochs: int,
        log_interval: int = 50,
        enable_tqdm: bool = True,
    ):
        """
        Args:
            total_epochs: Total number of training epochs
            log_interval: How often to log detailed metrics
            enable_tqdm: Whether to show progress bar
        """
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        self.enable_tqdm = enable_tqdm
        
        # Timing
        self.start_time = None
        self.epoch_times = []
        self.eta_smoothing = 0.9  # EMA coefficient for ETA
        self.avg_epoch_time = None
        
        # Progress bar
        self.pbar: Optional[tqdm] = None
        
        # Metrics history
        self.metrics_history = {
            'epoch': [],
            'wall_time': [],
            'eta': [],
        }
        
    def start(self):
        """Start training monitoring."""
        self.start_time = time.time()
        if self.enable_tqdm:
            self.pbar = tqdm(
                total=self.total_epochs,
                desc="Training",
                unit="epoch",
                dynamic_ncols=True,
                position=0,
                leave=True,
            )
    
    def update(self, epoch: int, metrics: Dict[str, Any]):
        """
        Update progress and compute ETA.
        
        Args:
            epoch: Current epoch number (0-indexed)
            metrics: Dictionary of metrics to log
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Compute epoch time
        if len(self.epoch_times) > 0:
            epoch_time = elapsed - sum(self.epoch_times)
        else:
            epoch_time = elapsed
        
        self.epoch_times.append(epoch_time)
        
        # Update average epoch time with EMA
        if self.avg_epoch_time is None:
            self.avg_epoch_time = epoch_time
        else:
            self.avg_epoch_time = (
                self.eta_smoothing * self.avg_epoch_time + 
                (1 - self.eta_smoothing) * epoch_time
            )
        
        # Compute ETA
        remaining_epochs = self.total_epochs - (epoch + 1)
        eta_seconds = self.avg_epoch_time * remaining_epochs
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        # Store metrics
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['wall_time'].append(elapsed)
        self.metrics_history['eta'].append(eta_seconds)
        
        # Update progress bar
        if self.pbar is not None:
            postfix = {
                'R': f"{metrics.get('rewards', 0):.1f}",
                'L': f"{metrics.get('actor_loss', 0):.2f}",
                'ETA': eta_str,
            }
            self.pbar.set_postfix(postfix)
            self.pbar.update(1)
        
        # Return formatted info
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        return {
            'elapsed': elapsed_str,
            'eta': eta_str,
            'epoch_time': epoch_time,
            'avg_epoch_time': self.avg_epoch_time,
        }
    
    def close(self):
        """Close progress bar and compute final statistics."""
        if self.pbar is not None:
            self.pbar.close()
        
        total_time = time.time() - self.start_time
        
        return {
            'total_time': total_time,
            'total_time_str': str(timedelta(seconds=int(total_time))),
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'median_epoch_time': np.median(self.epoch_times) if self.epoch_times else 0,
        }


class WandBLogger:
    """
    Enhanced WandB logging with detailed metrics and visualizations.
    """
    
    def __init__(
        self,
        project: str,
        entity: str,
        name: str,
        config: Dict[str, Any],
        enabled: bool = True,
    ):
        """
        Args:
            project: WandB project name
            entity: WandB entity name
            name: Run name
            config: Configuration dictionary
            enabled: Whether WandB logging is enabled
        """
        self.enabled = enabled
        
        if self.enabled:
            import wandb
            self.wandb = wandb
            
            # Initialize run
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                # Track code and system info
                save_code=True,
            )
            
            # Define custom metrics
            self._define_custom_metrics()
        else:
            self.wandb = None
            self.run = None
    
    def _define_custom_metrics(self):
        """Define custom metric summaries."""
        if not self.enabled:
            return
        
        # Track best values
        self.wandb.define_metric("rewards", summary="max")
        self.wandb.define_metric("policy_loss", summary="min")
        self.wandb.define_metric("best_policy_loss", summary="min")
        
        # Track gradients
        self.wandb.define_metric("actor_grad_norm", summary="mean")
        self.wandb.define_metric("critic_grad_norm", summary="mean")
        self.wandb.define_metric("wm_grad_norm", summary="mean")
    
    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log metrics to WandB.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step
            commit: Whether to commit the metrics
        """
        if not self.enabled:
            return
        
        self.wandb.log(metrics, step=step, commit=commit)
    
    def log_histogram(
        self,
        name: str,
        data: torch.Tensor,
        step: Optional[int] = None,
    ):
        """Log histogram of tensor data."""
        if not self.enabled:
            return
        
        self.wandb.log(
            {name: self.wandb.Histogram(data.detach().cpu().numpy())},
            step=step,
        )
    
    def log_gradient_distributions(
        self,
        model: torch.nn.Module,
        prefix: str = "gradients",
        step: Optional[int] = None,
    ):
        """
        Log gradient distributions for a model.
        
        Args:
            model: PyTorch model
            prefix: Prefix for metric names
            step: Current step
        """
        if not self.enabled:
            return
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                clean_name = name.replace('.', '/')
                self.wandb.log(
                    {
                        f"{prefix}/{clean_name}": self.wandb.Histogram(
                            param.grad.detach().cpu().numpy()
                        )
                    },
                    step=step,
                    commit=False,
                )
        
        # Commit all histograms together
        self.wandb.log({}, step=step, commit=True)
    
    def watch_model(
        self,
        model: torch.nn.Module,
        log: str = "all",
        log_freq: int = 1000,
    ):
        """
        Watch model parameters and gradients.
        
        Args:
            model: Model to watch
            log: What to log ("gradients", "parameters", "all")
            log_freq: How often to log
        """
        if not self.enabled:
            return
        
        self.wandb.watch(model, log=log, log_freq=log_freq)
    
    def finish(self):
        """Finish WandB run."""
        if self.enabled and self.run is not None:
            self.run.finish()


def compute_gradient_stats(model: torch.nn.Module) -> Dict[str, float]:
    """
    Compute gradient statistics for a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary of gradient statistics
    """
    grad_norms = []
    grad_values = []
    
    for param in model.parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
            grad_values.extend(param.grad.flatten().cpu().numpy().tolist())
    
    if not grad_norms:
        return {}
    
    grad_values = np.array(grad_values)
    
    return {
        'grad_norm_mean': np.mean(grad_norms),
        'grad_norm_std': np.std(grad_norms),
        'grad_norm_max': np.max(grad_norms),
        'grad_value_mean': np.mean(grad_values),
        'grad_value_std': np.std(grad_values),
        'grad_value_min': np.min(grad_values),
        'grad_value_max': np.max(grad_values),
    }
