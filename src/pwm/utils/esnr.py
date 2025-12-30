"""
Expected Signal-to-Noise Ratio (ESNR) computation for actor gradients.

Per plan Section 6, ESNR measures the gradient signal quality for actor updates.
This is a key metric for comparing baseline vs flow-matching dynamics.
"""

import torch
import torch.nn as nn
from typing import List, Optional


def compute_esnr(gradients: List[torch.Tensor], eps: float = 1e-10) -> torch.Tensor:
    """
    Compute Expected Signal-to-Noise Ratio (ESNR) from a collection of gradient samples.
    
    Per plan Section 6:
        μ = (1/M) Σ_i g_i
        E[||g||²] = (1/M) Σ_i ||g_i||²
        Var = E[||g||²] - ||μ||²
        ESNR = ||μ||² / max(Var, ε)
    
    Args:
        gradients: List of gradient tensors from M micro-batches [each (D,)]
        eps: Small constant to avoid division by zero
    
    Returns:
        ESNR scalar tensor
    """
    if len(gradients) == 0:
        return torch.tensor(0.0)
    
    M = len(gradients)
    
    # Stack gradients: [M, D]
    g_stack = torch.stack(gradients)
    
    # Mean gradient: μ = (1/M) Σ_i g_i
    mu = g_stack.mean(dim=0)
    
    # Expected squared norm: E[||g||²] = (1/M) Σ_i ||g_i||²
    g_norms_sq = (g_stack ** 2).sum(dim=1)  # [M]
    expected_norm_sq = g_norms_sq.mean()
    
    # Variance: Var = E[||g||²] - ||μ||²
    mu_norm_sq = (mu ** 2).sum()
    variance = expected_norm_sq - mu_norm_sq
    
    # ESNR = ||μ||² / max(Var, ε)
    esnr = mu_norm_sq / torch.maximum(variance, torch.tensor(eps, device=variance.device))
    
    return esnr


def compute_esnr_db(gradients: List[torch.Tensor], eps: float = 1e-10) -> torch.Tensor:
    """
    Compute ESNR in decibels (dB).
    
    ESNR_dB = 10 * log10(ESNR)
    
    Args:
        gradients: List of gradient tensors
        eps: Small constant
    
    Returns:
        ESNR in dB
    """
    esnr = compute_esnr(gradients, eps)
    esnr_db = 10.0 * torch.log10(esnr + eps)
    return esnr_db


class ESNRTracker:
    """
    Online tracker for ESNR estimation during training.
    
    This class maintains a buffer of recent gradient samples and computes
    ESNR periodically without excessive memory overhead.
    """
    
    def __init__(self, buffer_size: int = 32, ema_alpha: float = 0.1):
        """
        Args:
            buffer_size: Number of gradient samples to maintain (M)
            ema_alpha: Exponential moving average coefficient for smoothing
        """
        self.buffer_size = buffer_size
        self.ema_alpha = ema_alpha
        self.gradient_buffer: List[torch.Tensor] = []
        self.esnr_ema: Optional[torch.Tensor] = None
        self.esnr_db_ema: Optional[torch.Tensor] = None
    
    def update(self, gradient: torch.Tensor):
        """
        Add a new gradient sample to the buffer.
        
        Args:
            gradient: Flattened gradient tensor [D]
        """
        # Detach and clone to avoid holding computation graph
        g = gradient.detach().clone()
        
        self.gradient_buffer.append(g)
        
        # Maintain buffer size
        if len(self.gradient_buffer) > self.buffer_size:
            self.gradient_buffer.pop(0)
    
    def compute(self) -> dict:
        """
        Compute current ESNR metrics.
        
        Returns:
            Dictionary with 'esnr', 'esnr_db', 'esnr_ema', 'esnr_db_ema'
        """
        if len(self.gradient_buffer) < 2:
            return {
                'esnr': 0.0,
                'esnr_db': -float('inf'),
                'esnr_ema': 0.0 if self.esnr_ema is None else self.esnr_ema.item(),
                'esnr_db_ema': -float('inf') if self.esnr_db_ema is None else self.esnr_db_ema.item(),
            }
        
        esnr = compute_esnr(self.gradient_buffer)
        esnr_db = compute_esnr_db(self.gradient_buffer)
        
        # Update EMA
        if self.esnr_ema is None:
            self.esnr_ema = esnr
            self.esnr_db_ema = esnr_db
        else:
            self.esnr_ema = self.ema_alpha * esnr + (1 - self.ema_alpha) * self.esnr_ema
            self.esnr_db_ema = self.ema_alpha * esnr_db + (1 - self.ema_alpha) * self.esnr_db_ema
        
        return {
            'esnr': esnr.item(),
            'esnr_db': esnr_db.item(),
            'esnr_ema': self.esnr_ema.item(),
            'esnr_db_ema': self.esnr_db_ema.item(),
        }
    
    def reset(self):
        """Clear the gradient buffer."""
        self.gradient_buffer.clear()


def extract_flat_grad(model: nn.Module) -> torch.Tensor:
    """
    Extract and flatten all gradients from a model.
    
    Args:
        model: PyTorch model with computed gradients
    
    Returns:
        Flattened gradient vector
    """
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    
    if len(grads) == 0:
        return torch.tensor([])
    
    return torch.cat(grads)


# Example usage in training loop:
"""
# Initialize tracker
esnr_tracker = ESNRTracker(buffer_size=32)

# In training loop, after actor backward() but before optimizer.step():
for micro_batch in batches:
    actor_loss = compute_actor_loss(micro_batch)
    actor_loss.backward()
    
    # Extract and track gradient
    flat_grad = extract_flat_grad(actor)
    esnr_tracker.update(flat_grad)
    
    optimizer.step()
    optimizer.zero_grad()

# Periodically log ESNR
if iter % log_interval == 0:
    metrics = esnr_tracker.compute()
    wandb.log({
        'actor/esnr': metrics['esnr'],
        'actor/esnr_db': metrics['esnr_db'],
    })
"""
