"""SIGReg-style latent regularization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LatentGaussianStats:
    """Compact diagnostics for Gaussian-like latent batches."""

    mean_norm: torch.Tensor
    variance_mean: torch.Tensor
    variance_min: torch.Tensor
    variance_max: torch.Tensor
    covariance_offdiag_abs_mean: torch.Tensor


class SIGRegLoss(nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    The tensor contract follows LeWM: embeddings are time-major with shape
    ``(T, B, D)``. The empirical characteristic function is computed across the
    batch dimension for each time index and random projection.
    """

    def __init__(
        self,
        num_knots: int = 17,
        num_projections: int = 1024,
        t_max: float = 3.0,
        bandwidth: float = 1.0,
    ) -> None:
        super().__init__()
        if num_knots < 2:
            raise ValueError("num_knots must be at least 2")
        if num_projections < 1:
            raise ValueError("num_projections must be at least 1")
        if t_max <= 0:
            raise ValueError("t_max must be positive")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self.num_projections = int(num_projections)
        t = torch.linspace(0.0, float(t_max), int(num_knots), dtype=torch.float32)
        dt = float(t_max) / float(num_knots - 1)
        weights = torch.full((int(num_knots),), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / (2.0 * float(bandwidth) ** 2))
        self.register_buffer("t", t)
        self.register_buffer("phi", torch.exp(-0.5 * t.square()))
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("SIGRegLoss expects embeddings with shape (T, B, D)")
        if embeddings.shape[1] < 2:
            return embeddings.new_zeros(())
        directions = torch.randn(
            embeddings.shape[-1],
            self.num_projections,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-12)
        projected = embeddings @ directions
        t = self.t.to(device=embeddings.device, dtype=embeddings.dtype)
        phi = self.phi.to(device=embeddings.device, dtype=embeddings.dtype)
        weights = self.weights.to(device=embeddings.device, dtype=embeddings.dtype)
        x_t = projected.unsqueeze(-1) * t
        err = (x_t.cos().mean(dim=1) - phi).square() + x_t.sin().mean(dim=1).square()
        statistic = (err @ weights) * embeddings.shape[1]
        return statistic.mean()


def sigreg_loss(
    embeddings: torch.Tensor,
    num_knots: int = 17,
    num_projections: int = 1024,
    t_max: float = 3.0,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    regularizer = SIGRegLoss(
        num_knots=num_knots,
        num_projections=num_projections,
        t_max=t_max,
        bandwidth=bandwidth,
    ).to(device=embeddings.device, dtype=embeddings.dtype)
    return regularizer(embeddings)


def add_sigreg_loss(
    base_loss: torch.Tensor,
    embeddings: torch.Tensor,
    weight: float,
    regularizer: SIGRegLoss | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if float(weight) == 0.0:
        return base_loss, embeddings.new_zeros(())
    if regularizer is None:
        regularizer = SIGRegLoss().to(device=embeddings.device, dtype=embeddings.dtype)
    reg_loss = regularizer(embeddings)
    return base_loss + float(weight) * reg_loss, reg_loss


def latent_gaussian_stats(embeddings: torch.Tensor) -> LatentGaussianStats:
    if embeddings.ndim < 2:
        raise ValueError("embeddings must have a latent dimension")
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    mean = flat.mean(dim=0)
    centered = flat - mean
    if flat.shape[0] < 2:
        var = centered.square().mean(dim=0)
        offdiag = flat.new_zeros(())
    else:
        var = centered.var(dim=0, unbiased=False)
        cov = centered.T @ centered / float(flat.shape[0])
        offdiag_mask = ~torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
        offdiag = cov[offdiag_mask].abs().mean() if offdiag_mask.any() else cov.new_zeros(())
    return LatentGaussianStats(
        mean_norm=mean.norm(p=2),
        variance_mean=var.mean(),
        variance_min=var.min(),
        variance_max=var.max(),
        covariance_offdiag_abs_mean=offdiag,
    )
