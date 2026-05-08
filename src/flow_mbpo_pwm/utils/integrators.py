"""
ODE integrators for flow-matching dynamics.

Implements Euler and Heun's method for integrating the velocity field
from t=0 to t=1 to predict next states.
"""

import math

import torch


def _sample_tau(batch_size, device, dtype, tau_sampling):
    if tau_sampling == "uniform":
        return torch.rand((batch_size, 1), device=device, dtype=dtype)
    if tau_sampling == "midpoint":
        return torch.full((batch_size, 1), 0.5, device=device, dtype=dtype)
    if tau_sampling == "u_shaped":
        beta = torch.distributions.Beta(
            torch.tensor(0.5, device=device, dtype=dtype),
            torch.tensor(0.5, device=device, dtype=dtype),
        )
        return beta.sample((batch_size, 1)).to(device=device, dtype=dtype)
    raise ValueError(f"Unknown tau_sampling: {tau_sampling}")


def _interpolation_alpha(tau, path_family):
    if path_family == "linear":
        return tau
    if path_family == "smoothstep":
        return tau * tau * (3.0 - 2.0 * tau)
    if path_family == "quadratic":
        return tau * tau
    if path_family == "cosine":
        return 0.5 - 0.5 * torch.cos(torch.pi * tau)
    if path_family == "endpoint_sigmoid":
        k = 6.0
        lo = torch.sigmoid(torch.tensor(-0.5 * k, device=tau.device, dtype=tau.dtype))
        hi = torch.sigmoid(torch.tensor(0.5 * k, device=tau.device, dtype=tau.dtype))
        sig = torch.sigmoid(k * (tau - 0.5))
        return (sig - lo) / (hi - lo)
    raise ValueError(f"Unknown path_family: {path_family}")


def _normalize(x, eps):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _target_velocity(z_start, z_tau, z_target, target_mode, target_scale, target_mix, eps):
    delta = z_target - z_start
    if target_mode == "raw":
        target = delta
    elif target_mode == "endpoint_residual":
        target = z_target - z_tau
    elif target_mode == "hybrid_endpoint":
        endpoint = z_target - z_tau
        target = target_mix * delta + (1.0 - target_mix) * endpoint
    elif target_mode == "normalized":
        norm = delta.norm(dim=-1, keepdim=True).clamp_min(eps)
        target = delta / norm
    elif target_mode == "direction_magnitude":
        direction = _normalize(delta, eps)
        magnitude = delta.norm(dim=-1, keepdim=True)
        target = direction * magnitude
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
    return target * target_scale


def _pair_weight(z_start, z_target, mode, scale, eps):
    if mode == "none":
        return None
    if mode == "displacement_gibbs":
        dist2 = (z_target - z_start).pow(2).mean(dim=-1, keepdim=True)
        denom = max(float(scale), eps)
        return torch.exp(-dist2 / denom)
    raise ValueError(f"Unknown pair_weight_mode: {mode}")


def _couple_targets(z_start, z_target, mode, scale, eps):
    if mode != "ot_barycenter":
        return z_target
    dist2 = torch.cdist(z_start, z_target, p=2.0).pow(2)
    temperature = max(float(scale), eps)
    weights = torch.softmax(-dist2 / temperature, dim=-1)
    return weights @ z_target


def _maybe_whiten(z_start, z_target, eps):
    cat = torch.cat([z_start, z_target], dim=0)
    mean = cat.mean(dim=0, keepdim=True)
    std = cat.std(dim=0, keepdim=True).clamp_min(eps)
    return (z_start - mean) / std, (z_target - mean) / std


def euler_step(velocity_fn, z, a, task, substeps=1, reverse=False):
    """
    Euler integration: simple one-step forward method.
    
    For K substeps with dt = 1/K:
        z_{k+1} = z_k + dt * v_θ(z_k, a, t_k)
    
    Args:
        velocity_fn: Function that computes v_θ(z, a, τ, task)
        z: Initial latent state [batch_size, latent_dim]
        a: Action [batch_size, action_dim]
        task: Task ID or None
        substeps: Number of integration substeps (K)
    
    Returns:
        Final latent state after integration
    """
    dt = 1.0 / substeps
    batch_size = z.shape[0]
    
    for k in range(substeps):
        # Reverse integration runs from t=1 back to t=0.
        t_k = (1.0 - k * dt) if reverse else (k * dt)
        tau = torch.full((batch_size, 1), t_k, device=z.device, dtype=z.dtype)
        
        # Compute velocity
        v = velocity_fn(z, a, tau, task)
        
        # Euler step
        z = z - dt * v if reverse else z + dt * v
    
    return z


def heun_step(velocity_fn, z, a, task, substeps=1, reverse=False):
    """
    Heun's method (explicit trapezoidal / RK2): improved stability over Euler.
    
    For K substeps with dt = 1/K:
        k1 = v_θ(z_k, a, t_k)
        z' = z_k + dt * k1
        k2 = v_θ(z', a, t_k + dt)
        z_{k+1} = z_k + (dt/2) * (k1 + k2)
    
    Args:
        velocity_fn: Function that computes v_θ(z, a, τ, task)
        z: Initial latent state [batch_size, latent_dim]
        a: Action [batch_size, action_dim]
        task: Task ID or None
        substeps: Number of integration substeps (K)
    
    Returns:
        Final latent state after integration
    """
    dt = 1.0 / substeps
    batch_size = z.shape[0]
    
    for k in range(substeps):
        t_k = (1.0 - k * dt) if reverse else (k * dt)
        tau_k = torch.full((batch_size, 1), t_k, device=z.device, dtype=z.dtype)
        
        # First velocity evaluation
        k1 = velocity_fn(z, a, tau_k, task)
        
        # Predictor step
        z_pred = z - dt * k1 if reverse else z + dt * k1
        
        # Second velocity evaluation at predicted point
        t_next = (t_k - dt) if reverse else (t_k + dt)
        tau_k_plus_dt = torch.full((batch_size, 1), t_next, device=z.device, dtype=z.dtype)
        k2 = velocity_fn(z_pred, a, tau_k_plus_dt, task)
        
        # Corrector step (trapezoidal rule)
        step = (dt / 2.0) * (k1 + k2)
        z = z - step if reverse else z + step
    
    return z


def compute_flow_matching_loss(
    velocity_fn,
    z_start,
    z_target,
    a,
    task,
    tau_sampling="uniform",
    path_family="linear",
    target_mode="raw",
    target_scale=1.0,
    target_mix=0.5,
    target_eps=1e-6,
    pair_weight_mode="none",
    pair_weight_scale=1.0,
    stochastic_interp_sigma=0.0,
    whiten_latents=False,
    gamma_t=1.0,
):
    """
    Computes the flow-matching loss for a single transition.
    
    Per the plan (Section 3):
        - Sample τ ~ U[0,1] (or use midpoint 0.5)
        - Interpolate: z_τ = (1-τ) z_start + τ z_target
        - Target velocity: v* = z_target - z_start
        - Loss: ||v_θ(z_τ, a, τ) - v*||²
    
    Args:
        velocity_fn: Function that computes v_θ(z, a, τ, task)
        z_start: Starting latent state [batch_size, latent_dim]
        z_target: Target latent state [batch_size, latent_dim]
        a: Action [batch_size, action_dim]
        task: Task ID or None
        tau_sampling: 'uniform', 'midpoint', or 'u_shaped'
        gamma_t: Discount factor for this timestep
    
    Returns:
        Flow-matching loss (scalar)
    """
    batch_size = z_start.shape[0]
    device = z_start.device
    
    if pair_weight_mode == "ot_barycenter":
        z_target = _couple_targets(
            z_start,
            z_target,
            mode=pair_weight_mode,
            scale=pair_weight_scale,
            eps=target_eps,
        )
        pair_weight_mode = "none"

    if whiten_latents:
        z_start, z_target = _maybe_whiten(z_start, z_target, target_eps)

    tau = _sample_tau(batch_size, device, z_start.dtype, tau_sampling)
    alpha = _interpolation_alpha(tau, path_family)

    # Interpolate along the configured path family.
    z_tau = (1.0 - alpha) * z_start + alpha * z_target
    if stochastic_interp_sigma > 0:
        sigma = float(stochastic_interp_sigma)
        bridge_scale = torch.sqrt(torch.clamp(alpha * (1.0 - alpha), min=0.0))
        z_tau = z_tau + sigma * bridge_scale * torch.randn_like(z_tau)

    # Target velocity can optionally be rescaled / normalized.
    v_target = _target_velocity(
        z_start,
        z_tau,
        z_target,
        target_mode=target_mode,
        target_scale=target_scale,
        target_mix=target_mix,
        eps=target_eps,
    )

    # Predicted velocity
    v_pred = velocity_fn(z_tau, a, tau, task)

    if v_pred.ndim == 3:
        target_for_heads = v_target.unsqueeze(0)
        pred_delta = v_pred - target_for_heads
        err = pred_delta.pow(2)
    else:
        pred_delta = v_pred - v_target
        err = pred_delta.pow(2)

    if target_mode == "direction_magnitude":
        if v_pred.ndim == 3:
            v_pred_mean = v_pred.mean(dim=0)
        else:
            v_pred_mean = v_pred
        pred_mag = v_pred_mean.norm(dim=-1, keepdim=True)
        target_mag = v_target.norm(dim=-1, keepdim=True)
        pred_dir = _normalize(v_pred_mean, target_eps)
        target_dir = _normalize(v_target, target_eps)
        direction_loss = 1.0 - (pred_dir * target_dir).sum(dim=-1, keepdim=True)
        magnitude_loss = (pred_mag - target_mag).pow(2)
        err = target_mix * direction_loss + (1.0 - target_mix) * magnitude_loss

    pair_weight = _pair_weight(
        z_start,
        z_target,
        mode=pair_weight_mode,
        scale=pair_weight_scale,
        eps=target_eps,
    )
    if pair_weight is not None:
        err = err.mean(dim=-1, keepdim=True) * pair_weight
        loss = err.mean() * gamma_t
    else:
        loss = err.mean() * gamma_t

    return loss
