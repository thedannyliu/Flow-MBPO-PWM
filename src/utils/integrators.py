"""
ODE integrators for flow-matching dynamics.

Implements Euler and Heun's method for integrating the velocity field
from t=0 to t=1 to predict next states.
"""

import torch


def euler_step(velocity_fn, z, a, task, substeps=1):
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
        # Current time
        t_k = k * dt
        tau = torch.full((batch_size, 1), t_k, device=z.device, dtype=z.dtype)
        
        # Compute velocity
        v = velocity_fn(z, a, tau, task)
        
        # Euler step
        z = z + dt * v
    
    return z


def heun_step(velocity_fn, z, a, task, substeps=1):
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
        # Current time
        t_k = k * dt
        tau_k = torch.full((batch_size, 1), t_k, device=z.device, dtype=z.dtype)
        
        # First velocity evaluation
        k1 = velocity_fn(z, a, tau_k, task)
        
        # Predictor step
        z_pred = z + dt * k1
        
        # Second velocity evaluation at predicted point
        tau_k_plus_dt = torch.full((batch_size, 1), t_k + dt, device=z.device, dtype=z.dtype)
        k2 = velocity_fn(z_pred, a, tau_k_plus_dt, task)
        
        # Corrector step (trapezoidal rule)
        z = z + (dt / 2.0) * (k1 + k2)
    
    return z


def compute_flow_matching_loss(velocity_fn, z_start, z_target, a, task, 
                                 tau_sampling='uniform', gamma_t=1.0):
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
        tau_sampling: 'uniform' or 'midpoint'
        gamma_t: Discount factor for this timestep
    
    Returns:
        Flow-matching loss (scalar)
    """
    batch_size = z_start.shape[0]
    device = z_start.device
    
    # Sample time parameter τ
    if tau_sampling == 'uniform':
        tau = torch.rand((batch_size, 1), device=device, dtype=z_start.dtype)
    elif tau_sampling == 'midpoint':
        tau = torch.full((batch_size, 1), 0.5, device=device, dtype=z_start.dtype)
    else:
        raise ValueError(f"Unknown tau_sampling: {tau_sampling}")
    
    # Interpolate: z_τ = (1-τ) z_start + τ z_target
    z_tau = (1.0 - tau) * z_start + tau * z_target
    
    # Target velocity (rectified flow)
    v_target = z_target - z_start
    
    # Predicted velocity
    v_pred = velocity_fn(z_tau, a, tau, task)
    
    # MSE loss
    loss = ((v_pred - v_target) ** 2).mean() * gamma_t
    
    return loss
