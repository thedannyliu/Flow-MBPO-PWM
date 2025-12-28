"""
Flow-based Actor implementations for Flow Policy experiments.

This module provides ODE-based policy architectures that generate actions
by integrating a learned velocity field from noise to action space.
Based on Phase 2 of the master plan (Flow Policy with ODE-based sampling).
"""

from typing import List, Type
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from pwm.models import model_utils


class ActorFlowODE(nn.Module):
    """
    Flow-based Actor using ODE integration for action generation.
    
    Instead of directly outputting actions, this actor:
    1. Starts from a noise sample z ~ N(0, I)
    2. Integrates a learned velocity field v_θ(z, obs, τ) from τ=0 to τ=1
    3. The final state at τ=1 is the action
    
    This provides a more expressive action distribution than Gaussian
    while remaining differentiable for policy gradient training.
    
    Args:
        obs_dim: Observation/latent dimension
        action_dim: Action dimension
        units: Hidden layer sizes for velocity network
        activation_class: Activation function class
        init_gain: Weight initialization gain
        flow_substeps: Number of ODE integration substeps (K)
        flow_integrator: Integration method ('euler' or 'heun')
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        units: List[int],
        activation_class: Type = nn.Mish,
        init_gain: float = 1.0,
        flow_substeps: int = 2,
        flow_integrator: str = 'heun',
    ):
        super(ActorFlowODE, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.flow_substeps = flow_substeps
        self.flow_integrator = flow_integrator
        
        # Input to velocity net: obs + action_noise + time
        input_dim = obs_dim + action_dim + 1
        
        if isinstance(activation_class, str):
            activation_class = eval(activation_class)
        self.activation_class = activation_class
        
        # Build velocity network
        layer_dims = [input_dim] + units + [action_dim]
        
        modules = []
        for i in range(len(layer_dims) - 1):
            modules.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                modules.append(self.activation_class())
                modules.append(nn.LayerNorm(layer_dims[i + 1]))
        
        self.velocity_net = nn.Sequential(*modules)
        
        # Initialize weights
        for param in self.parameters():
            param.data *= init_gain
        
        # Learnable initial noise std (like logstd in Gaussian policy)
        self.init_logstd = nn.Parameter(
            torch.ones(action_dim, dtype=torch.float32) * -1.0
        )
        self.min_logstd = -10.0
    
    def get_logstd(self):
        """Return the logstd for compatibility with MLP actor interface."""
        return self.init_logstd.clamp(min=self.min_logstd)
    
    def clamp_std(self):
        """Clamp logstd to minimum value."""
        self.init_logstd.data = torch.clamp(self.init_logstd.data, self.min_logstd)
    
    def _velocity(self, z, obs, tau):
        """
        Compute velocity field v_θ(z, obs, τ).
        
        Args:
            z: Current action-space state [batch_size, action_dim]
            obs: Observation [batch_size, obs_dim]
            tau: Time parameter [batch_size, 1]
        
        Returns:
            Velocity [batch_size, action_dim]
        """
        # Concatenate inputs
        x = torch.cat([obs, z, tau], dim=-1)
        return self.velocity_net(x)
    
    def _integrate(self, z0, obs, substeps=None):
        """
        Integrate velocity field from τ=0 to τ=1.
        
        Args:
            z0: Initial noise [batch_size, action_dim]
            obs: Observation [batch_size, obs_dim]
            substeps: Override substeps count (optional)
        
        Returns:
            Final action [batch_size, action_dim]
        """
        K = substeps if substeps is not None else self.flow_substeps
        dt = 1.0 / K
        batch_size = z0.shape[0]
        
        z = z0
        
        for k in range(K):
            t_k = k * dt
            tau = torch.full((batch_size, 1), t_k, device=z.device, dtype=z.dtype)
            
            if self.flow_integrator == 'euler':
                # Euler step
                v = self._velocity(z, obs, tau)
                z = z + dt * v
            
            elif self.flow_integrator == 'heun':
                # Heun's method (RK2)
                k1 = self._velocity(z, obs, tau)
                z_pred = z + dt * k1
                
                tau_next = torch.full((batch_size, 1), t_k + dt, device=z.device, dtype=z.dtype)
                k2 = self._velocity(z_pred, obs, tau_next)
                
                z = z + (dt / 2.0) * (k1 + k2)
            else:
                raise ValueError(f"Unknown integrator: {self.flow_integrator}")
        
        return z
    
    def forward(self, obs, deterministic=False):
        """
        Generate actions by sampling noise and integrating velocity field.
        
        Args:
            obs: Observation [batch_size, obs_dim]
            deterministic: If True, use zero noise (mode)
        
        Returns:
            Actions [batch_size, action_dim]
        """
        self.clamp_std()
        batch_size = obs.shape[0]
        device = obs.device
        
        if deterministic:
            # Start from zero (mode of standard normal)
            z0 = torch.zeros(batch_size, self.action_dim, device=device)
        else:
            # Sample from learned initial distribution
            std = self.init_logstd.exp()
            z0 = torch.randn(batch_size, self.action_dim, device=device) * std
        
        # Integrate to get action
        action = self._integrate(z0, obs)
        
        return action
    
    def action_log_probs(self, obs):
        """
        Compute actions and their log probabilities.
        
        Note: For ODE-based policies, exact log prob computation requires
        trace estimation which is expensive. Here we use an approximation
        based on the initial noise distribution.
        
        Args:
            obs: Observation [batch_size, obs_dim]
        
        Returns:
            Tuple of (actions, approx_log_probs)
        """
        self.clamp_std()
        batch_size = obs.shape[0]
        device = obs.device
        
        # Sample initial noise
        std = self.init_logstd.exp()
        z0 = torch.randn(batch_size, self.action_dim, device=device) * std
        
        # Compute approx log prob of initial noise
        dist = Normal(torch.zeros_like(z0), std)
        log_prob = dist.log_prob(z0)
        
        # Integrate to get action
        action = self._integrate(z0, obs)
        
        return action, log_prob
    
    def forward_with_dist(self, obs, deterministic=False):
        """
        Forward with distribution info for compatibility.
        
        Args:
            obs: Observation
            deterministic: Use deterministic mode
        
        Returns:
            Tuple of (action, mean_action, std)
        """
        self.clamp_std()
        batch_size = obs.shape[0]
        device = obs.device
        std = self.init_logstd.exp()
        
        if deterministic:
            z0 = torch.zeros(batch_size, self.action_dim, device=device)
        else:
            z0 = torch.randn(batch_size, self.action_dim, device=device) * std
        
        action = self._integrate(z0, obs)
        
        # Mean action (from zero initial)
        mean_action = self._integrate(
            torch.zeros(batch_size, self.action_dim, device=device), 
            obs
        )
        
        return action, mean_action, std
    
    def log_probs(self, obs, actions):
        """
        Compute log probabilities of given actions.
        
        Note: This is an approximation as exact log prob for ODE policies
        requires solving an inverse ODE and computing Jacobian trace.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            actions: Actions [batch_size, action_dim]
        
        Returns:
            Approximate log probabilities [batch_size, action_dim]
        """
        # This is a rough approximation - we estimate based on how far 
        # the action is from the deterministic mode
        self.clamp_std()
        std = self.init_logstd.exp()
        
        # Get mean action (deterministic)
        mean_action = self.forward(obs, deterministic=True)
        
        # Approximate as Gaussian around mean
        dist = Normal(mean_action, std)
        return dist.log_prob(actions)


class ActorFlowODEStochastic(ActorFlowODE):
    """
    Stochastic Flow Actor with additional exploration noise.
    
    Adds Gaussian noise to the final action for additional exploration,
    similar to how SAC adds noise on top of its policy output.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        units: List[int],
        activation_class: Type = nn.Mish,
        init_gain: float = 1.0,
        flow_substeps: int = 2,
        flow_integrator: str = 'heun',
        exploration_noise: float = 0.1,
    ):
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            units=units,
            activation_class=activation_class,
            init_gain=init_gain,
            flow_substeps=flow_substeps,
            flow_integrator=flow_integrator,
        )
        self.exploration_noise = exploration_noise
    
    def forward(self, obs, deterministic=False):
        """Forward with optional exploration noise."""
        action = super().forward(obs, deterministic=True)  # Always start deterministic
        
        if not deterministic:
            # Add exploration noise
            noise = torch.randn_like(action) * self.exploration_noise
            action = action + noise
        
        return action
