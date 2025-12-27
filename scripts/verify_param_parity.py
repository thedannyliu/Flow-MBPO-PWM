#!/usr/bin/env python3
"""
Verify parameter parity between Baseline and Flow world models.

This script checks that the parameter counts are within ±2% as required
by the flow-world-model-plan.md (Section 5).
"""

import sys
import torch
from pathlib import Path

# Add PWM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pwm.models.world_model import WorldModel
from pwm.models.flow_world_model import FlowWorldModel


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_baseline_model(obs_dim=100, act_dim=20, latent_dim=768):
    """Create baseline world model (48M config)."""
    from pwm.models.mlp import SimNorm
    
    model = WorldModel(
        observation_dim=obs_dim,
        action_dim=act_dim,
        latent_dim=latent_dim,
        units=[1792, 1792],
        encoder_units=[1792, 1792, 1792],
        num_bins=101,
        vmin=-10.0,
        vmax=10.0,
        task_dim=96,
        multitask=True,
        action_dims=[act_dim],
        tasks=['task0'],
        encoder={
            'last_layer': 'normedlinear',
            'last_layer_kwargs': {'act': SimNorm(simnorm_dim=8)}
        },
        dynamics={
            'last_layer': 'normedlinear',
            'last_layer_kwargs': {'act': SimNorm(simnorm_dim=8)}
        },
        reward={
            'last_layer': 'linear',
            'last_layer_kwargs': {}
        }
    )
    return model


def create_flow_model(obs_dim=100, act_dim=20, latent_dim=768, units=None):
    """Create flow world model (48M config)."""
    from pwm.models.mlp import SimNorm
    
    if units is None:
        units = [1788, 1788]  # Adjusted for +1 time dimension
    
    model = FlowWorldModel(
        observation_dim=obs_dim,
        action_dim=act_dim,
        latent_dim=latent_dim,
        units=units,
        encoder_units=[1792, 1792, 1792],  # Same as baseline
        num_bins=101,
        vmin=-10.0,
        vmax=10.0,
        task_dim=96,
        multitask=True,
        action_dims=[act_dim],
        tasks=['task0'],
        encoder={
            'last_layer': 'normedlinear',
            'last_layer_kwargs': {'act': SimNorm(simnorm_dim=8)}
        },
        dynamics={
            'last_layer': 'normedlinear',
            'last_layer_kwargs': {'act': SimNorm(simnorm_dim=8)}
        },
        reward={
            'last_layer': 'linear',
            'last_layer_kwargs': {}
        }
    )
    return model


def verify_parity(obs_dim=100, act_dim=20, latent_dim=768, 
                  flow_units=None, threshold=2.0):
    """
    Verify parameter parity between baseline and flow models.
    
    Args:
        obs_dim: Observation dimension
        act_dim: Action dimension
        latent_dim: Latent dimension
        flow_units: Flow dynamics units (list of 2 ints)
        threshold: Maximum allowed percentage difference
    
    Returns:
        bool: True if parity satisfied, False otherwise
    """
    print("=" * 70)
    print("Parameter Parity Verification")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Latent dim: {latent_dim}")
    print()
    
    # Create models
    print("Creating baseline model...")
    baseline = create_baseline_model(obs_dim, act_dim, latent_dim)
    p_base = count_parameters(baseline)
    
    print("Creating flow model...")
    flow = create_flow_model(obs_dim, act_dim, latent_dim, flow_units)
    p_flow = count_parameters(flow)
    
    # Calculate difference
    diff = p_flow - p_base
    diff_pct = abs(diff) / p_base * 100
    
    # Report
    print()
    print("-" * 70)
    print(f"{'Model':<20} {'Parameters':>15} {'Difference':>15} {'%':>10}")
    print("-" * 70)
    print(f"{'Baseline':<20} {p_base:>15,} {'-':>15} {'-':>10}")
    print(f"{'Flow':<20} {p_flow:>15,} {diff:>+15,} {diff_pct:>9.2f}%")
    print("-" * 70)
    print()
    
    # Check threshold
    if diff_pct <= threshold:
        print(f"✓ PASS: Difference {diff_pct:.2f}% <= {threshold}%")
        print()
        return True
    else:
        print(f"✗ FAIL: Difference {diff_pct:.2f}% > {threshold}%")
        print()
        print("Suggested fix:")
        print(f"  Adjust 'units' in pwm_48M_flow.yaml")
        
        # Suggest new width
        # Approximate: param_diff ≈ 2 * num_layers * (width_diff * (latent + action + 1 + task))
        input_dim = latent_dim + act_dim + 1 + 96  # +1 for time
        output_dim = latent_dim
        num_layers = 2
        
        # Rough estimate
        width_adjustment = diff // (num_layers * (input_dim + output_dim))
        current_width = flow_units[0] if flow_units else 1788
        suggested_width = current_width - width_adjustment
        
        print(f"  Current units: {[current_width, current_width]}")
        print(f"  Try: units: [{suggested_width}, {suggested_width}]")
        print()
        return False


def main():
    """Main verification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify parameter parity")
    parser.add_argument("--obs-dim", type=int, default=100, help="Observation dim")
    parser.add_argument("--act-dim", type=int, default=20, help="Action dim")
    parser.add_argument("--latent-dim", type=int, default=768, help="Latent dim")
    parser.add_argument("--flow-units", type=int, nargs=2, default=[1788, 1788],
                        help="Flow dynamics units")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Max allowed difference (%)")
    
    args = parser.parse_args()
    
    success = verify_parity(
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        latent_dim=args.latent_dim,
        flow_units=args.flow_units,
        threshold=args.threshold
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
