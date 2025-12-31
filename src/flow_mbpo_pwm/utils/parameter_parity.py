"""
Utility for calculating and comparing world model parameter counts.

This module provides functions to ensure parameter parity between
baseline and flow-matching world models as specified in the plan.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


def count_parameters(model: nn.Module, detailed: bool = False) -> Dict[str, int]:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        detailed: If True, return per-component breakdown
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if not detailed:
        return {"total": total}
    
    counts = {"total": total}
    
    # Break down by component if available
    if hasattr(model, '_encoder'):
        counts['encoder'] = sum(p.numel() for p in model._encoder.parameters() if p.requires_grad)
    
    if hasattr(model, '_dynamics'):
        counts['dynamics'] = sum(p.numel() for p in model._dynamics.parameters() if p.requires_grad)
    
    if hasattr(model, '_velocity'):
        counts['velocity'] = sum(p.numel() for p in model._velocity.parameters() if p.requires_grad)
    
    if hasattr(model, '_reward'):
        counts['reward'] = sum(p.numel() for p in model._reward.parameters() if p.requires_grad)
    
    if hasattr(model, '_task_emb'):
        counts['task_emb'] = sum(p.numel() for p in model._task_emb.parameters() if p.requires_grad)
    
    return counts


def check_parameter_parity(baseline_count: int, flow_count: int, 
                           tolerance: float = 0.02) -> Tuple[bool, float, str]:
    """
    Check if two models satisfy parameter parity constraint.
    
    Per plan Section 5: |P_flow - P_base| / P_base <= 0.02
    
    Args:
        baseline_count: Parameter count of baseline model
        flow_count: Parameter count of flow model
        tolerance: Maximum allowed relative difference (default 2%)
    
    Returns:
        Tuple of (passes_check, relative_diff, message)
    """
    if baseline_count == 0:
        return False, float('inf'), "Baseline count is zero!"
    
    relative_diff = abs(flow_count - baseline_count) / baseline_count
    passes = relative_diff <= tolerance
    
    if passes:
        message = (f"✓ Parameter parity satisfied: "
                  f"baseline={baseline_count:,}, flow={flow_count:,}, "
                  f"diff={relative_diff*100:.2f}%")
    else:
        message = (f"✗ Parameter parity VIOLATED: "
                  f"baseline={baseline_count:,}, flow={flow_count:,}, "
                  f"diff={relative_diff*100:.2f}% (exceeds {tolerance*100}% threshold)")
    
    return passes, relative_diff, message


def print_parameter_summary(model_name: str, counts: Dict[str, int]):
    """Pretty print parameter counts."""
    print(f"\n{'='*60}")
    print(f"  {model_name} Parameter Summary")
    print(f"{'='*60}")
    
    # Print total first
    if 'total' in counts:
        print(f"  Total:        {counts['total']:>12,}")
    
    # Print components
    for key, value in counts.items():
        if key != 'total':
            print(f"    - {key:12s}: {value:>12,}")
    
    print(f"{'='*60}\n")


def compare_models(baseline_model: nn.Module, flow_model: nn.Module, 
                   tolerance: float = 0.02, verbose: bool = True):
    """
    Compare two world models and check parameter parity.
    
    Args:
        baseline_model: Baseline WorldModel
        flow_model: FlowWorldModel
        tolerance: Parity tolerance (default 2%)
        verbose: Print detailed breakdown
    
    Returns:
        Tuple of (passes_parity, relative_diff)
    """
    baseline_counts = count_parameters(baseline_model, detailed=verbose)
    flow_counts = count_parameters(flow_model, detailed=verbose)
    
    if verbose:
        print_parameter_summary("Baseline WorldModel", baseline_counts)
        print_parameter_summary("Flow WorldModel", flow_counts)
    
    passes, diff, message = check_parameter_parity(
        baseline_counts['total'], 
        flow_counts['total'], 
        tolerance
    )
    
    print(message)
    
    if not passes:
        print("\n⚠️  WARNING: Parameter parity constraint violated!")
        print("   Consider adjusting the flow model architecture.")
        print("   See Section 5 of flow-world-model-plan.md")
    
    return passes, diff


def suggest_width_adjustment(baseline_count: int, flow_count: int, 
                             current_width: int, num_layers: int) -> int:
    """
    Suggest width adjustment to achieve parameter parity.
    
    Simple heuristic: assumes difference comes from width change in uniform layers.
    
    Args:
        baseline_count: Baseline parameter count
        flow_count: Current flow parameter count
        current_width: Current hidden width
        num_layers: Number of layers
    
    Returns:
        Suggested new width
    """
    if flow_count == 0 or num_layers == 0:
        return current_width
    
    # Rough estimate: scale width proportionally
    target_ratio = baseline_count / flow_count
    suggested_width = int(current_width * (target_ratio ** 0.5))
    
    return suggested_width
