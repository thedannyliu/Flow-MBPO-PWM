#!/usr/bin/env python3
"""
Compare multiple training runs (e.g., different seeds or baseline vs flow).
"""

import argparse
import sys
from pathlib import Path
from typing import List
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style("whitegrid")

# Add PWM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_run_data(log_dir: Path):
    """Load visualizer data from a run."""
    viz_data_file = log_dir / "visualizer_data.pkl"
    
    if not viz_data_file.exists():
        print(f"Warning: Data not found for {log_dir}")
        return None
    
    with open(viz_data_file, 'rb') as f:
        visualizer = pickle.load(f)
    
    return visualizer.data


def plot_comparison(
    runs_data: List[dict],
    run_names: List[str],
    save_dir: Path,
    metric: str = "rewards",
    smooth_window: int = 50,
):
    """
    Plot comparison of multiple runs.
    
    Args:
        runs_data: List of data dictionaries from each run
        run_names: Names for each run
        save_dir: Directory to save plots
        metric: Metric to compare
        smooth_window: Smoothing window
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs_data)))
    
    for data, name, color in zip(runs_data, run_names, colors):
        steps = np.array(data['steps'])
        values = np.array(data.get(metric, []))
        
        # Remove NaNs
        valid = ~np.isnan(values)
        steps = steps[valid]
        values = values[valid]
        
        if len(values) == 0:
            continue
        
        # Plot raw data
        ax.plot(steps, values, alpha=0.3, color=color)
        
        # Plot smoothed
        if len(values) > smooth_window:
            smoothed = smooth_data(values, smooth_window)
            ax.plot(steps[:len(smoothed)], smoothed, 
                   label=name, color=color, linewidth=2)
        else:
            ax.plot(steps, values, label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / f'comparison_{metric}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {metric} comparison to {save_path}")


def plot_statistical_comparison(
    runs_data: List[dict],
    run_names: List[str],
    save_dir: Path,
):
    """
    Plot statistical comparison (mean ± std for each metric).
    
    Args:
        runs_data: List of data dictionaries
        run_names: Names for each run
        save_dir: Save directory
    """
    # Compute final statistics for each run
    metrics_to_compare = ['rewards', 'policy_loss', 'actor_loss', 'value_loss']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_compare):
        ax = axes[idx]
        
        means = []
        stds = []
        labels = []
        
        for data, name in zip(runs_data, run_names):
            values = np.array(data.get(metric, []))
            valid = ~np.isnan(values)
            
            if valid.sum() == 0:
                continue
            
            # Use last 20% of training
            last_20pct = int(valid.sum() * 0.8)
            final_values = values[valid][last_20pct:]
            
            if len(final_values) > 0:
                means.append(np.mean(final_values))
                stds.append(np.std(final_values))
                labels.append(name)
        
        if len(means) > 0:
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Final {metric.replace("_", " ").title()} (last 20%)')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = save_dir / 'statistical_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved statistical comparison to {save_path}")


def smooth_data(data, window):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def main():
    parser = argparse.ArgumentParser(description="Compare multiple training runs")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name")
    parser.add_argument("--algorithm", type=str, 
                        help="Algorithm name (for single algorithm comparison)")
    parser.add_argument("--seeds", type=int, nargs='+',
                        help="Seeds to compare")
    parser.add_argument("--run-dirs", type=str, nargs='+',
                        help="Explicit list of run directories")
    parser.add_argument("--run-names", type=str, nargs='+',
                        help="Names for each run (for legend)")
    parser.add_argument("--output-dir", type=str,
                        help="Output directory for comparison plots")
    parser.add_argument("--smooth-window", type=int, default=50,
                        help="Smoothing window size")
    
    args = parser.parse_args()
    
    # Determine run directories
    if args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
        run_names = args.run_names if args.run_names else [d.name for d in run_dirs]
    elif args.algorithm and args.seeds:
        # Construct paths from algorithm and seeds
        run_dirs = []
        run_names = []
        for seed in args.seeds:
            log_dir = Path(f"logs/{args.algorithm}_{args.task}_seed{seed}")
            if log_dir.exists():
                run_dirs.append(log_dir)
                run_names.append(f"{args.algorithm} (seed={seed})")
            else:
                print(f"Warning: Directory not found: {log_dir}")
    else:
        print("Error: Must specify either --run-dirs or (--algorithm and --seeds)")
        return 1
    
    if len(run_dirs) == 0:
        print("Error: No valid run directories found")
        return 1
    
    # Load data from all runs
    print(f"\nLoading data from {len(run_dirs)} runs...")
    runs_data = []
    valid_names = []
    
    for run_dir, name in zip(run_dirs, run_names):
        print(f"  Loading {run_dir}...")
        data = load_run_data(run_dir)
        if data is not None:
            runs_data.append(data)
            valid_names.append(name)
    
    if len(runs_data) == 0:
        print("Error: No valid run data found")
        return 1
    
    print(f"Successfully loaded {len(runs_data)} runs")
    
    # Create output directory
    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        save_dir = Path(f"logs/comparisons/{args.task}_{args.algorithm or 'comparison'}")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    print("="*60)
    
    metrics = ['rewards', 'policy_loss', 'actor_loss', 'value_loss', 
               'wm_loss', 'dynamics_loss']
    
    for metric in metrics:
        plot_comparison(runs_data, valid_names, save_dir, metric, args.smooth_window)
    
    # Statistical comparison
    plot_statistical_comparison(runs_data, valid_names, save_dir)
    
    print("="*60)
    print(f"\nComparison complete! Results saved to: {save_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
