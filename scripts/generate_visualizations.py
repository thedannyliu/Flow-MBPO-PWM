#!/usr/bin/env python3
"""
Generate visualizations from training logs.
"""

import argparse
import sys
from pathlib import Path
import pickle

# Add PWM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flow_mbpo_pwm.utils.visualization import TrainingVisualizer


def main():
    parser = argparse.ArgumentParser(description="Generate training visualizations")
    parser.add_argument("--log-dir", type=str, required=True,
                        help="Training log directory")
    parser.add_argument("--smooth-window", type=int, default=50,
                        help="Smoothing window size")
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return 1
    
    # Check if visualizer data exists
    viz_data_file = log_dir / "visualizer_data.pkl"
    
    if not viz_data_file.exists():
        print(f"Error: Visualizer data not found: {viz_data_file}")
        print("Make sure training completed and visualizer data was saved")
        return 1
    
    # Load visualizer
    print(f"Loading visualizer data from {viz_data_file}")
    with open(viz_data_file, 'rb') as f:
        visualizer = pickle.load(f)
    
    # Generate all plots
    visualizer.generate_all_plots()
    
    print(f"\nVisualization complete! Check {log_dir / 'visualizations'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
