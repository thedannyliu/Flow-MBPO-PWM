"""
Automatic visualization generation for training results.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import seaborn as sns
sns.set_style("whitegrid")


class TrainingVisualizer:
    """
    Automatically generate visualizations for training results.
    """
    
    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: Directory to save visualizations
        """
        self.log_dir = Path(log_dir)
        self.viz_dir = self.log_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Store data
        self.data = {
            'steps': [],
            'rewards': [],
            'policy_loss': [],
            'actor_loss': [],
            'value_loss': [],
            'wm_loss': [],
            'dynamics_loss': [],
            'reward_loss': [],
            'actor_grad_norm': [],
            'critic_grad_norm': [],
            'wm_grad_norm': [],
            'episode_lengths': [],
            'fps': [],
        }
    
    def add_data(self, step: int, metrics: Dict[str, float]):
        """
        Add training data point.
        
        Args:
            step: Training step
            metrics: Dictionary of metrics
        """
        self.data['steps'].append(step)
        
        for key in self.data.keys():
            if key == 'steps':
                continue
            self.data[key].append(metrics.get(key, np.nan))
    
    def plot_learning_curves(self, smooth_window: int = 50):
        """
        Plot learning curves (rewards, losses).
        
        Args:
            smooth_window: Window size for smoothing
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        steps = np.array(self.data['steps'])
        
        # Plot 1: Rewards
        ax = axes[0, 0]
        rewards = np.array(self.data['rewards'])
        ax.plot(steps, rewards, alpha=0.3, label='Raw', color='tab:blue')
        if len(rewards) > smooth_window:
            smoothed = self._smooth(rewards, smooth_window)
            ax.plot(steps, smoothed, label='Smoothed', color='tab:blue', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Rewards over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Policy Loss
        ax = axes[0, 1]
        policy_loss = np.array(self.data['policy_loss'])
        ax.plot(steps, policy_loss, alpha=0.3, label='Raw', color='tab:orange')
        if len(policy_loss) > smooth_window:
            smoothed = self._smooth(policy_loss, smooth_window)
            ax.plot(steps, smoothed, label='Smoothed', color='tab:orange', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Value Loss
        ax = axes[1, 0]
        value_loss = np.array(self.data['value_loss'])
        ax.plot(steps, value_loss, alpha=0.3, label='Raw', color='tab:green')
        if len(value_loss) > smooth_window:
            smoothed = self._smooth(value_loss, smooth_window)
            ax.plot(steps, smoothed, label='Smoothed', color='tab:green', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: FPS
        ax = axes[1, 1]
        fps = np.array(self.data['fps'])
        valid_fps = fps[~np.isnan(fps)]
        if len(valid_fps) > 0:
            ax.plot(steps[:len(valid_fps)], valid_fps, alpha=0.5, color='tab:purple')
            if len(valid_fps) > smooth_window:
                smoothed = self._smooth(valid_fps, smooth_window)
                ax.plot(steps[:len(smoothed)], smoothed, color='tab:purple', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('FPS')
        ax.set_title('Training Speed (FPS)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.viz_dir / 'learning_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved learning curves to {save_path}")
    
    def plot_world_model_losses(self, smooth_window: int = 50):
        """
        Plot world model losses (dynamics, reward).
        
        Args:
            smooth_window: Window size for smoothing
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        steps = np.array(self.data['steps'])
        
        # Plot 1: Total WM Loss
        ax = axes[0]
        wm_loss = np.array(self.data['wm_loss'])
        valid = ~np.isnan(wm_loss)
        ax.plot(steps[valid], wm_loss[valid], alpha=0.3, label='Raw', color='tab:red')
        if valid.sum() > smooth_window:
            smoothed = self._smooth(wm_loss[valid], smooth_window)
            ax.plot(steps[valid][:len(smoothed)], smoothed, label='Smoothed', 
                   color='tab:red', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Total WM Loss')
        ax.set_title('World Model Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Dynamics Loss
        ax = axes[1]
        dyn_loss = np.array(self.data['dynamics_loss'])
        valid = ~np.isnan(dyn_loss)
        ax.plot(steps[valid], dyn_loss[valid], alpha=0.3, label='Raw', color='tab:cyan')
        if valid.sum() > smooth_window:
            smoothed = self._smooth(dyn_loss[valid], smooth_window)
            ax.plot(steps[valid][:len(smoothed)], smoothed, label='Smoothed', 
                   color='tab:cyan', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Dynamics Loss')
        ax.set_title('Dynamics Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Reward Loss
        ax = axes[2]
        rew_loss = np.array(self.data['reward_loss'])
        valid = ~np.isnan(rew_loss)
        ax.plot(steps[valid], rew_loss[valid], alpha=0.3, label='Raw', color='tab:pink')
        if valid.sum() > smooth_window:
            smoothed = self._smooth(rew_loss[valid], smooth_window)
            ax.plot(steps[valid][:len(smoothed)], smoothed, label='Smoothed', 
                   color='tab:pink', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Reward Loss')
        ax.set_title('Reward Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.viz_dir / 'world_model_losses.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved WM losses to {save_path}")
    
    def plot_gradient_norms(self, smooth_window: int = 50):
        """
        Plot gradient norms for all components.
        
        Args:
            smooth_window: Window size for smoothing
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = np.array(self.data['steps'])
        
        # Actor gradients
        actor_grad = np.array(self.data['actor_grad_norm'])
        valid = ~np.isnan(actor_grad)
        if valid.sum() > 0:
            ax.plot(steps[valid], actor_grad[valid], alpha=0.3, color='tab:blue')
            if valid.sum() > smooth_window:
                smoothed = self._smooth(actor_grad[valid], smooth_window)
                ax.plot(steps[valid][:len(smoothed)], smoothed, 
                       label='Actor', color='tab:blue', linewidth=2)
        
        # Critic gradients
        critic_grad = np.array(self.data['critic_grad_norm'])
        valid = ~np.isnan(critic_grad)
        if valid.sum() > 0:
            ax.plot(steps[valid], critic_grad[valid], alpha=0.3, color='tab:green')
            if valid.sum() > smooth_window:
                smoothed = self._smooth(critic_grad[valid], smooth_window)
                ax.plot(steps[valid][:len(smoothed)], smoothed, 
                       label='Critic', color='tab:green', linewidth=2)
        
        # WM gradients
        wm_grad = np.array(self.data['wm_grad_norm'])
        valid = ~np.isnan(wm_grad)
        if valid.sum() > 0:
            ax.plot(steps[valid], wm_grad[valid], alpha=0.3, color='tab:red')
            if valid.sum() > smooth_window:
                smoothed = self._smooth(wm_grad[valid], smooth_window)
                ax.plot(steps[valid][:len(smoothed)], smoothed, 
                       label='World Model', color='tab:red', linewidth=2)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms over Training')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        save_path = self.viz_dir / 'gradient_norms.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved gradient norms to {save_path}")
    
    def plot_summary_statistics(self):
        """
        Plot summary statistics and final distributions.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Reward distribution (last 20%)
        ax = axes[0, 0]
        rewards = np.array(self.data['rewards'])
        last_20pct = int(len(rewards) * 0.8)
        final_rewards = rewards[last_20pct:]
        final_rewards = final_rewards[~np.isnan(final_rewards)]
        if len(final_rewards) > 0:
            ax.hist(final_rewards, bins=30, alpha=0.7, color='tab:blue', edgecolor='black')
            ax.axvline(np.mean(final_rewards), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(final_rewards):.2f}')
            ax.axvline(np.median(final_rewards), color='green', linestyle='--', 
                      label=f'Median: {np.median(final_rewards):.2f}')
        ax.set_xlabel('Episode Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Final Reward Distribution (last 20%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Episode lengths
        ax = axes[0, 1]
        lengths = np.array(self.data['episode_lengths'])
        valid = ~np.isnan(lengths)
        if valid.sum() > 0:
            ax.plot(self.data['steps'], lengths, alpha=0.5, color='tab:orange')
            ax.axhline(np.mean(lengths[valid]), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(lengths[valid]):.1f}')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Training progress summary
        ax = axes[1, 0]
        metrics_summary = []
        labels = []
        
        for metric_name in ['actor_loss', 'value_loss', 'wm_loss']:
            data = np.array(self.data.get(metric_name, []))
            valid = ~np.isnan(data)
            if valid.sum() > 0:
                # Compare first 20% vs last 20%
                first_20 = data[valid][:int(valid.sum() * 0.2)]
                last_20 = data[valid][int(valid.sum() * 0.8):]
                if len(first_20) > 0 and len(last_20) > 0:
                    metrics_summary.append([np.mean(first_20), np.mean(last_20)])
                    labels.append(metric_name.replace('_', ' ').title())
        
        if metrics_summary:
            x = np.arange(len(labels))
            width = 0.35
            metrics_summary = np.array(metrics_summary)
            ax.bar(x - width/2, metrics_summary[:, 0], width, label='First 20%', alpha=0.7)
            ax.bar(x + width/2, metrics_summary[:, 1], width, label='Last 20%', alpha=0.7)
            ax.set_ylabel('Loss Value')
            ax.set_title('Loss Improvement (First 20% vs Last 20%)')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Best rewards timeline
        ax = axes[1, 1]
        rewards = np.array(self.data['rewards'])
        valid = ~np.isnan(rewards)
        if valid.sum() > 0:
            steps = np.array(self.data['steps'])[valid]
            rewards_valid = rewards[valid]
            best_rewards = np.maximum.accumulate(rewards_valid)
            ax.plot(steps, rewards_valid, alpha=0.3, color='tab:blue', label='Current')
            ax.plot(steps, best_rewards, color='tab:red', linewidth=2, label='Best')
            ax.fill_between(steps, rewards_valid, best_rewards, alpha=0.2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Reward')
        ax.set_title('Best Reward Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.viz_dir / 'summary_statistics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved summary statistics to {save_path}")
    
    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("\n" + "="*60)
        print("Generating visualizations...")
        print("="*60)
        
        self.plot_learning_curves()
        self.plot_world_model_losses()
        self.plot_gradient_norms()
        self.plot_summary_statistics()
        
        print("="*60)
        print(f"All visualizations saved to: {self.viz_dir}")
        print("="*60 + "\n")
    
    @staticmethod
    def _smooth(data: np.ndarray, window: int) -> np.ndarray:
        """
        Apply moving average smoothing.
        
        Args:
            data: Data to smooth
            window: Window size
        
        Returns:
            Smoothed data (same length as input)
        """
        if len(data) < window:
            return data
        
        # Use convolution with 'same' mode to return same-length array
        kernel = np.ones(window) / window
        smoothed = np.convolve(data, kernel, mode='same')
        return smoothed


def compare_runs(
    run_dirs: List[str],
    run_names: List[str],
    save_path: str = "comparison.png",
    metric: str = "rewards",
):
    """
    Compare multiple training runs.
    
    Args:
        run_dirs: List of run directories
        run_names: List of run names for legend
        save_path: Where to save comparison plot
        metric: Metric to compare
    """
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_dirs)))
    
    for run_dir, name, color in zip(run_dirs, run_names, colors):
        # Load data from visualizer's stored data
        # In practice, you'd load from saved CSV or pickle
        pass  # Implement based on your data storage format
    
    plt.xlabel('Steps')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()} Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to {save_path}")
