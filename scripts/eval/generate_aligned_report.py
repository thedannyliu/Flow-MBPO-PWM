#!/usr/bin/env python3
"""
Generate a CSV specifically for ALIGNED experiments (Ant, Anymal, Humanoid).
Filters by wandb project name containing 'aligned'.
"""
import os
import csv
import glob
import yaml

PROJECT_ROOT = "/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
EVAL_DIR = "/storage/scratch1/9/eliu354/flow_mbpo/eval_results"
OUTPUT_CSV = f"{PROJECT_ROOT}/scripts/eval/aligned_experiments_report.csv"

def get_config(timestamp_str):
    """Load config for a given timestamp."""
    date_part, time_part = timestamp_str.split('_')
    config_path = f"{PROJECT_ROOT}/scripts/outputs/{date_part}/{time_part}/.hydra/config.yaml"
    
    if not os.path.exists(config_path):
        return None
        
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except:
        return None

def main():
    eval_files = glob.glob(f"{EVAL_DIR}/eval_*.csv")
    print(f"Found {len(eval_files)} evaluation files.")
    
    rows = []
    
    for eval_file in eval_files:
        filename = os.path.basename(eval_file)
        timestamp_str = filename.replace("eval_", "").replace(".csv", "")
        
        cfg = get_config(timestamp_str)
        if not cfg:
            continue
            
        # Filter: only aligned experiments
        wb_project = cfg.get('wandb', {}).get('project', '') or ''
        if 'aligned' not in wb_project.lower():
            continue
            
        # Parse Task
        task = "Unknown"
        if 'ant' in wb_project.lower(): task = "Ant"
        elif 'anymal' in wb_project.lower(): task = "Anymal"
        elif 'humanoid' in wb_project.lower(): task = "Humanoid"
        
        # Parse Variant
        variant = "Unknown"
        alg = cfg.get('alg', {})
        if isinstance(alg, dict):
            actor_cfg = alg.get('actor_config', {})
            target = actor_cfg.get('_target_', '')
            flow_k = alg.get('flow_substeps', None)
            
            if 'ActorStochasticMLP' in target:
                variant = "Baseline"
            elif 'FlowActor' in target or 'FlowODE' in target:
                if flow_k:
                    variant = f"FlowWM_K{flow_k}"
                else:
                    variant = "FlowPolicy"
        
        seed = cfg.get('general', {}).get('seed', 'Unknown')
        rew_rms = alg.get('rew_rms', False) if isinstance(alg, dict) else False
        
        # Read eval result
        mean_reward = 0
        mean_len = 0
        try:
            with open(eval_file, 'r') as f:
                reader = csv.DictReader(f)
                row = next(reader)
                mean_reward = float(row.get('MeanReward', 0))
                mean_len = float(row.get('MeanLength', 0))
        except:
            continue
            
        rows.append({
            "Task": task,
            "Variant": variant,
            "Seed": seed,
            "Reward": round(mean_reward, 2),
            "EpLength": round(mean_len, 2),
            "rew_rms": rew_rms,
            "Timestamp": timestamp_str
        })
    
    # Sort by Task, Variant, Seed
    rows.sort(key=lambda x: (x['Task'], x['Variant'], str(x['Seed'])))
    
    # Write CSV
    headers = ["Task", "Variant", "Seed", "Reward", "EpLength", "rew_rms", "Timestamp"]
    with open(OUTPUT_CSV, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nGenerated aligned report: {OUTPUT_CSV}")
    print(f"Total aligned runs: {len(rows)}")
    
    # Summary by Task/Variant
    from collections import defaultdict
    summary = defaultdict(list)
    for r in rows:
        key = (r['Task'], r['Variant'])
        summary[key].append(r['Reward'])
    
    print("\n=== Summary by Task/Variant ===")
    print(f"{'Task':<12} {'Variant':<15} {'Count':<6} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 75)
    for (task, variant), rewards in sorted(summary.items()):
        import statistics
        mean = statistics.mean(rewards)
        std = statistics.stdev(rewards) if len(rewards) > 1 else 0
        print(f"{task:<12} {variant:<15} {len(rewards):<6} {mean:<10.2f} {std:<10.2f} {min(rewards):<10.2f} {max(rewards):<10.2f}")

if __name__ == "__main__":
    main()
