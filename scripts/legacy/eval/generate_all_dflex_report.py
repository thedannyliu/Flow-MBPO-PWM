#!/usr/bin/env python3
"""
Generate a comprehensive CSV for ALL Ant, Anymal, Humanoid experiments.
Detects Task from wandb.name, wandb.project, or env config.
"""
import os
import csv
import glob
import yaml
import statistics

PROJECT_ROOT = "/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
EVAL_DIR = "/storage/scratch1/9/eliu354/flow_mbpo/eval_results"
OUTPUT_CSV = f"{PROJECT_ROOT}/scripts/eval/all_dflex_experiments.csv"

def get_config(timestamp_str):
    """Load config for a given timestamp."""
    try:
        date_part, time_part = timestamp_str.split('_')
    except:
        return None
    config_path = f"{PROJECT_ROOT}/scripts/outputs/{date_part}/{time_part}/.hydra/config.yaml"
    
    if not os.path.exists(config_path):
        return None
        
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except:
        return None

def detect_task(cfg):
    """Detect task from config."""
    # Try wandb.project
    wb_project = (cfg.get('wandb', {}) or {}).get('project', '') or ''
    wb_name = (cfg.get('wandb', {}) or {}).get('name', '') or ''
    combined = (wb_project + wb_name).lower()
    
    if 'ant' in combined: return 'Ant'
    if 'anymal' in combined: return 'Anymal'
    if 'humanoid' in combined: return 'Humanoid'
    if 'hopper' in combined: return 'Hopper'
    
    # Try env
    env = cfg.get('env', {})
    env_str = str(env).lower()
    if 'ant' in env_str: return 'Ant'
    if 'anymal' in env_str: return 'Anymal'
    if 'humanoid' in env_str: return 'Humanoid'
    if 'hopper' in env_str: return 'Hopper'
    
    return 'Unknown'

def detect_variant(cfg):
    """Detect variant from config."""
    alg = cfg.get('alg', {})
    if not isinstance(alg, dict):
        return 'Unknown', 'Unknown', None
    
    actor_cfg = alg.get('actor_config', {})
    target = actor_cfg.get('_target_', '')
    flow_k = alg.get('flow_substeps', None)
    flow_int = alg.get('flow_integrator', 'euler')
    
    if 'ActorStochasticMLP' in target:
        return 'Baseline', 'Gaussian', None
    elif 'FlowActor' in target or 'FlowODE' in target:
        # Check if world model is also flow
        wm_cfg = alg.get('world_model_config', {})
        wm_target = wm_cfg.get('_target_', '') if wm_cfg else ''
        
        if 'FlowODE' in wm_target or 'flow' in str(wm_target).lower():
            variant = f"FullFlow_K{flow_k}" if flow_k else "FullFlow"
        elif flow_k:
            variant = f"FlowWM_K{flow_k}"
        else:
            variant = "FlowPolicy"
        return variant, 'Flow', flow_k
    
    return 'Unknown', 'Unknown', None

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
        
        task = detect_task(cfg)
        # Only include Ant, Anymal, Humanoid
        if task not in ['Ant', 'Anymal', 'Humanoid']:
            continue
            
        variant, actor_type, flow_k = detect_variant(cfg)
        seed = cfg.get('general', {}).get('seed', 'Unknown')
        
        alg = cfg.get('alg', {})
        rew_rms = alg.get('rew_rms', False) if isinstance(alg, dict) else False
        
        # Determine phase from wandb project
        wb_project = (cfg.get('wandb', {}) or {}).get('project', '') or ''
        if 'aligned' in wb_project.lower():
            phase = 'Aligned'
        else:
            phase = 'Legacy'
        
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
            "Phase": phase,
            "Task": task,
            "Variant": variant,
            "Seed": seed,
            "Reward": round(mean_reward, 2),
            "EpLength": round(mean_len, 2),
            "rew_rms": rew_rms,
            "FlowK": flow_k if flow_k else "N/A",
            "Actor": actor_type,
            "Timestamp": timestamp_str
        })
    
    # Sort by Phase, Task, Variant, Seed
    rows.sort(key=lambda x: (x['Phase'], x['Task'], x['Variant'], str(x['Seed'])))
    
    # Write CSV
    headers = ["Phase", "Task", "Variant", "Seed", "Reward", "EpLength", "rew_rms", "FlowK", "Actor", "Timestamp"]
    with open(OUTPUT_CSV, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nGenerated report: {OUTPUT_CSV}")
    print(f"Total runs: {len(rows)}")
    
    # Summary by Phase/Task/Variant
    from collections import defaultdict
    summary = defaultdict(list)
    for r in rows:
        key = (r['Phase'], r['Task'], r['Variant'])
        summary[key].append(r['Reward'])
    
    print("\n" + "=" * 90)
    print(f"{'Phase':<10} {'Task':<12} {'Variant':<18} {'N':<4} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("=" * 90)
    for (phase, task, variant), rewards in sorted(summary.items()):
        mean = statistics.mean(rewards)
        std = statistics.stdev(rewards) if len(rewards) > 1 else 0
        print(f"{phase:<10} {task:<12} {variant:<18} {len(rewards):<4} {mean:<10.2f} {std:<10.2f} {min(rewards):<10.2f} {max(rewards):<10.2f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
