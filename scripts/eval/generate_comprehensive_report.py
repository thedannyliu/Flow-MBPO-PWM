
import os
import csv
import glob
import re
import yaml
from pathlib import Path

# Paths
PROJECT_ROOT = "/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
DOCS_LOG = f"{PROJECT_ROOT}/docs/experiment_log.md"
EVAL_DIR = "/storage/scratch1/9/eliu354/flow_mbpo/eval_results"
OUTPUT_DIR = f"{PROJECT_ROOT}/scripts/eval"
OUTPUT_CSV = f"{OUTPUT_DIR}/comprehensive_eval_report.csv"

def parse_experiment_log():
    """Parses experiment_log.md to map Timestamp -> {Hardware, Runtime, JobID, Phase}"""
    print(f"Parsing {DOCS_LOG}...")
    
    registry = {}
    current_phase = "Unknown"
    
    with open(DOCS_LOG, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        
        # Detect Phases (Major and Minor headers)
        if line.startswith("# ALIGNED EXPERIMENTS V2"):
            current_phase = "Aligned V2"
        elif line.startswith("# ALIGNED EXPERIMENTS V1"):
            current_phase = "Aligned V1 (Failed)"
        elif line.startswith("# ACTIVE JOBS") or line.startswith("# ACTIVE EXPERIMENTS"):
            current_phase = "Active (Jan 5)"
        elif line.startswith("# COMPLETED EXPERIMENTS (Legacy)"):
            current_phase = "Legacy (Dec 2025)"
        elif line.startswith("## Aligned V5"):
            current_phase = "Aligned V5 (Recovery)"
        elif line.startswith("## Aligned V4"):
            current_phase = "Aligned V4 (Packed)"
        elif line.startswith("## Aligned V6"):
            current_phase = "Aligned V6 (Recovery)"
            
        # Parse Table Rows
        # Looking for lines with | Job ID | ...
        if line.startswith("|") and "Job ID" not in line and "---" not in line:
            # Simple heuristic: Look for `outputs/YYYY-MM-DD/HH-MM-SS` pattern
            storage_match = re.search(r'outputs/(\d{4}-\d{2}-\d{2}/\d{2}-\d{2}-\d{2})', line)
            
            if storage_match:
                timestamp_path = storage_match.group(1) # e.g. 2026-01-04/01-11-20
                key = timestamp_path.replace('/', '_')  # e.g. 2026-01-04_01-11-20
                
                # Extract columns based on regex
                
                # Runtime
                runtime = "Unknown"
                runtime_match = re.search(r'(\d{2}:\d{2}:\d{2})', line)
                if runtime_match:
                    runtime = runtime_match.group(1)
                elif "~" in line:
                    for w in line.split("|"):
                        if "~" in w:
                            runtime = w.strip()
                            break
                            
                # Hardware
                hardware = "Unknown"
                hw_match = re.search(r'(atl1-[\w-]+)', line)
                if hw_match:
                    hardware = hw_match.group(1)
                    
                job_id = "Unknown"
                job_match = re.search(r'\| (\d{7}) \|', line)
                if job_match:
                    job_id = job_match.group(1)
                    
                registry[key] = {
                    "Phase": current_phase,
                    "JobID": job_id,
                    "Runtime": runtime,
                    "Hardware": hardware,
                    "Hardware": hardware,
                    "LogLine": line
                }
    
    print(f"Found {len(registry)} entries in experiment_log.md")
    return registry

def load_config_params(timestamp_str):
    """Loads key params from .hydra/config.yaml for a given timestamp."""
    date_part, time_part = timestamp_str.split('_')
    config_path = f"{PROJECT_ROOT}/scripts/outputs/{date_part}/{time_part}/.hydra/config.yaml"
    
    params = {
        "Task": "Unknown",
        "Variant": "Unknown",
        "Seed": "Unknown",
        "rew_rms": "Unknown",
        "actor_type": "Unknown",
        "flow_k": "Unknown",
        "flow_int": "Unknown"
    }
    
    if not os.path.exists(config_path):
        return params
        
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        # Parse Task from WandB or Env
        wb_project = cfg.get('wandb', {}).get('project', '')
        if 'ant' in wb_project: params['Task'] = 'Ant'
        elif 'anymal' in wb_project: params['Task'] = 'Anymal'
        elif 'humanoid' in wb_project: params['Task'] = 'Humanoid'
        else:
            # Fallback to env
            env = cfg.get('env', {})
            # env might be a dict or string depending on hydra
            env_str = str(env)
            if 'ant' in env_str: params['Task'] = 'Ant'
            elif 'anymal' in env_str: params['Task'] = 'Anymal'
            elif 'humanoid' in env_str: params['Task'] = 'Humanoid'
        
        params['Seed'] = cfg.get('general', {}).get('seed', 'Unknown')
        
        # Deep checks for params
        if 'alg' in cfg and isinstance(cfg['alg'], dict):
            params['rew_rms'] = cfg['alg'].get('rew_rms', False)
            params['flow_k'] = cfg['alg'].get('flow_substeps', 'N/A')
            params['flow_int'] = cfg['alg'].get('flow_integrator', 'N/A')
            
            actor_cfg = cfg['alg'].get('actor_config', {})
            target = actor_cfg.get('_target_', '')
            if 'ActorStochasticMLP' in target:
                params['actor_type'] = 'Baseline'
                params['Variant'] = 'Baseline'
            elif 'FlowActor' in target or 'FlowODE' in target:
                params['actor_type'] = 'Flow'
                if params['flow_k'] != 'N/A':
                    params['Variant'] = f"Flow_K{params['flow_k']}"
                else:
                    params['Variant'] = "FlowPolicy"
            
    except Exception as e:
        print(f"Error parse config {config_path}: {e}")
        
    return params

def main():
    # 1. Get Registry from Docs
    doc_registry = parse_experiment_log()
    
    # 2. Find all Eval CSVs
    eval_files = glob.glob(f"{EVAL_DIR}/eval_*.csv")
    print(f"Found {len(eval_files)} evaluation files.")
    
    final_rows = []
    
    for eval_file in eval_files:
        # Filename format: eval_2026-01-04_01-11-20.csv
        filename = os.path.basename(eval_file)
        timestamp_str = filename.replace("eval_", "").replace(".csv", "")
        
        # Read Eval Data
        mean_reward = 0
        mean_len = 0
        try:
            with open(eval_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    if 'MeanReward' in rows[0]:
                        mean_reward = float(rows[0]['MeanReward'])
                        mean_len = float(rows[0]['MeanLength']) if 'MeanLength' in rows[0] else 0
                    elif 'episode_reward' in rows[0]:
                        # Legacy format aggregation
                        rews = [float(r['episode_reward']) for r in rows]
                        mean_reward = sum(rews) / len(rews)
        except Exception as e:
            print(f"Error reading {eval_file}: {e}")
            continue
            
        # Get Config Data
        cfg_data = load_config_params(timestamp_str)
        
        # Get Doc Registry Data
        doc_data = doc_registry.get(timestamp_str, {})
        
        # Merge
        entry = {
            "Timestamp": timestamp_str,
            "Phase": doc_data.get("Phase", "Legacy/Other"),
            "Task": cfg_data['Task'],
            "Variant": cfg_data['Variant'],
            "Seed": cfg_data['Seed'],
            "Reward": round(mean_reward, 2),
            "Length": round(mean_len, 2),
            "Runtime": doc_data.get("Runtime", "Unknown"),
            "Hardware": doc_data.get("Hardware", "Unknown"),
            "JobID": doc_data.get("JobID", "Unknown"),
            "Rew_RMS": cfg_data['rew_rms'],
            "Actor": cfg_data['actor_type'],
            "Steps(K)": cfg_data['flow_k']
        }
        
        final_rows.append(entry)
        
    # Sort by Task, Phase, Variant, Seed
    final_rows.sort(key=lambda x: (x['Task'], x['Phase'], x['Variant'], str(x['Seed'])))
    
    # Write CSV
    headers = ["Phase", "Task", "Variant", "Seed", "Reward", "Length", "Runtime", "Hardware", "Rew_RMS", "Steps(K)", "Actor", "JobID", "Timestamp"]
    
    with open(OUTPUT_CSV, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(final_rows)
        
    print(f"Generated comprehensive report at {OUTPUT_CSV}")
    
    # Print preview
    print("\nPreview Top 10:")
    print(f"{'Task':<10} | {'Phase':<20} | {'Variant':<15} | {'Reward':<8} | {'Hardware':<15}")
    print("-" * 80)
    for row in final_rows[:10]:
        print(f"{row['Task']:<10} | {row['Phase']:<20} | {row['Variant']:<15} | {row['Reward']:<8} | {row['Hardware']:<15}")

if __name__ == "__main__":
    main()
