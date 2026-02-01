#!/usr/bin/env python3
"""Fix variant labels in evaluation CSV files."""
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import glob

RESULTS_DIR = Path("/storage/scratch1/9/eliu354/flow_mbpo/eval_results")
PROJECT = Path("/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM")

def get_correct_variant(run_dir):
    """Determine correct variant from config."""
    config_path = run_dir / '.hydra' / 'config.yaml'
    if not config_path.exists():
        return None
    
    cfg = OmegaConf.load(config_path)
    alg = cfg.get('alg', {})
    
    use_flow_wm = alg.get('use_flow_dynamics', False)
    integrator = alg.get('flow_integrator', 'heun')
    substeps = alg.get('flow_substeps', 4)
    
    # Check actor class name (not module path)
    actor_target = str(alg.get('actor_config', {}).get('_target_', ''))
    actor_class = actor_target.split('.')[-1]  # Get just class name
    
    # FlowODE or FlowActor means flow policy
    flow_policy = 'FlowODE' in actor_class or 'FlowActor' in actor_class
    
    if use_flow_wm and flow_policy:
        return f"FullFlow_K{substeps}"
    elif use_flow_wm:
        return f"FlowWM_K{substeps}_{integrator}"
    elif flow_policy:
        return "FlowPolicy"
    else:
        return "Baseline"

# Process all eval files
for csv_file in glob.glob(str(RESULTS_DIR / "eval_*.csv")):
    if 'final' in csv_file:
        continue
    
    # Parse run dir from filename
    fname = Path(csv_file).stem  # e.g. eval_2025-12-29_20-17-36
    parts = fname.split('_')
    if len(parts) >= 3:
        date = parts[1]
        time = parts[2]
        run_dir = PROJECT / 'scripts' / 'outputs' / date / time
        
        if run_dir.exists():
            correct_variant = get_correct_variant(run_dir)
            if correct_variant:
                df = pd.read_csv(csv_file)
                old_variant = df.iloc[0]['Variant']
                if old_variant != correct_variant:
                    print(f"FIX: {fname}: {old_variant} -> {correct_variant}")
                    df['Variant'] = correct_variant
                    df.to_csv(csv_file, index=False)

print("\nDone fixing variants!")
