#!/usr/bin/env python3
"""Submit all eval jobs using Python for cleaner path handling."""
import os
import subprocess
import tempfile
from pathlib import Path

PROJECT = Path("/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM")
SCRATCH = Path("/storage/scratch1/9/eliu354/flow_mbpo")
RESULTS_DIR = SCRATCH / "eval_results"
LOG_DIR = SCRATCH / "logs"
EVAL_SCRIPT = PROJECT / "scripts" / "eval" / "eval_pwm.py"  # Correct path to eval script
SEARCH_DIR = PROJECT / "scripts" / "outputs"  # Correct path to outputs

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Find all best_policy.pt files recursively
print(f"Searching for checkpoints in {SEARCH_DIR}...")
checkpoints = sorted(list(SEARCH_DIR.rglob("best_policy.pt"))) # Sort for consistency
print(f"Found {len(checkpoints)} checkpoints (total)")

job_count = 0
for ckpt_path in checkpoints:
    # ckpt_path is like .../outputs/2026-01-04/01-11-20/logs/best_policy.pt
    # We want a unique ID. 
    # Let's use date_time relative to outputs
    # e.g. 2026-01-04_01-11-20
    
    try:
        # relative path from outputs
        rel_path = ckpt_path.relative_to(SEARCH_DIR)
        # parts: [date, time, logs, best_policy.pt] or [date, time, best_policy.pt] depending on structure
        # Standard structure seems to be date/time/logs/best_policy.pt or date/time/best_policy.pt
        parts = rel_path.parts
        if len(parts) >= 2:
            ckpt_id = f"{parts[0]}_{parts[1]}"
        else:
            ckpt_id = str(rel_path).replace('/', '_').replace('.pt', '')
            
    except ValueError:
        ckpt_id = ckpt_path.name

    output_file = RESULTS_DIR / f"eval_{ckpt_id}.csv"
    
    # Skip if already evaluated
    if output_file.exists():
        # Check size to ensure it's not empty
        if output_file.stat().st_size > 10:
             print(f"SKIP: {ckpt_id} (already evaluated)")
             continue
    
    # Create SLURM script
    # Use exclusive if possible or just request enough resources. 
    # Eval is cpu-heavy for simulation but lightweight on GPU. 
    # But we want to be safe.
    
    script_content = f'''#!/bin/bash
#SBATCH --job-name=eval_{ckpt_id}
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output={LOG_DIR}/eval_{ckpt_id}_%j.out
#SBATCH --error={LOG_DIR}/eval_{ckpt_id}_%j.err

cd {PROJECT}
source ~/.bashrc
conda activate pwm
export PYTHONPATH={PROJECT}/src

# Run evaluation
python {EVAL_SCRIPT} --checkpoint "{ckpt_path}" --num-games 20 --output "{output_file}"
'''
    
    # Write to temp file and submit
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sbatch', delete=False) as f:
        f.write(script_content)
        tmp_path = f.name
    
    try:
        result = subprocess.run(['sbatch', tmp_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Submitted: {ckpt_id}")
            job_count += 1
        else:
            print(f"FAILED: {ckpt_id} - {result.stderr}")
    finally:
        os.unlink(tmp_path)

print(f"\nSubmitted {job_count} evaluation jobs")
