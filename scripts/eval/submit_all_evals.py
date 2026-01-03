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
EVAL_SCRIPT = SCRATCH / "scripts" / "eval_pwm.py"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Read all checkpoint directories
checkpoints = []
for task_file in ['ant_dirs.txt', 'anymal_dirs.txt', 'humanoid_dirs.txt']:
    filepath = SCRATCH / "checkpoints_to_eval" / task_file
    if filepath.exists():
        with open(filepath) as f:
            for line in f:
                dir_path = line.strip()
                if dir_path:
                    checkpoints.append(dir_path)

print(f"Found {len(checkpoints)} checkpoints to evaluate")

job_count = 0
for dir_path in checkpoints:
    ckpt_path = PROJECT / "scripts" / dir_path / "logs" / "best_policy.pt"
    if not ckpt_path.exists():
        continue
    
    # Parse date and time
    parts = dir_path.split('/')
    date = parts[1]  # e.g., 2025-12-29
    time = parts[2]  # e.g., 07-10-46
    ckpt_id = f"{date}_{time}"
    output_file = RESULTS_DIR / f"eval_{ckpt_id}.csv"
    
    # Skip if already evaluated
    if output_file.exists():
        print(f"SKIP: {ckpt_id} (already done)")
        continue
    
    # Create SLURM script
    script_content = f'''#!/bin/bash
#SBATCH --job-name=eval_{ckpt_id}
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=300GB
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output={LOG_DIR}/eval_{ckpt_id}_%j.out
#SBATCH --error={LOG_DIR}/eval_{ckpt_id}_%j.err

cd {PROJECT}
source ~/.bashrc
conda activate pwm
export PYTHONPATH={PROJECT}/src

python {EVAL_SCRIPT} --checkpoint "{ckpt_path}" --num-games 100 --output "{output_file}"
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
