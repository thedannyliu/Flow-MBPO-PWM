#!/usr/bin/env python3
"""
V6 Recovery Script - Fixed version with:
1. source ~/.bashrc for conda activation
2. --exclude=atl1-1-03-004 to avoid defective node
3. Proper log directory creation
"""
import os
import time
import subprocess

PROJECT_ROOT = "/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
LOG_DIR = f"{PROJECT_ROOT}/logs/slurm/v6"
os.makedirs(LOG_DIR, exist_ok=True)

# Fixed list of failed/hung jobs to resubmit
JOBS_TO_RESUBMIT = [
    # Ant Flow s5 (was hung)
    ("ant_Flow_s5", f"python train_dflex.py env=dflex_ant alg=pwm_5M_flow_v3_aligned general.seed=5 ++wandb.project=flow-mbpo-aligned-ant ++wandb.name=Ant_FlowWM_K8_aligned_s5"),
    # Ant s8-9
    ("ant_Base_s8", f"python train_dflex.py env=dflex_ant alg=pwm_5M_baseline_final general.seed=8 ++wandb.project=flow-mbpo-aligned-ant ++wandb.name=Ant_Baseline_aligned_s8"),
    ("ant_Flow_s8", f"python train_dflex.py env=dflex_ant alg=pwm_5M_flow_v3_aligned general.seed=8 ++wandb.project=flow-mbpo-aligned-ant ++wandb.name=Ant_FlowWM_K8_aligned_s8"),
    ("ant_Base_s9", f"python train_dflex.py env=dflex_ant alg=pwm_5M_baseline_final general.seed=9 ++wandb.project=flow-mbpo-aligned-ant ++wandb.name=Ant_Baseline_aligned_s9"),
    ("ant_Flow_s9", f"python train_dflex.py env=dflex_ant alg=pwm_5M_flow_v3_aligned general.seed=9 ++wandb.project=flow-mbpo-aligned-ant ++wandb.name=Ant_FlowWM_K8_aligned_s9"),
    # Anymal s0-1
    ("any_Base_s0", f"python train_dflex.py env=dflex_anymal alg=pwm_5M_baseline_final general.seed=0 ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_Baseline_aligned_s0"),
    ("any_Flow_s0", f"python train_dflex.py env=dflex_anymal alg=pwm_5M_flowpolicy_aligned general.seed=0 ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_FlowPolicy_aligned_s0"),
    ("any_Base_s1", f"python train_dflex.py env=dflex_anymal alg=pwm_5M_baseline_final general.seed=1 ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_Baseline_aligned_s1"),
    ("any_Flow_s1", f"python train_dflex.py env=dflex_anymal alg=pwm_5M_flowpolicy_aligned general.seed=1 ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_FlowPolicy_aligned_s1"),
]

JOBS_PER_NODE = 3  # Safe density

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=v6_batch_{batch_id}
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=40:00:00
#SBATCH --mem=0
#SBATCH --exclude=atl1-1-03-004-29-0,atl1-1-03-004-31-0
#SBATCH --output={log_dir}/v6_batch_{batch_id}_%j.out
#SBATCH --error={log_dir}/v6_batch_{batch_id}_%j.err

cd {project_root}
source ~/.bashrc
conda activate pwm
export PYTHONPATH={project_root}/src

echo "=== V6 Batch {batch_id} ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L)"

cd scripts
declare -a pids

{job_commands}

# Wait for all PIDs
for pid in "${{pids[@]}}"; do
    wait $pid
done

echo "=== V6 Batch {batch_id} Complete ==="
"""

def main():
    # Split into batches
    chunks = [JOBS_TO_RESUBMIT[i:i + JOBS_PER_NODE] for i in range(0, len(JOBS_TO_RESUBMIT), JOBS_PER_NODE)]
    
    for i, chunk in enumerate(chunks):
        job_commands = ""
        batch_names = []
        
        for idx, (name, cmd) in enumerate(chunk):
            job_commands += f"""
echo "Starting {name} on GPU {idx}"
(
  export CUDA_VISIBLE_DEVICES={idx}
  {cmd}
) > {LOG_DIR}/{name}.out 2> {LOG_DIR}/{name}.err &
pids[{idx}]=$!
"""
            batch_names.append(name)
            
        content = TEMPLATE.format(
            batch_id=i, 
            log_dir=LOG_DIR,
            project_root=PROJECT_ROOT,
            job_commands=job_commands
        )
        
        script_path = f"{PROJECT_ROOT}/scripts/submit_v6_batch_{i}.sh"
        with open(script_path, "w") as f:
            f.write(content)
            
        print(f"Submitting V6 Batch {i}: {batch_names}")
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        print(result.stdout.strip())
        time.sleep(1)

if __name__ == "__main__":
    main()
