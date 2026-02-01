
import os
import time
import subprocess

# Fixed list of failed/hung jobs to resubmit
# Ant Flow s5 (Hung), Ant s8-9, Anymal s0-1 (OOM in Packed_0)
JOBS_TO_RESUBMIT = [
    # Batch 0
    "ant_Flow_s5", "ant_Base_s8", "ant_Flow_s8",
    # Batch 1
    "ant_Base_s9", "ant_Flow_s9", "anymal_Base_s0",
    # Batch 2
    "anymal_Flow_s0", "anymal_Base_s1", "anymal_Flow_s1"
]

# Map short names to full commands
JOB_MAP = {
    # Ant
    "ant_Base": "python scripts/train_dflex.py env=dflex_ant alg=pwm_5M_baseline_final general.seed={seed} ++wandb.name=ant_Baseline_s{seed} ++wandb.project=flow-mbpo-aligned-ant",
    "ant_Flow": "python scripts/train_dflex.py env=dflex_ant alg=pwm_48M_flow general.seed={seed} ++wandb.name=ant_Flow_s{seed} ++wandb.project=flow-mbpo-aligned-ant",
    
    # Anymal
    "anymal_Base": "python scripts/train_dflex.py env=dflex_anymal alg=pwm_5M_baseline_final general.seed={seed} ++wandb.name=Anymal_Baseline_aligned_s{seed} ++wandb.project=flow-mbpo-aligned-anymal",
    "anymal_Flow": "python scripts/train_dflex.py env=dflex_anymal alg=pwm_48M_flow general.seed={seed} ++wandb.name=Anymal_Flow_aligned_s{seed} ++wandb.project=flow-mbpo-aligned-anymal",
}

JOBS_PER_NODE = 3 # Safe density

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=packed_v5_{batch_id}
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=40:00:00
#SBATCH --output=logs/slurm/packed_v5_%j.out
#SBATCH --exclusive

cd $PROJECT_ROOT
conda activate pwm

echo "Starting Packed Batch {batch_id} with {num_jobs} jobs..."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

{job_commands}

wait
echo "Batch {batch_id} Complete"
"""

def main():
    jobs = []
    
    # Expand job list
    for item in JOBS_TO_RESUBMIT:
        parts = item.split('_s')
        base_key = parts[0]
        seed = int(parts[1])
        
        cmd_template = JOB_MAP.get(base_key)
        if not cmd_template:
            print(f"Error: Unknown job type {base_key}")
            continue
            
        cmd = cmd_template.format(seed=seed)
        jobs.append((item, cmd))

    # Split into batches
    chunks = [jobs[i:i + JOBS_PER_NODE] for i in range(0, len(jobs), JOBS_PER_NODE)]
    
    for i, chunk in enumerate(chunks):
        batch_script = ""
        batch_names = []
        
        for idx, (name, cmd) in enumerate(chunk):
            # Assign specific GPU to each job
            # Job 0 -> GPU 0, Job 1 -> GPU 1, etc.
            batch_script += f"CUDA_VISIBLE_DEVICES={idx} {cmd} > logs/slurm/resubmit_{name}.out 2>&1 &\n"
            batch_names.append(name)
            
        content = TEMPLATE.format(
            batch_id=i, 
            num_jobs=len(chunk),
            job_commands=batch_script
        )
        
        script_name = f"scripts/submit_v5_batch_{i}.sh"
        with open(script_name, "w") as f:
            f.write(content)
            
        print(f"Submitting Batch {i}: {batch_names}")
        subprocess.run(["sbatch", script_name])
        time.sleep(1)

if __name__ == "__main__":
    main()
