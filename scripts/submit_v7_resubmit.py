#!/usr/bin/env python3
"""
V7 Resubmission Script - Submit all missing aligned experiments.
Uses --nodelist to force valid node and packed batches (3 jobs/node).
"""
import os
import subprocess
import time

PROJECT_ROOT = "/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
LOG_DIR = f"{PROJECT_ROOT}/logs/slurm/v7"
os.makedirs(LOG_DIR, exist_ok=True)

# All missing runs to submit
MISSING_RUNS = [
    # Ant FlowWM_K8: s0, s8
    ("Ant_FlowWM_K8_s0", "python train_dflex.py env=dflex_ant alg=pwm_5M_flow_v3_aligned general.seed=0 ++wandb.project=flow-mbpo-aligned-ant ++wandb.name=Ant_FlowWM_K8_aligned_s0"),
    ("Ant_FlowWM_K8_s8", "python train_dflex.py env=dflex_ant alg=pwm_5M_flow_v3_aligned general.seed=8 ++wandb.project=flow-mbpo-aligned-ant ++wandb.name=Ant_FlowWM_K8_aligned_s8"),
    
    # Anymal Baseline: s2-8
    ("Anymal_Base_s2", "python train_dflex.py env=dflex_anymal alg=pwm_5M_baseline_final general.seed=2 ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_Baseline_aligned_s2"),
    ("Anymal_Base_s3", "python train_dflex.py env=dflex_anymal alg=pwm_5M_baseline_final general.seed=3 ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_Baseline_aligned_s3"),
    ("Anymal_Base_s4", "python train_dflex.py env=dflex_anymal alg=pwm_5M_baseline_final general.seed=4 ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_Baseline_aligned_s4"),
    ("Anymal_Base_s5", "python train_dflex.py env=dflex_anymal alg=pwm_5M_baseline_final general.seed=5 ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_Baseline_aligned_s5"),
    ("Anymal_Base_s6", "python train_dflex.py env=dflex_anymal alg=pwm_5M_baseline_final general.seed=6 ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_Baseline_aligned_s6"),
    ("Anymal_Base_s7", "python train_dflex.py env=dflex_anymal alg=pwm_5M_baseline_final general.seed=7 ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_Baseline_aligned_s7"),
    ("Anymal_Base_s8", "python train_dflex.py env=dflex_anymal alg=pwm_5M_baseline_final general.seed=8 ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_Baseline_aligned_s8"),
    
    # Anymal FlowPolicy: s2, s6
    ("Anymal_Flow_s2", "python train_dflex.py env=dflex_anymal alg=pwm_5M_flowpolicy_aligned general.seed=2 ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_FlowPolicy_aligned_s2"),
    ("Anymal_Flow_s6", "python train_dflex.py env=dflex_anymal alg=pwm_5M_flowpolicy_aligned general.seed=6 ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_FlowPolicy_aligned_s6"),
    
    # Humanoid Baseline: s0, s1, s2, s5, s7, s8, s9
    ("Humanoid_Base_s0", "python train_dflex.py env=dflex_humanoid alg=pwm_5M_baseline_final general.seed=0 ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_Baseline_aligned_s0"),
    ("Humanoid_Base_s1", "python train_dflex.py env=dflex_humanoid alg=pwm_5M_baseline_final general.seed=1 ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_Baseline_aligned_s1"),
    ("Humanoid_Base_s2", "python train_dflex.py env=dflex_humanoid alg=pwm_5M_baseline_final general.seed=2 ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_Baseline_aligned_s2"),
    ("Humanoid_Base_s5", "python train_dflex.py env=dflex_humanoid alg=pwm_5M_baseline_final general.seed=5 ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_Baseline_aligned_s5"),
    ("Humanoid_Base_s7", "python train_dflex.py env=dflex_humanoid alg=pwm_5M_baseline_final general.seed=7 ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_Baseline_aligned_s7"),
    ("Humanoid_Base_s8", "python train_dflex.py env=dflex_humanoid alg=pwm_5M_baseline_final general.seed=8 ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_Baseline_aligned_s8"),
    ("Humanoid_Base_s9", "python train_dflex.py env=dflex_humanoid alg=pwm_5M_baseline_final general.seed=9 ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_Baseline_aligned_s9"),
    
    # Humanoid FlowPolicy: s0, s2, s3, s5, s7, s9
    ("Humanoid_Flow_s0", "python train_dflex.py env=dflex_humanoid alg=pwm_5M_flowpolicy_aligned general.seed=0 ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_FlowPolicy_aligned_s0"),
    ("Humanoid_Flow_s2", "python train_dflex.py env=dflex_humanoid alg=pwm_5M_flowpolicy_aligned general.seed=2 ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_FlowPolicy_aligned_s2"),
    ("Humanoid_Flow_s3", "python train_dflex.py env=dflex_humanoid alg=pwm_5M_flowpolicy_aligned general.seed=3 ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_FlowPolicy_aligned_s3"),
    ("Humanoid_Flow_s5", "python train_dflex.py env=dflex_humanoid alg=pwm_5M_flowpolicy_aligned general.seed=5 ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_FlowPolicy_aligned_s5"),
    ("Humanoid_Flow_s7", "python train_dflex.py env=dflex_humanoid alg=pwm_5M_flowpolicy_aligned general.seed=7 ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_FlowPolicy_aligned_s7"),
    ("Humanoid_Flow_s9", "python train_dflex.py env=dflex_humanoid alg=pwm_5M_flowpolicy_aligned general.seed=9 ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_FlowPolicy_aligned_s9"),
]

JOBS_PER_NODE = 3  # Conservative to avoid OOM
HEALTHY_NODES = "atl1-1-01-010-29-0,atl1-1-01-010-33-0,atl1-1-01-010-35-0,atl1-1-03-007-29-0,atl1-1-03-007-31-0,atl1-1-01-004-33-0"

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=v7_batch_{batch_id}
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=40:00:00
#SBATCH --mem=0
#SBATCH --nodelist={healthy_nodes}
#SBATCH --output={log_dir}/v7_batch_{batch_id}_%j.out
#SBATCH --error={log_dir}/v7_batch_{batch_id}_%j.err

cd {project_root}
source ~/.bashrc
conda activate pwm
export PYTHONPATH={project_root}/src

echo "=== V7 Batch {batch_id} ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L)"

cd scripts
declare -a pids

{job_commands}

# Wait for all PIDs
for pid in "${{pids[@]}}"; do
    wait $pid
done

echo "=== V7 Batch {batch_id} Complete ==="
"""

def main():
    print(f"Total missing runs: {len(MISSING_RUNS)}")
    
    # Split into batches
    chunks = [MISSING_RUNS[i:i + JOBS_PER_NODE] for i in range(0, len(MISSING_RUNS), JOBS_PER_NODE)]
    print(f"Total batches: {len(chunks)} ({JOBS_PER_NODE} jobs/node)")
    
    for i, chunk in enumerate(chunks):
        if i == 1:
            print(f"Skipping Batch {i} (already running on valid node)")
            continue
            
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
            healthy_nodes=HEALTHY_NODES,
            job_commands=job_commands
        )
        
        script_path = f"{PROJECT_ROOT}/scripts/submit_v7_batch_{i}.sh"
        with open(script_path, "w") as f:
            f.write(content)
            
        print(f"Submitting V7 Batch {i}: {batch_names}")
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        print(f"  {result.stdout.strip()}")
        time.sleep(1)
    
    print(f"\nSubmitted {len(chunks)} batches with {len(MISSING_RUNS)} total jobs.")

if __name__ == "__main__":
    main()
