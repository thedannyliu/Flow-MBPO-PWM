import os
import math

# Define all remaining tasks
tasks = []

# Ant Remaining (s8, s9)
for seed in [8, 9]:
    tasks.append({
        "name": f"ant_Base_s{seed}",
        "cmd": f"python train_dflex.py env=dflex_ant alg=pwm_5M_baseline_final general.seed={seed} general.run_wandb=true ++wandb.project=flow-mbpo-aligned-ant ++wandb.name=Ant_Baseline_aligned_s{seed}"
    })
    tasks.append({
        "name": f"ant_Flow_s{seed}",
        "cmd": f"python train_dflex.py env=dflex_ant alg=pwm_5M_flow_v3_aligned general.seed={seed} general.run_wandb=true ++wandb.project=flow-mbpo-aligned-ant ++wandb.name=Ant_FlowWM_K8_aligned_s{seed}"
    })

# Anymal (s0-s9)
for seed in range(10):
    tasks.append({
        "name": f"any_Base_s{seed}",
        "cmd": f"python train_dflex.py env=dflex_anymal alg=pwm_5M_baseline_final general.seed={seed} general.run_wandb=true ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_Baseline_aligned_s{seed}"
    })
    tasks.append({
        "name": f"any_Flow_s{seed}",
        "cmd": f"python train_dflex.py env=dflex_anymal alg=pwm_5M_flowpolicy_aligned general.seed={seed} general.run_wandb=true ++wandb.project=flow-mbpo-aligned-anymal ++wandb.name=Anymal_FlowPolicy_aligned_s{seed}"
    })

# Humanoid (s0-s9)
for seed in range(10):
    tasks.append({
        "name": f"hum_Base_s{seed}",
        "cmd": f"python train_dflex.py env=dflex_humanoid alg=pwm_5M_baseline_final general.seed={seed} general.run_wandb=true ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_Baseline_aligned_s{seed}"
    })
    tasks.append({
        "name": f"hum_Flow_s{seed}",
        "cmd": f"python train_dflex.py env=dflex_humanoid alg=pwm_5M_flowpolicy_aligned general.seed={seed} general.run_wandb=true ++wandb.project=flow-mbpo-aligned-humanoid ++wandb.name=Humanoid_FlowPolicy_aligned_s{seed}"
    })

# Scaling & Tuning (Ant Flow)
tasks.append({
    "name": "ant_Flow_48M",
    "cmd": "python train_dflex.py env=dflex_ant alg=pwm_48M_flow general.seed=42 general.run_wandb=true ++wandb.project=flow-mbpo-scaling ++wandb.name=Ant_FlowWM_48M_s42 alg.flow_substeps=8"
})
tasks.append({
    "name": "ant_Flow_K16",
    "cmd": "python train_dflex.py env=dflex_ant alg=pwm_5M_flow_v3_aligned general.seed=42 general.run_wandb=true ++wandb.project=flow-mbpo-tuning ++wandb.name=Ant_FlowWM_K16_s42 alg.flow_substeps=16"
})
tasks.append({
    "name": "ant_Flow_K32",
    "cmd": "python train_dflex.py env=dflex_ant alg=pwm_5M_flow_v3_aligned general.seed=42 general.run_wandb=true ++wandb.project=flow-mbpo-tuning ++wandb.name=Ant_FlowWM_K32_s42 alg.flow_substeps=32"
})

# Parameters
JOBS_PER_NODE = 7  # Safe limit for memory (400GB / 7 approx 57GB per job)
PROJECT_ROOT = "/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM"
LOG_DIR = f"{PROJECT_ROOT}/logs/slurm/packed"
os.makedirs(LOG_DIR, exist_ok=True)

chunks = [tasks[i:i + JOBS_PER_NODE] for i in range(0, len(tasks), JOBS_PER_NODE)]

print(f"Total tasks: {len(tasks)}")
print(f"Total chunks: {len(chunks)} (filling {JOBS_PER_NODE} jobs per node)")

for i, chunk in enumerate(chunks):
    script_path = f"scripts/submit_packed_{i}.sh"
    
    # Construct the commands block
    cmds_txt = ""
    for j, task in enumerate(chunk):
        # We assign GPU j to this task (j goes 0..6)
        cmds_txt += f"""
echo "Starting {task['name']} on GPU {j}"
(
  export CUDA_VISIBLE_DEVICES={j}
  {task['cmd']}
) > {LOG_DIR}/{task['name']}.out 2> {LOG_DIR}/{task['name']}.err &
pids[{j}]=$!
        """
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=packed_batch_{i}
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=40:00:00
#SBATCH --mem=0
#SBATCH --output={LOG_DIR}/packed_{i}_%j.out
#SBATCH --error={LOG_DIR}/packed_{i}_%j.err

cd {PROJECT_ROOT}
source ~/.bashrc
conda activate pwm
export PYTHONPATH={PROJECT_ROOT}/src

echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L)"

# Launch parallel jobs
declare -a pids

cd scripts
{cmds_txt}

# Wait for all PIDs
for pid in "${{pids[@]}}"; do
    wait $pid
done

echo "All jobs in this batch finished."
"""
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    print(f"Created {script_path} with {len(chunk)} tasks")
    
    # Submit
    os.system(f"sbatch {script_path}")
