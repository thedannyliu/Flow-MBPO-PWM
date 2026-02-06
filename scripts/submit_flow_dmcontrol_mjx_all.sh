#!/bin/bash

# Submit Flow-MBPO PWM jobs for DM Control tasks via MuJoCo Playground (MJX)
# Tests Flow v1, v2, v3 on 10 tasks with offline training

# Tasks (env config names)
TASKS=(
    "dmcontrol_mjx_walker"
    "dmcontrol_mjx_ballincup"
    "dmcontrol_mjx_acrobot"
    "dmcontrol_mjx_fish"
    "dmcontrol_mjx_pendulum"
    "dmcontrol_mjx_finger_hard"
    "dmcontrol_mjx_finger_easy"
    "dmcontrol_mjx_finger_spin"
    "dmcontrol_mjx_reacher_hard"
    "dmcontrol_mjx_reacher_easy"
)

# Flow algorithm variants
ALGORITHMS=(
    "pwm_5M_flow_v1_substeps2"
    "pwm_5M_flow_v2_substeps4"
    "pwm_5M_flow_v3_substeps8_euler"
)

# Seeds for robustness (using 1 seed for now)
SEEDS=(42)

echo "Submitting $(( ${#TASKS[@]} * ${#ALGORITHMS[@]} * ${#SEEDS[@]} )) jobs..."
echo "Tasks: ${#TASKS[@]}"
echo "Algorithms: ${#ALGORITHMS[@]}"
echo "Seeds: ${#SEEDS[@]}"
echo ""

job_count=0

for task in "${TASKS[@]}"; do
    for alg in "${ALGORITHMS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            job_count=$((job_count + 1))
            
            # Extract task name for display
            task_name="${task#dmcontrol_mjx_}"
            alg_short="${alg#pwm_5M_flow_}"
            
            job_name="flow_${task_name}_${alg_short}_s${seed}"
            
            echo "[$job_count] Submitting: $job_name"
            
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=embers
#SBATCH --output=/storage/home/hcoda1/6/nnguyen349/r-agarg35-0/logs/${job_name}_%j.out
#SBATCH --error=/storage/home/hcoda1/6/nnguyen349/r-agarg35-0/logs/${job_name}_%j.err

# Environment setup
source /storage/project/r-agarg35-0/nnguyen349/.conda_envs/pwm/bin/activate
export PYTHONPATH=/storage/home/hcoda1/6/nnguyen349/r-agarg35-0/Flow-MBPO-PWM/mujoco_playground:\$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

cd /storage/home/hcoda1/6/nnguyen349/r-agarg35-0/Flow-MBPO-PWM/scripts

echo "=========================================="
echo "Job: $job_name"
echo "Task: $task"
echo "Algorithm: $alg"
echo "Seed: $seed"
echo "Node: \$(hostname)"
echo "GPU: \$CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Run training
python train_dmcontrol_mjx.py \
    env=$task \
    alg=$alg \
    general.seed=$seed \
    ++wandb.name=${job_name} \
    ++wandb.notes='Flow-MBPO_DM-Control_MJX'

echo "Job completed: $job_name"
EOF
            
            # Small delay to avoid overwhelming the scheduler
            sleep 0.5
        done
    done
done

echo ""
echo "=========================================="
echo "All $job_count jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "=========================================="
