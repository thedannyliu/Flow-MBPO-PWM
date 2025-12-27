#!/bin/bash
# Resume all failed/incomplete training jobs with fixes applied

set -e

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

echo "======================================"
echo "Resuming All Training Jobs with Fixes"
echo "======================================"
echo ""

# Job 1: pwm_5M baseline (completed 14994/15000, crashed at end due to log_metrics bug)
echo "[1/4] Resuming pwm_5M baseline (14994/15000 -> 15000)"
echo "--------------------------------------"
if [ -f "logs/pwm_5M_dflex_ant_seed42/checkpoints/checkpoint_epoch_14500.pt" ]; then
    echo "✓ Found checkpoint at epoch 14500"
    echo "Command: ./scripts/resume_training.sh pwm_5M dflex_ant 42"
    ./scripts/resume_training.sh pwm_5M dflex_ant 42
    echo "Submitted: Job ID will be shown above"
else
    echo "✗ No checkpoint found, submitting fresh job"
    ./scripts/submit_job.sh single pwm_5M dflex_ant 42
fi
echo ""

# Job 2: pwm_48M baseline (failed immediately due to task_dim bug)
echo "[2/4] Submitting pwm_48M baseline (0/15000, fresh start)"
echo "--------------------------------------"
echo "Previous failure: tensor shape mismatch (task_dim=96 bug, now fixed to 0)"
echo "Command: ./scripts/submit_job.sh single pwm_48M dflex_ant 42"
./scripts/submit_job.sh single pwm_48M dflex_ant 42
echo ""

# Job 3: pwm_5M_flow (OOM at 7737/15000, memory reduced)
echo "[3/4] Resuming pwm_5M_flow (7737/15000 -> 15000)"
echo "--------------------------------------"
if [ -f "logs/pwm_5M_flow_dflex_ant_seed42/checkpoints/checkpoint_epoch_7500.pt" ]; then
    echo "✓ Found checkpoint at epoch 7500"
    echo "Previous OOM killed with 96GB, now reduced batch_size & buffer"
    echo "Command: ./scripts/resume_training.sh pwm_5M_flow dflex_ant 42"
    ./scripts/resume_training.sh pwm_5M_flow dflex_ant 42
    echo "Submitted: Job ID will be shown above"
else
    echo "✗ No checkpoint found, submitting fresh job"
    ./scripts/submit_job.sh single pwm_5M_flow dflex_ant 42
fi
echo ""

# Job 4: pwm_48M_flow (OOM at 7644/15000, memory reduced + task_dim fixed)
echo "[4/4] Resuming pwm_48M_flow (7644/15000 -> 15000)"
echo "--------------------------------------"
if [ -f "logs/pwm_48M_flow_dflex_ant_seed42/checkpoints/checkpoint_epoch_7500.pt" ]; then
    echo "✓ Found checkpoint at epoch 7500"
    echo "Previous OOM killed with 96GB, now reduced batch_size & buffer + fixed task_dim"
    echo "Command: ./scripts/resume_training.sh pwm_48M_flow dflex_ant 42"
    ./scripts/resume_training.sh pwm_48M_flow dflex_ant 42
    echo "Submitted: Job ID will be shown above"
else
    echo "✗ No checkpoint found, submitting fresh job"
    ./scripts/submit_job.sh single pwm_48M_flow dflex_ant 42
fi
echo ""

echo "======================================"
echo "All Jobs Submitted!"
echo "======================================"
echo ""
echo "Check status with: squeue -u \$USER"
echo "Monitor logs in: logs/slurm/"
echo ""
echo "Expected completion times (H200 GPU):"
echo "  [1] pwm_5M baseline:    ~2 minutes  (6 epochs remaining)"
echo "  [2] pwm_48M baseline:   ~2 hours    (15000 epochs, 7.5 FPS)"
echo "  [3] pwm_5M_flow:        ~3.5 hours  (7263 epochs, 1.0 FPS)"
echo "  [4] pwm_48M_flow:       ~3.5 hours  (7356 epochs, 1.0 FPS)"
echo ""
