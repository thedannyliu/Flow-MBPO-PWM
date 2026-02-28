#!/bin/bash
# Master script to submit all 5M training jobs (1 baseline + 3 flow variants)

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

echo "========================================="
echo "Submitting 5M Training Jobs on L40s"
echo "Date: $(date)"
echo "========================================="
echo ""

# Make scripts executable
chmod +x scripts/submit_5M_baseline_l40s_final.sh
chmod +x scripts/submit_5M_flow_v1_l40s_final.sh
chmod +x scripts/submit_5M_flow_v2_l40s_final.sh
chmod +x scripts/submit_5M_flow_v3_l40s_final.sh

# Submit baseline (PWM paper reproduction)
echo "1. Submitting 5M Baseline..."
BASELINE_JOB=$(sbatch scripts/submit_5M_baseline_l40s_final.sh | awk '{print $4}')
echo "   Job ID: $BASELINE_JOB"
echo "   Expected: R ~ 1200 (15k iters, ~3-4 hours)"
echo ""

# Submit Flow V1 (substeps=2, conservative)
echo "2. Submitting 5M Flow V1 (substeps=2, heun)..."
FLOW_V1_JOB=$(sbatch scripts/submit_5M_flow_v1_l40s_final.sh | awk '{print $4}')
echo "   Job ID: $FLOW_V1_JOB"
echo "   Config: flow_substeps=2, heun integrator"
echo ""

# Submit Flow V2 (substeps=4, balanced)
echo "3. Submitting 5M Flow V2 (substeps=4, heun)..."
FLOW_V2_JOB=$(sbatch scripts/submit_5M_flow_v2_l40s_final.sh | awk '{print $4}')
echo "   Job ID: $FLOW_V2_JOB"
echo "   Config: flow_substeps=4, heun integrator"
echo ""

# Submit Flow V3 (substeps=8, euler)
echo "4. Submitting 5M Flow V3 (substeps=8, euler)..."
FLOW_V3_JOB=$(sbatch scripts/submit_5M_flow_v3_l40s_final.sh | awk '{print $4}')
echo "   Job ID: $FLOW_V3_JOB"
echo "   Config: flow_substeps=8, euler integrator"
echo ""

echo "========================================="
echo "All jobs submitted!"
echo "========================================="
echo ""
echo "Job IDs:"
echo "  Baseline: $BASELINE_JOB"
echo "  Flow V1:  $FLOW_V1_JOB"
echo "  Flow V2:  $FLOW_V2_JOB"
echo "  Flow V3:  $FLOW_V3_JOB"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/train_5M_*_l40s_*.out"
echo ""
echo "Expected timeline:"
echo "  - Baseline: ~3-4 hours (15k iters)"
echo "  - Flow models: ~4-5 hours (20k iters)"
echo ""
