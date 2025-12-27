#!/bin/bash

# Monitor all retraining jobs
# Usage: ./scripts/monitor_retrain.sh

echo "========================================"
echo "PWM Retraining Jobs Status"
echo "========================================"
echo ""

# Job status
echo "📊 Job Queue:"
squeue -u $USER -o "%.10i %.9P %.30j %.8T %.10M %.4C %R" | head -20
echo ""

# H200 Jobs
echo "========================================" 
echo "H200 GPU - 5M Flow (Job 2193443)"
echo "========================================" 
if [ -f logs/retrain_5M_flow_2193443.out ]; then
    echo "Latest training progress:"
    grep -E "Iteration.*R:" logs/retrain_5M_flow_2193443.out | tail -5
    echo ""
    echo "Current status:"
    tail -3 logs/retrain_5M_flow_2193443.out
else
    echo "Log file not found yet (waiting for job to start)"
fi
echo ""

echo "========================================" 
echo "H200 GPU - 5M Baseline (Job 2193445)"
echo "========================================" 
if [ -f logs/retrain_5M_baseline_2193445.out ]; then
    echo "Latest training progress:"
    grep -E "Iteration.*R:" logs/retrain_5M_baseline_2193445.out | tail -5
    echo ""
    echo "Current status:"
    tail -3 logs/retrain_5M_baseline_2193445.out
else
    echo "Log file not found yet (waiting for job to start)"
fi
echo ""

# L40s Jobs
echo "========================================" 
echo "L40s GPU - 5M Flow (Job 2193447)"
echo "========================================" 
if [ -f logs/retrain_5M_flow_l40s_2193447.out ]; then
    echo "Latest training progress:"
    grep -E "Iteration.*R:" logs/retrain_5M_flow_l40s_2193447.out | tail -5
    echo ""
    echo "Current status:"
    tail -3 logs/retrain_5M_flow_l40s_2193447.out
else
    echo "Log file not found yet (waiting for job to start)"
fi
echo ""

# Performance comparison (if both have data)
echo "========================================" 
echo "📈 Performance Comparison"
echo "========================================" 
if grep -q "Iteration.*1000" logs/retrain_5M_flow_2193443.out 2>/dev/null && \
   grep -q "Iteration.*1000" logs/retrain_5M_flow_l40s_2193447.out 2>/dev/null; then
    
    echo "At Iteration ~1000:"
    echo "  H200 5M Flow: $(grep -E "\[1[0-9]{3}/20000\].*R:" logs/retrain_5M_flow_2193443.out | head -1 | grep -oP 'R:\K[\d.]+')"
    echo "  L40s 5M Flow: $(grep -E "\[1[0-9]{3}/20000\].*R:" logs/retrain_5M_flow_l40s_2193447.out | head -1 | grep -oP 'R:\K[\d.]+')"
fi
echo ""

echo "🔍 To view full logs:"
echo "  H200 Flow:     tail -f logs/retrain_5M_flow_2193443.out"
echo "  H200 Baseline: tail -f logs/retrain_5M_baseline_2193445.out"
echo "  L40s Flow:     tail -f logs/retrain_5M_flow_l40s_2193447.out"
echo ""

echo "Last updated: $(date)"
