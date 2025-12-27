#!/bin/bash
################################################################################
# Monitor Evaluation Jobs
#
# Usage:
#   ./scripts/monitor_evaluation.sh
################################################################################

echo "=========================================="
echo "Evaluation Job Status"
echo "=========================================="
echo ""

# Check job queue
echo "Current jobs:"
squeue -u $USER -o "%.18i %.9P %.20j %.8T %.10M %.6D %R"
echo ""

# Find latest evaluation logs
EVAL_LOGS=$(find eval_logs/slurm -name "eval_*_dflex_ant_*.out" -type f 2>/dev/null | sort -r | head -2)

if [ -z "$EVAL_LOGS" ]; then
    echo "No evaluation logs found yet. Jobs are still queued."
    echo ""
    echo "When jobs start, monitor with:"
    echo "  watch -n 5 ./scripts/monitor_evaluation.sh"
    exit 0
fi

echo "Latest logs:"
echo ""

for LOG in $EVAL_LOGS; do
    JOBID=$(basename "$LOG" | grep -oP '\d+$' | head -1)
    MODELSIZE=$(basename "$LOG" | grep -oP '\d+M')
    
    echo "----------------------------------------"
    echo "Model: $MODELSIZE (Job: $JOBID)"
    echo "Log: $LOG"
    echo "----------------------------------------"
    
    if [ -f "$LOG" ]; then
        # Get job status
        STATE=$(squeue -j $JOBID -h -o "%T" 2>/dev/null || echo "COMPLETED/FAILED")
        echo "Status: $STATE"
        echo ""
        
        # Show last 20 lines
        echo "Last 20 lines:"
        tail -20 "$LOG"
        echo ""
        
        # Check if evaluation completed
        if grep -q "Evaluation Complete" "$LOG"; then
            echo "✅ EVALUATION COMPLETE!"
            echo ""
            
            # Show results if available
            OUTPUT_DIR=$(grep "Output directory:" "$LOG" | tail -1 | awk '{print $3}')
            if [ -n "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/comparison.csv" ]; then
                echo "Results:"
                cat "$OUTPUT_DIR/comparison.csv"
                echo ""
                echo "Plot: $OUTPUT_DIR/comparison.png"
            fi
        elif grep -q "FAILED" "$LOG"; then
            echo "❌ EVALUATION FAILED"
            echo ""
            echo "Error details:"
            grep -A 5 "Error\|Exception\|Traceback" "$LOG" | tail -20
        else
            echo "⏳ Still running..."
            
            # Show progress if available
            EPISODES=$(grep -oP "Episode \d+/\d+" "$LOG" | tail -1)
            if [ -n "$EPISODES" ]; then
                echo "Progress: $EPISODES"
            fi
        fi
    else
        echo "Log file not created yet."
    fi
    
    echo ""
done

echo "=========================================="
echo "Refresh this view with:"
echo "  watch -n 5 ./scripts/monitor_evaluation.sh"
echo "=========================================="
