#!/bin/bash
################################################################################
# Cleanup Old Checkpoints Script
# 
# This script removes redundant intermediate checkpoints, keeping only:
# - init_policy.pt
# - best_policy.pt
# - latest_checkpoint.pt (most recent intermediate)
# - final_policy.pt + final_policy.buffer
#
# Usage:
#   ./scripts/cleanup_checkpoints.sh [--dry-run] [--all] [log_dir]
#
# Examples:
#   # Dry run (show what would be deleted):
#   ./scripts/cleanup_checkpoints.sh --dry-run
#
#   # Clean all logs directories:
#   ./scripts/cleanup_checkpoints.sh --all
#
#   # Clean specific directory:
#   ./scripts/cleanup_checkpoints.sh logs/pwm_5M_dflex_ant_seed42
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false
ALL_DIRS=false
TARGET_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --all)
            ALL_DIRS=true
            shift
            ;;
        *)
            TARGET_DIR="$1"
            shift
            ;;
    esac
done

print_usage() {
    echo "Usage: $0 [--dry-run] [--all] [log_dir]"
    echo ""
    echo "Options:"
    echo "  --dry-run    Show what would be deleted without actually deleting"
    echo "  --all        Process all log directories"
    echo "  log_dir      Specific log directory to clean"
    echo ""
    echo "Examples:"
    echo "  $0 --dry-run                           # Preview cleanup"
    echo "  $0 --all                               # Clean all logs"
    echo "  $0 logs/pwm_5M_dflex_ant_seed42        # Clean specific dir"
}

cleanup_directory() {
    local dir="$1"
    local total_deleted=0
    local total_size_saved=0
    
    echo -e "${BLUE}Checking: ${dir}${NC}"
    
    if [ ! -d "$dir" ]; then
        echo -e "${RED}  ✗ Directory not found${NC}"
        return
    fi
    
    # Find all intermediate checkpoints (PWM_iter*.pt pattern)
    local checkpoints=($(find "$dir" -maxdepth 1 -name "PWM_iter*.pt" -type f 2>/dev/null))
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo -e "${GREEN}  ✓ No intermediate checkpoints to clean${NC}"
        return
    fi
    
    echo -e "${YELLOW}  Found ${#checkpoints[@]} intermediate checkpoints${NC}"
    
    # Calculate total size
    for checkpoint in "${checkpoints[@]}"; do
        if [ -f "$checkpoint" ]; then
            size=$(stat -f%z "$checkpoint" 2>/dev/null || stat -c%s "$checkpoint" 2>/dev/null || echo 0)
            total_size_saved=$((total_size_saved + size))
            
            if [ "$DRY_RUN" = true ]; then
                size_mb=$(echo "scale=1; $size / 1024 / 1024" | bc)
                echo -e "${YELLOW}  [DRY RUN] Would delete: $(basename "$checkpoint") (${size_mb} MB)${NC}"
            else
                size_mb=$(echo "scale=1; $size / 1024 / 1024" | bc)
                echo -e "${GREEN}  Deleting: $(basename "$checkpoint") (${size_mb} MB)${NC}"
                rm -f "$checkpoint"
                total_deleted=$((total_deleted + 1))
            fi
        fi
    done
    
    # Summary
    total_size_mb=$(echo "scale=1; $total_size_saved / 1024 / 1024" | bc)
    if [ "$DRY_RUN" = true ]; then
        echo -e "${GREEN}  Summary: Would free ${total_size_mb} MB (${#checkpoints[@]} files)${NC}"
    else
        echo -e "${GREEN}  ✓ Cleaned ${total_deleted} files, freed ${total_size_mb} MB${NC}"
    fi
    
    # Show what's kept
    echo -e "${BLUE}  Kept checkpoints:${NC}"
    [ -f "$dir/init_policy.pt" ] && echo "    - init_policy.pt"
    [ -f "$dir/best_policy.pt" ] && echo "    - best_policy.pt"
    [ -f "$dir/latest_checkpoint.pt" ] && echo "    - latest_checkpoint.pt"
    [ -f "$dir/final_policy.pt" ] && echo "    - final_policy.pt"
    [ -f "$dir/final_policy.buffer" ] && echo "    - final_policy.buffer"
    
    echo ""
}

# Main logic
if [ "$ALL_DIRS" = true ]; then
    echo -e "${GREEN}Cleaning all log directories...${NC}"
    echo ""
    
    # Find all log directories containing checkpoints
    # Check both logs/ and outputs/ directories
    for base_dir in "logs" "outputs"; do
        if [ -d "$base_dir" ]; then
            # Find directories containing .pt files
            while IFS= read -r -d '' dir; do
                cleanup_directory "$dir"
            done < <(find "$base_dir" -type f -name "*.pt" -exec dirname {} \; | sort -u | tr '\n' '\0')
        fi
    done
    
elif [ -n "$TARGET_DIR" ]; then
    cleanup_directory "$TARGET_DIR"
    
else
    echo -e "${RED}Error: Must specify --all or provide a log directory${NC}"
    echo ""
    print_usage
    exit 1
fi

# Final summary
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}DRY RUN MODE - No files were deleted${NC}"
    echo -e "${YELLOW}Remove --dry-run flag to actually clean${NC}"
    echo -e "${YELLOW}========================================${NC}"
else
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Cleanup completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
fi

exit 0
