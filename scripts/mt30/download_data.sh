#!/bin/bash
# Download TD-MPC2 MT30/MT80 data and PWM checkpoints
# Usage: ./download_data.sh

set -e

# Configuration
DATA_ROOT="${1:-/home/hice1/eliu354/scratch/Data/tdmpc2}"
CKPT_ROOT="/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/checkpoints"

echo "=== TD-MPC2 and PWM Data Download Script ==="
echo "Data will be stored in: ${DATA_ROOT}"
echo "Checkpoints will be stored in: ${CKPT_ROOT}"

# Create directories
mkdir -p ${DATA_ROOT}/mt30
mkdir -p ${DATA_ROOT}/mt80
mkdir -p ${CKPT_ROOT}

echo ""
echo "=== Step 1: Download PWM Pretrained Checkpoints ==="
echo "Downloading from HuggingFace: imgeorgiev/pwm/multitask"

# Option 1: Using huggingface-cli (if available)
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli..."
    huggingface-cli download imgeorgiev/pwm multitask/mt30_48M_4900000.pt --local-dir ${CKPT_ROOT} --local-dir-use-symlinks False
    huggingface-cli download imgeorgiev/pwm multitask/mt80_48M_2700000.pt --local-dir ${CKPT_ROOT} --local-dir-use-symlinks False
else
    echo "huggingface-cli not found. Using wget..."
    # Direct download URLs
    wget -O ${CKPT_ROOT}/mt30_48M_4900000.pt "https://huggingface.co/imgeorgiev/pwm/resolve/main/multitask/mt30_48M_4900000.pt"
    wget -O ${CKPT_ROOT}/mt80_48M_2700000.pt "https://huggingface.co/imgeorgiev/pwm/resolve/main/multitask/mt80_48M_2700000.pt"
fi

echo ""
echo "=== Step 2: Download TD-MPC2 Dataset ==="
echo ""
echo "TD-MPC2 dataset must be downloaded manually from:"
echo "  https://www.tdmpc2.com/dataset"
echo ""
echo "Instructions:"
echo "1. Visit the website above"
echo "2. Download the MT30 and/or MT80 dataset chunks"
echo "3. Extract to:"
echo "   - MT30: ${DATA_ROOT}/mt30/"
echo "   - MT80: ${DATA_ROOT}/mt80/"
echo ""
echo "The data should contain .pt files with episode data."
echo ""

echo "=== Download Status ==="
echo "Checkpoints:"
ls -la ${CKPT_ROOT}/*.pt 2>/dev/null || echo "  No checkpoints found"
echo ""
echo "MT30 Data:"
ls -la ${DATA_ROOT}/mt30/*.pt 2>/dev/null | head -5 || echo "  No data found - please download manually"
echo ""
echo "MT80 Data:"
ls -la ${DATA_ROOT}/mt80/*.pt 2>/dev/null | head -5 || echo "  No data found - please download manually"
echo ""

echo "=== Data Paths for Experiments ==="
echo "MT30 Checkpoint: ${CKPT_ROOT}/mt30_48M_4900000.pt"
echo "MT80 Checkpoint: ${CKPT_ROOT}/mt80_48M_2700000.pt"
echo "MT30 Data Dir:   ${DATA_ROOT}/mt30"
echo "MT80 Data Dir:   ${DATA_ROOT}/mt80"
echo ""
echo "Done!"
