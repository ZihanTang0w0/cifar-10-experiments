#!/bin/bash

# ============================================================================
# Script: Generate CIFAR-10 datasets with varying CFG scales (CLI version)
# Usage: ./run_generation_cli.sh <mode> <checkpoint> <cfg_start> <cfg_end> <cfg_stride>
# Example: ./run_generation_cli.sh diffusion model.pt 1.0 5.0 0.5
# ============================================================================

# Check arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <mode> <checkpoint> <cfg_start> <cfg_end> <cfg_stride>"
    echo ""
    echo "Arguments:"
    echo "  mode         : 'diffusion' or 'flow'"
    echo "  checkpoint   : Path to model checkpoint (.pt file)"
    echo "  cfg_start    : Starting CFG scale (e.g., 1.0)"
    echo "  cfg_end      : Ending CFG scale (e.g., 5.0)"
    echo "  cfg_stride   : Stride between values (e.g., 0.5)"
    echo ""
    echo "Example:"
    echo "  $0 diffusion checkpoints/model_epoch50.pt 1.0 5.0 0.5"
    exit 1
fi

# Parse arguments
MODE="$1"
CHECKPOINT="$2"
CFG_START="$3"
CFG_END="$4"
CFG_STRIDE="$5"

# Fixed parameters
NUM_PER_CLASS=1000
TOTAL_STEPS=100
BASE_CHANNELS=128
BATCH_SIZE=100
FILENAME="generated_batch"

# Validate mode
if [ "$MODE" != "diffusion" ] && [ "$MODE" != "flow" ]; then
    echo "Error: mode must be 'diffusion' or 'flow'"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file '$CHECKPOINT' not found"
    exit 1
fi

echo "========================================="
echo "  CIFAR-10 Dataset Generation Script"
echo "========================================="
echo "Mode:           $MODE"
echo "Checkpoint:     $CHECKPOINT"
echo "Num per class:  $NUM_PER_CLASS"
echo "Total steps:    $TOTAL_STEPS"
echo "CFG range:      $CFG_START to $CFG_END (stride: $CFG_STRIDE)"
echo "========================================="
echo ""

# Generate CFG scale values
cfg_values=$(awk -v start="$CFG_START" -v end="$CFG_END" -v stride="$CFG_STRIDE" '
BEGIN {
    for (i = start; i <= end; i += stride) {
        printf "%.1f ", i
    }
}')

cfg_array=($cfg_values)

echo "Will generate ${#cfg_array[@]} dataset(s) with CFG scales: ${cfg_array[*]}"
echo ""

# Loop over each CFG scale value
for cfg_scale in "${cfg_array[@]}"; do
    echo "========================================="
    echo "Running with CFG scale: $cfg_scale"
    echo "========================================="
    
    python sample.py \
        --checkpoint "$CHECKPOINT" \
        --mode "$MODE" \
        --num_per_class $NUM_PER_CLASS \
        --batch_size $BATCH_SIZE \
        --steps $TOTAL_STEPS \
        --cfg_scale $cfg_scale \
        --base_channels $BASE_CHANNELS \
        --filename "$FILENAME"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed CFG scale $cfg_scale"
    else
        echo "✗ Error with CFG scale $cfg_scale"
        exit 1
    fi
    
    echo ""
done

echo "========================================="
echo "  All generations completed!"
echo "========================================="