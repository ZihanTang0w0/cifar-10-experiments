#!/bin/bash

# ============================================================================
# Script: Generate CIFAR-10 datasets with varying CFG scales & Steps
# Usage: ./run_generation_cli.sh <mode> <checkpoint> <cfg_start> <cfg_end> <cfg_stride>
# Example: ./run_generation_cli.sh diffusion checkpoints/model.pt 1.0 4.0 0.5
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
BASE_CHANNELS=128
BATCH_SIZE=100
FILENAME="generated_batch"

# Define the step counts to evaluate
STEPS_LIST=(20 40 60 80 100)

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

# ==========================================
# 1. SEED STRATEGY SETUP
# ==========================================
# We use a base seed plus offsets to guarantee uniqueness across params.
GLOBAL_BASE_SEED=1000

# Mode Offset: Ensures 'diffusion' runs never share seeds with 'flow' runs
if [ "$MODE" == "diffusion" ]; then
    MODE_OFFSET=0
else
    # Large offset (e.g. 100,000) so they don't overlap
    MODE_OFFSET=100000 
fi

echo "========================================="
echo "  CIFAR-10 Dataset Generation Script"
echo "========================================="
echo "Mode:           $MODE (Seed Offset: $MODE_OFFSET)"
echo "Checkpoint:     $CHECKPOINT"
echo "Steps to run:   ${STEPS_LIST[*]}"
echo "CFG range:      $CFG_START to $CFG_END (stride: $CFG_STRIDE)"
echo "Base Seed:      $GLOBAL_BASE_SEED"
echo "========================================="
echo ""

# ==========================================
# 2. GENERATE CFG VALUES
# ==========================================
cfg_values=$(awk -v start="$CFG_START" -v end="$CFG_END" -v stride="$CFG_STRIDE" '
BEGIN {
    for (i = start; i <= end; i += stride) {
        printf "%.1f ", i
    }
}')
cfg_array=($cfg_values)

echo "Will generate datasets for ${#cfg_array[@]} CFG scales."
echo ""

# Initialize a counter to increment seed for each CFG scale
cfg_index=0

# ==========================================
# 3. MAIN LOOPS
# ==========================================

# Outer Loop: CFG Scale
for cfg_scale in "${cfg_array[@]}"; do
    
    # CALCULATE SEED FOR THIS GROUP
    # Final Seed = 1000 + (0 or 100000) + CFG_Index
    # This ensures:
    # 1. Different Modes have totally different ranges
    # 2. Different CFG scales have different seeds
    # 3. Same CFG (inner loop) keeps the SAME seed
    CURRENT_SEED=$((GLOBAL_BASE_SEED + MODE_OFFSET + cfg_index))
    
    echo ">>> Starting CFG Group: $cfg_scale (Seed: $CURRENT_SEED)"

    # Inner Loop: Steps (Run independent process for each step count)
    for steps in "${STEPS_LIST[@]}"; do
        echo "   -> Running: Steps $steps | Seed $CURRENT_SEED"
        
        # Call Python script (Must utilize the sample.py with --seed support!)
        python sample.py \
            --checkpoint "$CHECKPOINT" \
            --mode "$MODE" \
            --num_per_class $NUM_PER_CLASS \
            --batch_size $BATCH_SIZE \
            --steps $steps \
            --cfg_scale $cfg_scale \
            --seed $CURRENT_SEED \
            --base_channels $BASE_CHANNELS \
            --filename "$FILENAME"

        if [ $? -ne 0 ]; then
            echo "✗ Error occurred at CFG $cfg_scale, Steps $steps"
            exit 1
        fi
    done
    
    # Increment index for the next CFG value to get a new seed
    cfg_index=$((cfg_index + 1))
    echo ""
done

echo "========================================="
echo "  All generations completed!"
echo "========================================="