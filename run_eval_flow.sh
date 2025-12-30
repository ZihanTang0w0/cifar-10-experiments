#!/bin/bash

# ============================================================================
# Script: Evaluate FLOW models with TERMINAL OUTPUT
# ============================================================================

# Configuration
REFERENCE_DATASET="../data/cifar-10-batches-py/test_batch"
GENERATED_DATA_DIR="./generated_data"
OUTPUT_DIR="./eval_results"
BATCH_SIZE="${BATCH_SIZE:-100}"
DEVICE="${DEVICE:-cuda}"
CROSS_CLASS="${CROSS_CLASS:-false}"

# CPU Thread Limiting
NUM_THREADS="${NUM_THREADS:-24}"

export OMP_NUM_THREADS=$NUM_THREADS
export MKL_NUM_THREADS=$NUM_THREADS
export OPENBLAS_NUM_THREADS=$NUM_THREADS
export VECLIB_MAXIMUM_THREADS=$NUM_THREADS
export NUMEXPR_NUM_THREADS=$NUM_THREADS

echo "========================================="
echo "  FID Evaluation - FLOW (Interactive)"
echo "========================================="
echo "Reference dataset: $REFERENCE_DATASET"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "CPU threads: $NUM_THREADS"
echo "========================================="
echo ""

# Check reference
if [ ! -f "$REFERENCE_DATASET" ]; then
    echo "Error: Reference dataset not found at: $REFERENCE_DATASET"
    exit 1
fi

# Find datasets
mapfile -t ALL_DATASETS < <(find "$GENERATED_DATA_DIR" -name "generated_batch" -type f | sort)

# Filter for flow
DATASETS=()
for dataset in "${ALL_DATASETS[@]}"; do
    dir_name=$(basename "$(dirname "$dataset")")
    if [[ "$dir_name" == *"mode_flow"* ]]; then
        DATASETS+=("$dataset")
    fi
done

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "Error: No FLOW datasets found"
    exit 1
fi

echo "Found ${#DATASETS[@]} FLOW dataset(s)"
echo ""

mkdir -p "$OUTPUT_DIR"

# Base command
BASE_CMD="python eval.py --dataset2 \"$REFERENCE_DATASET\" --batch_size $BATCH_SIZE --device $DEVICE"
if [ "$CROSS_CLASS" = "true" ]; then
    BASE_CMD="$BASE_CMD --cross_class"
fi

success_count=0
fail_count=0

for dataset_path in "${DATASETS[@]}"; do
    dir_name=$(basename "$(dirname "$dataset_path")")
    log_name="${dir_name#generated_}"
    log_file="$OUTPUT_DIR/eval_log_${log_name}.txt"
    
    echo "========================================="
    echo "[$((success_count + fail_count + 1))/${#DATASETS[@]}] $dir_name"
    echo "========================================="
    
    # Run with TEE - output to BOTH terminal AND log file
    (
        echo "Evaluation of: $dir_name"
        echo "Dataset: $dataset_path"
        echo "Started: $(date)"
        echo ""
        
        eval "$BASE_CMD --dataset1 \"$dataset_path\""
        eval_exit=$?
        
        echo ""
        echo "Completed: $(date)"
        echo "Exit code: $eval_exit"
        
        exit $eval_exit  # Propagate the exit code
    ) 2>&1 | tee "$log_file"
    
    subshell_exit=${PIPESTATUS[0]}
    if [ $subshell_exit -eq 0 ]; then
        echo "✓ Success"
        ((success_count++))
    else
        echo "✗ Failed"
        ((fail_count++))
    fi
    echo ""
done

echo "========================================="
echo "  COMPLETE"
echo "========================================="
echo "Total: ${#DATASETS[@]}"
echo "Success: $success_count"
echo "Failed: $fail_count"
echo "========================================="