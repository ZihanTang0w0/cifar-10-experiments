#!/bin/bash

# ============================================================================
# Script: Optimized Evaluation with Cached Statistics
# ============================================================================

# Configuration
REFERENCE_DATASET="../data/cifar-10-batches-py/test_batch"
GENERATED_DATA_DIR="./generated_data"
OUTPUT_DIR="./eval_results"
CACHE_FILE="./cached_ref_stats2.npz" # Where to save the pre-computed stats
BATCH_SIZE="${BATCH_SIZE:-100}"
DEVICE="${DEVICE:-cuda}"
NUM_THREADS="${NUM_THREADS:-24}"

export OMP_NUM_THREADS=$NUM_THREADS
export MKL_NUM_THREADS=$NUM_THREADS
export OPENBLAS_NUM_THREADS=$NUM_THREADS
export VECLIB_MAXIMUM_THREADS=$NUM_THREADS
export NUMEXPR_NUM_THREADS=$NUM_THREADS
# 1. Setup
mkdir -p "$OUTPUT_DIR"

# 2. Check/Create Cache for Reference Dataset
if [ -f "$CACHE_FILE" ]; then
    echo "Using existing cache: $CACHE_FILE"
else
    echo "--------------------------------------------------------"
    echo "Creating Statistics Cache for Reference Dataset..."
    echo "--------------------------------------------------------"
    # Runs eval.py in "compute_stats_only" mode
    python eval.py \
        --compute_stats_only \
        --dataset1 "$REFERENCE_DATASET" \
        --save_to "$CACHE_FILE" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE"
        
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create cache."
        exit 1
    fi
    echo "Cache created successfully!"
fi

# 3. Find Generated Datasets
mapfile -t ALL_DATASETS < <(find "$GENERATED_DATA_DIR" -name "generated_batch" -type f | sort)
DATASETS=()
for dataset in "${ALL_DATASETS[@]}"; do
    if [[ "$dataset" == *mode_diffusion* && "$dataset" == *seed* ]]; then
        DATASETS+=("$dataset")
    fi
done

echo ""
echo "Found ${#DATASETS[@]} diffusion datasets to evaluate."
echo "--------------------------------------------------------"

# 4. Evaluation Loop
for dataset_path in "${DATASETS[@]}"; do
    dir_name=$(basename "$(dirname "$dataset_path")")
    log_file="$OUTPUT_DIR/eval_log_${dir_name}.txt"
    
    echo "Evaluating: $dir_name"
    
    # NOTE: We pass the CACHE_FILE as dataset1 (Reference) 
    # and the generated pickle as dataset2 (Generated).
    python eval.py \
        --dataset1 "$CACHE_FILE" \
        --dataset2 "$dataset_path" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --cross_class \
        | tee "$log_file"
        
    echo "--------------------------------------------------------"
done