#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your trained model checkpoint (Change this to your actual best epoch)
CHECKPOINT="/home/tangzihan/cifar10-experiments/runs/cifar10_diffusion_cfg_0.13_bs_128_lr_0.0003/checkpoints/epoch_100.pt"

# Sampling settings
MODE="diffusion"          # 'diffusion' or 'diffusion'
STEPS=30             # Number of sampling steps
NUM_PER_CLASS=2500   # 5000 * 10 classes = 50,000 images (Matches CIFAR Train Set size)
BATCH_SIZE=100       # Adjust based on your GPU VRAM

# Paths
OUT_ROOT="./generated_data"
# Path to the ORIGINAL CIFAR-10 'test_batch' (for validation comparison)
# OR 'data_batch_1' etc combined if you want to compare against training set.
# Ideally, use the test set for fair evaluation.
REF_DATASET="../data/cifar-10-batches-py/test_batch" 

# Create output directory
mkdir -p $OUT_ROOT

# List of Guidance Scales to test
SCALES=(0 1 2 3 4 5)

echo "========================================================"
echo "STARTING EXPERIMENT: 6 Datasets (CFG: ${SCALES[*]})"
echo "Model: $CHECKPOINT"
echo "Mode: $MODE | Steps: $STEPS | Imgs/Class: $NUM_PER_CLASS"
echo "========================================================"

for S in "${SCALES[@]}"; do

    # Define specific filename for this run
    FILENAME="generated_2500x10_mode_diffusion_steps_30_cfg_scale_${S}.0/gen_cfg_${S}.pkl"

    # ==========================================
    # 2. EVALUATION LOOP
    # ==========================================
    echo ""
    echo "--------------------------------------------------------"
    echo "[Step 2/2] Evaluating CFG Scale = $S ..."
    echo "--------------------------------------------------------"
    
    GEN_DATASET="$OUT_ROOT/$FILENAME"
    LOG_FILE="eval_results/eval_log_2500x10_mode_diffusion_steps_30_cfg_scale_${S}.0.txt"

    # Run evaluation and save output to a log file
    python eval.py \
        --dataset1 "$GEN_DATASET" \
        --dataset2 "$REF_DATASET" \
        --device "cuda" \
        --batch_size 100 | tee "$LOG_FILE"
        
    echo "Evaluation complete. Results saved to $LOG_FILE"
done

echo ""
echo "========================================================"
echo "EXPERIMENT COMPLETED SUCCESSFULLY"
echo "All datasets and logs are in $OUT_ROOT"
echo "========================================================"