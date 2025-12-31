import os
import glob
import re
import argparse
import pandas as pd
import numpy as np
import torch
import time
from tqdm import tqdm

# Import helper functions from your existing eval.py
# Ensure eval.py is in the same directory
try:
    from eval import (
        InceptionV3FeatureExtractor,
        calculate_frechet_distance_torch,
        process_dataset,
        load_statistics
    )
except ImportError:
    print("Error: Could not import from eval.py. Please ensure it is in the same directory.")
    exit(1)

# ==========================================
# 1. Helper: Metadata Parsing
# ==========================================
def parse_folder_name(folder_name):
    """
    Extracts hyperparameters from the folder name.
    Expected format: generated_seed_1007_1000x10_mode_diffusion_steps_80_cfg_scale_3.1
    """
    # Regex to capture relevant parts
    pattern = r"seed_(\d+)_.*mode_([a-zA-Z]+)_steps_(\d+)_cfg_scale_([\d\.]+)"
    match = re.search(pattern, folder_name)
    
    if match:
        return {
            "seed": int(match.group(1)),
            "mode": match.group(2),
            "steps": int(match.group(3)),
            "cfg_scale": float(match.group(4))
        }
    else:
        print(f"Warning: Could not parse metadata from {folder_name}")
        return None

# ==========================================
# 2. Helper: Metric Computations
# ==========================================
def compute_metrics(stats1_class, stats1_full, stats2_class, stats2_full):
    """Computes all 4 key metrics for a single dataset pair."""
    metrics = {}
    
    # --- A. Global FID ---
    mu1, sig1 = stats1_full
    mu2, sig2 = stats2_full
    metrics['global_fid'] = calculate_frechet_distance_torch(mu1, sig1, mu2, sig2).item()

    # --- B. Class-wise FID (Quality) ---
    # Diagonal of Task 1&2: Real Class i vs Gen Class i
    class_fids = []
    for i in range(10):
        if i in stats1_class and i in stats2_class:
            m1, s1 = stats1_class[i]
            m2, s2 = stats2_class[i]
            class_fids.append(calculate_frechet_distance_torch(m1, s1, m2, s2).item())
    metrics['avg_class_fid'] = np.mean(class_fids) if class_fids else np.nan

    # --- C. Internal Separability (Cross-Class) ---
    # Off-diagonal of Task 5: Gen Class i vs Gen Class j
    # This measures how distinct the generated classes are from each other
    cross_fids = []
    for i in range(10):
        for j in range(10):
            if i != j and i in stats2_class and j in stats2_class:
                m_i, s_i = stats2_class[i]
                m_j, s_j = stats2_class[j]
                cross_fids.append(calculate_frechet_distance_torch(m_i, s_i, m_j, s_j).item())
    metrics['avg_separability'] = np.mean(cross_fids) if cross_fids else np.nan

    # --- D. Intra-Class Diversity (Trace Ratio) ---
    # Task 6: Ratio of Gen Variance to Real Variance
    ratios = []
    for i in range(10):
        if i in stats1_class and i in stats2_class:
            _, s1 = stats1_class[i] # Real Sigma
            _, s2 = stats2_class[i] # Gen Sigma
            tr1 = torch.trace(s1).item()
            tr2 = torch.trace(s2).item()
            ratios.append(tr2 / (tr1 + 1e-8))
    metrics['avg_diversity_ratio'] = np.mean(ratios) if ratios else np.nan

    return metrics

def generate_log_text(folder_name, metrics):
    """Creates a formatted string similar to the original log."""
    log = f"Evaluation Results for: {folder_name}\n"
    log += "=" * 60 + "\n"
    log += f"Global FID:         {metrics['global_fid']:.4f}\n"
    log += f"Avg Class FID:      {metrics['avg_class_fid']:.4f} (Quality - Lower is better)\n"
    log += f"Avg Separability:   {metrics['avg_separability']:.4f} (distinctness - Higher is better)\n"
    log += f"Avg Diversity Ratio:{metrics['avg_diversity_ratio']:.4f} (1.0 is ideal)\n"
    log += "=" * 60 + "\n"
    return log

# ==========================================
# 3. Main Script
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation for Generative Models")
    parser.add_argument("--gen_dir", type=str, default="./generated_data", help="Root folder containing generated subfolders")
    parser.add_argument("--ref_stats", type=str, default="./cached_ref_stats2.npz", help="Path to pre-computed real stats")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./eval_results_batch", help="Where to save results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_save_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_save_dir, exist_ok=True)

    # 1. Load Inception Model (Once)
    print("Loading InceptionV3...")
    model = InceptionV3FeatureExtractor().to(args.device)

    # 2. Load Reference Stats (Once)
    print(f"Loading Reference Stats from {args.ref_stats}...")
    if not os.path.exists(args.ref_stats):
        print("Error: Reference stats file not found. Please run eval.py --compute_stats_only first.")
        return
    stats1_class, stats1_full = load_statistics(args.ref_stats, args.device)

    # 3. Compute GT Baseline Separability (Real vs Real Cross-Class)
    # Useful for your "Graph 3"
    print("Computing GT Baseline Separability...")
    gt_cross_fids = []
    for i in range(10):
        for j in range(10):
            if i != j and i in stats1_class and j in stats1_class:
                m_i, s_i = stats1_class[i]
                m_j, s_j = stats1_class[j]
                gt_cross_fids.append(calculate_frechet_distance_torch(m_i, s_i, m_j, s_j).item())
    gt_separability = np.mean(gt_cross_fids)
    print(f"GT Baseline Separability (Real vs Real): {gt_separability:.4f}")

    # 4. Find Generated Folders
    # We look for folders containing "generated_batch" files or just the folders themselves
    subdirs = sorted(glob.glob(os.path.join(args.gen_dir, "*")))
    # Filter only directories
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    
    print(f"Found {len(subdirs)} generated datasets to evaluate.")

    results_list = []

    # 5. Main Loop
    for folder_path in tqdm(subdirs, desc="Evaluating Datasets"):
        folder_name = os.path.basename(folder_path)
        
        # A. Parse Metadata
        meta = parse_folder_name(folder_name)
        if not meta:
            continue # Skip folders that don't match naming convention
            
        # B. Check for data file
        # We assume standard naming "generated_batch" inside
        batch_file = os.path.join(folder_path, "generated_batch")
        if not os.path.exists(batch_file):
            print(f"Skipping {folder_name}: 'generated_batch' not found.")
            continue

        # C. Compute Gen Stats
        # We use process_dataset from eval.py to handle loading + inception
        # Note: We pass None for model if using .npz, but here we pass the loaded model
        try:
            stats2_class, stats2_full = process_dataset(batch_file, model, args.batch_size, args.device)
        except Exception as e:
            print(f"Failed to process {folder_name}: {e}")
            continue

        # D. Calculate Metrics
        metrics = compute_metrics(stats1_class, stats1_full, stats2_class, stats2_full)
        
        # E. Save Log File
        log_text = generate_log_text(folder_name, metrics)
        with open(os.path.join(log_save_dir, f"{folder_name}.txt"), "w") as f:
            f.write(log_text)

        # F. Add to Results List
        entry = {
            "folder": folder_name,
            **meta, # seed, mode, steps, cfg
            **metrics, # global_fid, avg_class_fid, avg_separability, avg_diversity
            "gt_separability": gt_separability # Save baseline for easy plotting later
        }
        results_list.append(entry)

    # 6. Save Final DataFrame
    if results_list:
        df = pd.DataFrame(results_list)
        
        csv_path = os.path.join(args.output_dir, "results.csv")
        pkl_path = os.path.join(args.output_dir, "results.pkl")
        
        df.to_csv(csv_path, index=False)
        df.to_pickle(pkl_path)
        
        print(f"\nBatch Evaluation Complete!")
        print(f"Results saved to:\n  - {csv_path}\n  - {pkl_path}")
        print(f"Logs saved to: {log_save_dir}")
        print("\nSnapshot of Results:")
        print(df.head())
    else:
        print("No valid datasets evaluated.")

if __name__ == "__main__":
    main()