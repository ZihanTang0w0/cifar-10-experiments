import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
import pickle
import argparse
import os
import time
from tqdm import tqdm

# ==========================================
# 1. Inception V3 Wrapper
# ==========================================
class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity()
        self.inception.eval()
        
    def forward(self, x):
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.inception(x)
        return x

# ==========================================
# 2. PyTorch FID Math Helper (GPU)
# ==========================================
def calculate_frechet_distance_torch(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """PyTorch-native implementation of Frechet Distance."""
    # Ensure all tensors are on the same device
    device = mu1.device
    mu2 = mu2.to(device)
    sigma1 = sigma1.to(device)
    sigma2 = sigma2.to(device)

    diff = mu1 - mu2
    mean_term = diff @ diff

    eigvals_1, eigvecs_1 = torch.linalg.eigh(sigma1)
    eigvals_1 = torch.clamp(eigvals_1, min=eps)
    sqrt_sigma1 = eigvecs_1 @ torch.diag(torch.sqrt(eigvals_1)) @ eigvecs_1.T

    sigma_prod_sym = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    
    eigvals_prod = torch.linalg.eigvalsh(sigma_prod_sym)
    eigvals_prod = torch.clamp(eigvals_prod, min=eps)
    
    tr_covmean = torch.sum(torch.sqrt(eigvals_prod))

    return mean_term + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean

# ==========================================
# 3. Data Loading & Stats Management
# ==========================================
def load_cifar_batch_as_tensor(filepath):
    """Loads raw CIFAR images from pickle."""
    print(f"Loading raw images from {filepath}...")
    with open(filepath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    data = d[b'data']
    labels = np.array(d[b'labels'])
    
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = data.astype(np.float32) / 255.0
    data = torch.from_numpy(data).permute(0, 3, 1, 2)
    
    class_dict = {}
    for i in range(10):
        indices = np.where(labels == i)[0]
        if len(indices) > 0:
            class_dict[i] = data[indices]
            
    return class_dict, data

def save_statistics(filepath, stats_full, stats_by_class):
    """Saves computed stats to .npz file."""
    flat_dict = {}
    # Save Full
    flat_dict['full_mu'] = stats_full[0].cpu().numpy()
    flat_dict['full_sigma'] = stats_full[1].cpu().numpy()
    # Save Classes
    for k, (m, s) in stats_by_class.items():
        flat_dict[f'class_{k}_mu'] = m.cpu().numpy()
        flat_dict[f'class_{k}_sigma'] = s.cpu().numpy()
    
    np.savez(filepath, **flat_dict)
    print(f"Saved statistics to {filepath}")

def load_statistics(filepath, device):
    """Loads stats from .npz file directly to GPU tensors."""
    print(f"Loading pre-computed stats from {filepath}...")
    data = np.load(filepath)
    
    def to_torch(arr):
        return torch.from_numpy(arr).to(device)

    stats_full = (to_torch(data['full_mu']), to_torch(data['full_sigma']))
    
    stats_by_class = {}
    for key in data.files:
        if key.startswith('class_') and key.endswith('_mu'):
            cls_idx = int(key.split('_')[1])
            mu = to_torch(data[f'class_{cls_idx}_mu'])
            sigma = to_torch(data[f'class_{cls_idx}_sigma'])
            stats_by_class[cls_idx] = (mu, sigma)
            
    return stats_by_class, stats_full

def get_statistics_from_images(images, model, batch_size=50, device='cuda', desc='Processing'):
    """Runs InceptionV3 on images to get stats."""
    model.eval()
    activations = []
    n_samples = len(images)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc=desc, leave=False):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            batch = images[start:end].to(device)
            feat = model(batch)
            activations.append(feat) 
            
    activations = torch.cat(activations, dim=0)
    mu = torch.mean(activations, dim=0)
    sigma = torch.cov(activations.T)
    return mu, sigma

# ==========================================
# 4. Helper: Processing Logic
# ==========================================
def process_dataset(filepath, model, batch_size, device):
    """Decides whether to calculate or load stats based on extension."""
    if filepath.endswith('.npz'):
        # Load pre-computed
        return load_statistics(filepath, device)
    else:
        # Calculate from scratch
        if model is None:
            raise ValueError("Model is required to process raw images but was not loaded!")
            
        class_dict, full_data = load_cifar_batch_as_tensor(filepath)
        
        # Stats By Class
        stats_by_class = {}
        for i in range(10):
            if i in class_dict:
                stats_by_class[i] = get_statistics_from_images(
                    class_dict[i], model, batch_size, device, desc=f"Class {i}"
                )
        # Stats Full
        stats_full = get_statistics_from_images(
            full_data, model, batch_size, device, desc="Full Dataset"
        )
        return stats_by_class, stats_full

# ==========================================
# 5. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    # Mode switch
    parser.add_argument("--compute_stats_only", action="store_true", help="Only compute and save stats for dataset1")
    parser.add_argument("--save_to", type=str, default=None, help="Path to save .npz file (used with --compute_stats_only)")
    
    # Standard Eval Args
    parser.add_argument("--dataset1", type=str, required=True, help="Reference or Gen")
    parser.add_argument("--dataset2", type=str, default=None, help="Required unless compute_stats_only is set")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cross_class", action="store_true")
    args = parser.parse_args()
    
    # 1. Lazy Load Model
    # We only need the model if at least one input is NOT a .npz file
    needs_model = not (str(args.dataset1).endswith('.npz'))
    if args.dataset2 and not str(args.dataset2).endswith('.npz'):
        needs_model = True
        
    model = None
    if needs_model:
        print("Loading InceptionV3 (Model needed for raw images)...")
        model = InceptionV3FeatureExtractor().to(args.device)
    else:
        print("Skipping Model Load (Using pre-computed stats)...")

    # 2. Mode: Compute Stats Only
    if args.compute_stats_only:
        if not args.save_to:
            print("Error: --save_to is required for compute_stats_only mode")
            return
        
        print(f"--- Pre-computing statistics for {args.dataset1} ---")
        stats_class, stats_full = process_dataset(args.dataset1, model, args.batch_size, args.device)
        save_statistics(args.save_to, stats_full, stats_class)
        return

    # 3. Mode: Evaluation
    if not args.dataset2:
        print("Error: --dataset2 is required for evaluation mode")
        return

    # Load D1
    print(f"\n--- Processing Dataset 1: {args.dataset1} ---")
    stats1_by_class, stats1_full = process_dataset(args.dataset1, model, args.batch_size, args.device)

    # Load D2
    print(f"\n--- Processing Dataset 2: {args.dataset2} ---")
    stats2_by_class, stats2_full = process_dataset(args.dataset2, model, args.batch_size, args.device)

    # =========================================================
    # Task 1 & 2: D1 vs D2 Matrix
    # =========================================================
    print("\n" + "="*50)
    print("TASK 1 & 2: Dataset 1 (Rows) vs Dataset 2 (Cols)")
    print("="*50)
    print("      " + "".join([f"D2_C{j:<6}" for j in range(10)]))
    
    for i in range(10):
        row_str = f"D1_C{i} "
        for j in range(10):
            if args.cross_class or i == j:
                if i in stats1_by_class and j in stats2_by_class:
                    mu1, sig1 = stats1_by_class[i]
                    mu2, sig2 = stats2_by_class[j]
                    fid = calculate_frechet_distance_torch(mu1, sig1, mu2, sig2)
                    row_str += f"{fid.item():>8.2f}"
                else:
                    row_str += "    ----"
            else:
                row_str += "     ..."
        print(row_str)

    # =========================================================
    # Task 3: Class-Specific vs Unconditional (D1_Class vs D2_All)
    # =========================================================
    print("\n" + "="*50)
    print("Task 3: Class-Specific vs Unconditional (FID(D1_Class_i, D2_All))")
    print("Measures how well a specific class subset resembles the entire distribution of the other set")
    print("="*50)
    
    mu2_all, sig2_all = stats2_full
    for i in range(10):
        if i in stats1_by_class:
            mu1, sig1 = stats1_by_class[i]
            fid = calculate_frechet_distance_torch(mu1, sig1, mu2_all, sig2_all)
            print(f"Class {i} vs D2_All: {fid.item():.4f}")

    # =========================================================
    # Task 3b: Unconditional vs Class-Specific (D1_All vs D2_Class)
    # =========================================================
    print("\n" + "="*50)
    print("Task 3b: Unconditional vs Class-Specific (FID(D1_All, D2_Class_i))")
    print("Measures how well each D2 class resembles the entire distribution of D1")
    print("="*50)
    
    mu1_all, sig1_all = stats1_full
    for i in range(10):
        if i in stats2_by_class:
            mu2, sig2 = stats2_by_class[i]
            fid = calculate_frechet_distance_torch(mu1_all, sig1_all, mu2, sig2)
            print(f"D1_All vs Class {i}: {fid.item():.4f}")

    # =========================================================
    # Task 4: Unconditional vs Unconditional
    # =========================================================
    print("\n" + "="*50)
    print("Task 4: Full Dataset FID (Unconditional vs Unconditional)")
    print("="*50)
    
    mu1_all, sig1_all = stats1_full
    fid_full = calculate_frechet_distance_torch(mu1_all, sig1_all, mu2_all, sig2_all)
    print(f"FID(D1_All, D2_All): {fid_full.item():.4f}")
    print("="*50)

    # =========================================================
    # Task 5: Internal Generated Diversity (D2 Class vs D2 Class)
    # =========================================================
    print("\n" + "="*50)
    print("Task 5: Internal Separability (D2 Class i vs D2 Class j)")
    print("Measures if Dataset2 Class 0 looks distinct from Dataset2 Class 1")
    print("Diagonal is always 0. High off-diagonal = Good separation.")
    print("="*50)
    
    print("      " + "".join([f"D2_C{j:<6}" for j in range(10)]))
    
    for i in range(10):
        row_str = f"D2_C{i} "
        for j in range(10):
            if i in stats2_by_class and j in stats2_by_class:
                if i == j:
                    row_str += f"{0.0:>8.1f}" # Identity
                else:
                    mu_i, sig_i = stats2_by_class[i]
                    mu_j, sig_j = stats2_by_class[j]
                    fid = calculate_frechet_distance_torch(mu_i, sig_i, mu_j, sig_j)
                    row_str += f"{fid.item():>8.1f}"
            else:
                row_str += "    ----"
        print(row_str)

    # =========================================================
    # Task 6: Intra-Class Diversity (Trace of Covariance)
    # =========================================================
    print("\n" + "="*65)
    print("Task 6: Intra-Class Diversity (Total Variance)")
    print("Measures the 'width' of the distribution in feature space.")
    print("Low Gen Trace (< 1.0)  = Mode Collapse (Zero Diversity)")
    print("High Gen Trace (~Real) = Good Diversity")
    print("="*65)
    
    print(f"{'Class':<6} | {'Real Trace':<12} | {'Gen Trace':<12} | {'Ratio (Gen/Real)':<15}")
    print("-" * 55)
    
    for i in range(10):
        # We need both stats to be present
        if i in stats1_by_class and i in stats2_by_class:
            _, sig1 = stats1_by_class[i] # Real Sigma
            _, sig2 = stats2_by_class[i] # Gen Sigma
            
            # Calculate Trace (Sum of variances)
            tr1 = torch.trace(sig1).item()
            tr2 = torch.trace(sig2).item()
            
            # Ratio of 1.0 means perfect diversity match
            ratio = tr2 / (tr1 + 1e-8)
            
            print(f"{i:<6} | {tr1:>12.2f} | {tr2:>12.2f} | {ratio:>15.2f}")

    print("="*65)
if __name__ == "__main__":
    main()