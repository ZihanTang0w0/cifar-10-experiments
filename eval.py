import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from scipy import linalg
import pickle
import argparse
import os
from tqdm import tqdm

# ==========================================
# 1. Inception V3 Wrapper (Standard FID)
# ==========================================
class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained InceptionV3
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity() # Remove classification layer
        self.inception.eval()
        
    def forward(self, x):
        # Resize from 32x32 to 299x299 (Standard FID requirement)
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Inception expects input normalized roughly to [-1, 1] or [0, 1] depending on implementation
        # The standard torch weights expect standardized inputs, but for FID
        # usually 0-1 range is passed and then normalized internally. 
        # Here we assume inputs are already [0, 1] tensors.
        
        # Get features (mixed_7c output is typically used, but fc identity is 2048 dim)
        # Using the standard pytorch implementation, .fc identity gives the pooled features.
        x = self.inception(x)
        return x

# ==========================================
# 2. FID Math Helper
# ==========================================
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.isclose(np.diagonal(covmean).imag, 0, atol=1e-3).all():
            return np.float64(-1) # Failure case
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

# ==========================================
# 3. Data Loading
# ==========================================
def load_cifar_batch_as_tensor(filepath):
    """
    Loads a CIFAR-style pickle and returns:
    - dict: {class_idx: tensor_images_of_that_class}
    - tensor: all_images
    """
    print(f"Loading {filepath}...")
    with open(filepath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        
    # Extract data
    data = d[b'data'] # (N, 3072)
    labels = np.array(d[b'labels'])
    
    # Reshape to (N, 3, 32, 32) and normalize to [0, 1]
    # CIFAR is stored uint8 0-255. Inception needs floats.
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) # HWC for now
    data = data.astype(np.float32) / 255.0
    data = torch.from_numpy(data).permute(0, 3, 1, 2) # Back to CHW (N, 3, 32, 32)
    
    # Split by class
    class_dict = {}
    for i in range(10):
        indices = np.where(labels == i)[0]
        if len(indices) > 0:
            class_dict[i] = data[indices]
            
    return class_dict, data

# ==========================================
# 4. Feature Extraction & Stats
# ==========================================
def get_statistics(images, model, batch_size=50, device='cuda'):
    """
    Computes mu, sigma for a tensor of images.
    """
    model.eval()
    activations = []
    
    # Process in batches
    n_samples = len(images)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            batch = images[start:end].to(device)
            
            # Forward
            feat = model(batch)
            activations.append(feat.cpu().numpy())
            
    activations = np.concatenate(activations, axis=0) # (N, 2048)
    
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    
    return mu, sigma

# ==========================================
# 5. Main Eval Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset1", type=str, required=True, help="Path to reference pickle (e.g., original test_batch)")
    parser.add_argument("--dataset2", type=str, required=True, help="Path to generated pickle")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # 1. Load Model
    print("Loading InceptionV3...")
    model = InceptionV3FeatureExtractor().to(args.device)
    
    # 2. Load Datasets
    # Returns dict {0: tensor, 1: tensor...} and full tensor
    d1_classes, d1_full = load_cifar_batch_as_tensor(args.dataset1)
    d2_classes, d2_full = load_cifar_batch_as_tensor(args.dataset2)
    
    # 3. Pre-compute Statistics (Optimization)
    # We calculate Mu/Sigma ONCE for every subset to avoid re-running the model
    print("\n--- Computing Statistics for Dataset 1 ---")
    stats1_by_class = {}
    for i in tqdm(range(10), desc="Classes D1"):
        if i in d1_classes:
            stats1_by_class[i] = get_statistics(d1_classes[i], model, args.batch_size, args.device)
    
    print("Computing stats for D1 Full...")
    stats1_full = get_statistics(d1_full, model, args.batch_size, args.device)

    print("\n--- Computing Statistics for Dataset 2 ---")
    stats2_by_class = {}
    for i in tqdm(range(10), desc="Classes D2"):
        if i in d2_classes:
            stats2_by_class[i] = get_statistics(d2_classes[i], model, args.batch_size, args.device)

    print("Computing stats for D2 Full...")
    stats2_full = get_statistics(d2_full, model, args.batch_size, args.device)

    # =========================================================
    # Task 1 & 2: Class-Wise and Cross-Class Matrix
    # =========================================================
    print("\n" + "="*50)
    print("FID MATRIX (Rows: Dataset1 Class, Cols: Dataset2 Class)")
    print("Diagonal elements = Class-wise FID (Same class comparison)")
    print("="*50)
    
    # Print Header
    header = "      " + "".join([f"D2_C{j:<6}" for j in range(10)])
    print(header)
    
    fid_matrix = np.zeros((10, 10))
    
    for i in range(10):
        row_str = f"D1_C{i} "
        for j in range(10):
            if i in stats1_by_class and j in stats2_by_class:
                mu1, sig1 = stats1_by_class[i]
                mu2, sig2 = stats2_by_class[j]
                
                fid = calculate_frechet_distance(mu1, sig1, mu2, sig2)
                fid_matrix[i, j] = fid
                row_str += f"{fid:>8.2f}"
            else:
                row_str += "    ----"
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
            fid = calculate_frechet_distance(mu1, sig1, mu2_all, sig2_all)
            print(f"Class {i} vs D2_All: {fid:.4f}")

    # =========================================================
    # Task 4: Unconditional vs Unconditional
    # =========================================================
    print("\n" + "="*50)
    print("Task 4: Full Dataset FID (Unconditional vs Unconditional)")
    print("="*50)
    
    mu1_all, sig1_all = stats1_full
    fid_full = calculate_frechet_distance(mu1_all, sig1_all, mu2_all, sig2_all)
    print(f"FID(D1_All, D2_All): {fid_full:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()