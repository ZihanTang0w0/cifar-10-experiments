import torch
import numpy as np
import pickle
import os
import argparse
import random
from tqdm import tqdm

# Imports from your project modules
from model import DiffusionUNet
from diffusion import DiffusionManager

def get_args():
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 Samples")
    
    # Checkpoint & Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt model checkpoint")
    parser.add_argument("--mode", type=str, default="diffusion", choices=["diffusion", "flow"], help="Sampling mode")
    parser.add_argument("--base_channels", type=int, default=128, help="Must match training setting")
    
    # Generation Settings
    parser.add_argument("--num_per_class", type=int, default=1000, help="Number of samples per class")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for generation")
    parser.add_argument("--steps", type=int, default=20, help="Sampling steps (DDIM/Euler)")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-Free Guidance scale")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for identical initial noise across runs")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./generated_data", help="Root directory for output")
    parser.add_argument("--filename", type=str, default="generated_batch", help="Name of the pickled output file")
    
    return parser.parse_args()

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Optional: Deterministic algorithms (may slow down slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_as_cifar_batch(images, labels, output_path):
    """Converts (B, 3, 32, 32) float tensors to CIFAR-10 binary dict format."""
    print(f"Saving to {output_path}...")
    
    # 1. Convert to Uint8 [0, 255]
    images = (images * 255).astype(np.uint8)
    
    # 2. Reshape to (N, 3072) with R...G...B layout
    num_images = images.shape[0]
    data_flat = images.reshape(num_images, -1)
    
    # 3. Create Dictionary
    cifar_dict = {
        b'data': data_flat,
        b'labels': labels
    }
    
    # 4. Pickle Dump
    with open(output_path, 'wb') as f:
        pickle.dump(cifar_dict, f)

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Set Seed (Crucial for sharing initial noise)
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # 2. Load Model
    print(f"Loading checkpoint from {args.checkpoint}...")
    model = DiffusionUNet(num_classes=10, base_channels=args.base_channels).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True) 
    # Note: weights_only=True is safe here for model weights, but False is needed for your bad data dicts
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # 3. Initialize Diffusion Engine
    diffusion = DiffusionManager(mode=args.mode, device=device)
    
    # 4. Setup Output Directory
    # We include all hyperparams in the folder name for clarity
    folder_name = f"generated_seed_{args.seed}_{args.num_per_class}x10_mode_{args.mode}_steps_{args.steps}_cfg_scale_{args.cfg_scale}"
    save_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 5. Storage
    all_images = []
    all_labels = []
    
    # 6. Generation Loop
    print(f"Generating {args.num_per_class} images per class (Steps: {args.steps})...")
    
    # Because we fixed the seed at the start, this loop will consume random numbers 
    # in the exact same order every time you run the script.
    for class_idx in range(10):
        samples_generated = 0
        pbar = tqdm(total=args.num_per_class, desc=f"Class {class_idx}")
        
        while samples_generated < args.num_per_class:
            current_batch_size = min(args.batch_size, args.num_per_class - samples_generated)
            
            # Prepare labels
            labels = torch.full((current_batch_size,), class_idx, dtype=torch.long, device=device)
            
            with torch.no_grad():
                # The FIRST thing sample() does is generate noise: torch.randn(...)
                # Since the seed is fixed, this noise will be identical across runs.
                batch_images = diffusion.sample(
                    model, 
                    n_samples=current_batch_size, 
                    labels=labels, 
                    steps=args.steps, 
                    cfg_scale=args.cfg_scale
                )
            
            all_images.append(batch_images.cpu().numpy())
            all_labels.extend([class_idx] * current_batch_size)
            
            samples_generated += current_batch_size
            pbar.update(current_batch_size)
        
        pbar.close()

    # 7. Save Final Dataset
    final_images = np.concatenate(all_images, axis=0)
    save_path = os.path.join(save_dir, args.filename)
    save_as_cifar_batch(final_images, all_labels, save_path)
    
    print(f"\nDone! Output saved to: {save_path}")

if __name__ == "__main__":
    main()