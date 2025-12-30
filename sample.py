import torch
import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm

# Imports from your project modules
from model import DiffusionUNet
from diffusion import DiffusionManager

def get_checkpoint_steps(total_steps):
    """
    Calculate which step counts to save datasets for.
    Returns multiples of 20 up to (but not including) total_steps, plus total_steps itself.
    
    Example: total_steps=90 -> [20, 40, 60, 80, 90]
    """
    checkpoints = []
    current = 20
    while current < total_steps:
        checkpoints.append(current)
        current += 20
    # Always include the final step count if not already included
    if not checkpoints or checkpoints[-1] != total_steps:
        checkpoints.append(total_steps)
    return checkpoints

def get_args():
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 Samples in Test Batch Format")
    
    # Checkpoint & Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt model checkpoint")
    parser.add_argument("--mode", type=str, default="diffusion", choices=["diffusion", "flow"], help="Sampling mode")
    parser.add_argument("--base_channels", type=int, default=128, help="Must match training setting")
    
    # Generation Settings
    parser.add_argument("--num_per_class", type=int, default=1000, help="Number of samples per class (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for generation")
    parser.add_argument("--steps", type=int, default=20, help="Sampling steps (DDIM/Euler)")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-Free Guidance scale")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./generated_data", help="Where to save the output file")
    parser.add_argument("--filename", type=str, default="generated_batch", help="Name of the pickled output file")
    
    return parser.parse_args()

def save_as_cifar_batch(images, labels, output_path):
    """
    Converts (B, 3, 32, 32) tensor images and integer labels into 
    the CIFAR-10 binary dictionary format.
    """
    print("Formatting data to CIFAR-10 specs...")
    
    # 1. Convert to Numpy Uint8 [0, 255]
    # Images are currently float [0, 1].
    images = (images * 255).astype(np.uint8)
    
    # 2. Reshape to (N, 3072) with R...G...B layout
    # Input shape: (N, 3, 32, 32)
    # PyTorch/Numpy flatten "C" order (last index changes fastest).
    # Since '3' (Channels) is the first dimension after Batch, flattening implies:
    # [Channel 0 (all pixels)] -> [Channel 1 (all pixels)] -> [Channel 2 (all pixels)]
    # This PERFECTLY matches CIFAR-10's requirement: 1024 R, then 1024 G, then 1024 B.
    
    num_images = images.shape[0]
    data_flat = images.reshape(num_images, -1) # Shape becomes (N, 3072)
    
    # 3. Create Dictionary
    # Note: 'labels' should be a list, 'data' a numpy array
    cifar_dict = {
        b'data': data_flat,
        b'labels': labels
    }
    
    # 4. Pickle Dump
    with open(output_path, 'wb') as f:
        pickle.dump(cifar_dict, f)
        
    print(f"Saved {num_images} samples to {output_path}")

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model (only once)
    print(f"Loading checkpoint from {args.checkpoint}...")
    model = DiffusionUNet(num_classes=10, base_channels=args.base_channels).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Handle cases where checkpoint includes optimizer state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # 2. Initialize Diffusion Engine (only once)
    diffusion = DiffusionManager(mode=args.mode, device=device)
    
    # 3. Calculate checkpoint steps to save at
    checkpoint_steps = get_checkpoint_steps(args.steps)
    print(f"\nWill generate datasets at steps: {checkpoint_steps}")
    print(f"Sampling {args.steps} total steps, saving {len(checkpoint_steps)} checkpoints\n")
    
    # 4. Setup output directories for all checkpoint steps
    output_dirs = {}
    for step_count in checkpoint_steps:
        output_dir = f"./generated_data/generated_{args.num_per_class}x10_mode_{args.mode}_steps_{step_count}_cfg_scale_{args.cfg_scale}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dirs[step_count] = output_dir
    
    # 5. Initialize storage for all checkpoints
    # Structure: {step_count: {'images': [], 'labels': []}}
    all_data = {step: {'images': [], 'labels': []} for step in checkpoint_steps}
    
    # 6. Generation Loop - Sample ONCE per batch, get multiple checkpoints
    print(f"Generating {args.num_per_class} images per class (Total: {args.num_per_class * 10})...")
    print(f"Each batch will be sampled once for {args.steps} steps\n")
    
    # Loop over classes 0-9
    for class_idx in range(10):
        samples_generated = 0
        pbar = tqdm(total=args.num_per_class, desc=f"Class {class_idx}")
        
        while samples_generated < args.num_per_class:
            # Calculate current batch size (don't over-generate)
            current_batch = min(args.batch_size, args.num_per_class - samples_generated)
            
            # Prepare labels
            labels = torch.full((current_batch,), class_idx, dtype=torch.long, device=device)
            
            # Sample ONCE and get checkpoints at multiple steps
            with torch.no_grad():
                # Returns dict {step_count: images_tensor}
                checkpoint_images = diffusion.sample(
                    model, 
                    n_samples=current_batch, 
                    labels=labels, 
                    steps=args.steps,  # Total steps
                    cfg_scale=args.cfg_scale,
                    checkpoint_steps=checkpoint_steps  # Which steps to save
                )
            
            # Store results for each checkpoint
            for step_count, imgs in checkpoint_images.items():
                all_data[step_count]['images'].append(imgs.numpy())
                all_data[step_count]['labels'].extend([class_idx] * current_batch)
            
            samples_generated += current_batch
            pbar.update(current_batch)
        
        pbar.close()

    # 7. Save all checkpoint datasets
    print(f"\n{'='*60}")
    print("Saving all checkpoint datasets...")
    print(f"{'='*60}\n")
    
    for step_count in checkpoint_steps:
        # Concatenate all batches
        final_images = np.concatenate(all_data[step_count]['images'], axis=0)
        final_labels = all_data[step_count]['labels']
        
        # Save in CIFAR Format
        save_path = os.path.join(output_dirs[step_count], args.filename)
        save_as_cifar_batch(final_images, final_labels, save_path)
        
    print(f"\n{'='*60}")
    print(f"All datasets generated successfully!")
    print(f"Sampled {args.steps} steps total, saved {len(checkpoint_steps)} datasets")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()