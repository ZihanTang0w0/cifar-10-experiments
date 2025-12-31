import torch
import torchvision
from torchvision.utils import save_image, make_grid
import numpy as np
import argparse
import os
import copy
from tqdm import tqdm

# Imports from your project
from model import DiffusionUNet
from diffusion import DiffusionManager

# ==========================================
# Configuration
# ==========================================
# The 16 fixed seeds to be used across ALL modes and scales
# We use a fixed list to ensure absolute reproducibility
FIXED_SEEDS = [42, 100, 2023, 7, 1234, 9999, 56, 888, 
               101, 555, 777, 303, 909, 111, 222, 333]

def get_args():
    parser = argparse.ArgumentParser(description="Generate Consistent CFG Grids for Flow and Diffusion")
    
    # Checkpoints (Need both!)
    parser.add_argument("--flow_checkpoint", type=str, default="/home/tangzihan/cifar10-experiments/runs/cifar10_flow_cfg_0.13_bs_128_lr_0.0003/checkpoints/epoch_100.pt", help="Path to Flow model .pt file")
    parser.add_argument("--diffusion_checkpoint", type=str, default="/home/tangzihan/cifar10-experiments/runs/cifar10_diffusion_cfg_0.13_bs_128_lr_0.0003/checkpoints/epoch_100.pt", help="Path to Diffusion model .pt file")
    
    # Settings
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--output_dir", type=str, default="./final_grids", help="Output root directory")
    
    return parser.parse_args()

def generate_for_mode(model, mode, checkpoint_path, scales, args, device):
    """
    Handles the generation loop for a specific mode (Flow/Diffusion).
    """
    print(f"\n{'='*60}")
    print(f"Starting generation for mode: {mode.upper()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Scales: {scales}")
    print(f"{'='*60}")

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 2. Initialize Manager
    # This sets the correct math (ODE vs SDE)
    manager = DiffusionManager(mode=mode, device=device)

    # 3. Create Output Folder
    mode_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(mode_dir):
        os.makedirs(mode_dir)

    # 4. Nested Loops: Class -> CFG Scale -> 16 Seeds
    classes = range(10)
    
    # We loop over classes first so we can group files nicely
    for class_idx in classes:
        print(f"Processing Class {class_idx}...")
        
        for scale in scales:
            # We will collect 16 images for this specific (Class, Scale) combo
            grid_images = []
            
            # Iterate through our fixed seeds
            # We sample 1 image at a time to enforce the seed strictly
            for seed in FIXED_SEEDS:
                # SET SEED: This ensures the initial noise (x_T) is identical
                # regardless of mode, scale, or class.
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                
                label_tensor = torch.tensor([class_idx], device=device)
                
                with torch.no_grad():
                    # Note: We disable the progress bar inside sample() to avoid spam
                    # You might need to modify diffusion.py to accept 'disable_tqdm=True'
                    # or just live with the bars.
                    img = manager.sample(
                        model, 
                        n_samples=1, 
                        labels=label_tensor, 
                        steps=args.steps, 
                        cfg_scale=scale
                    )
                    grid_images.append(img.cpu())

            # Stack 16 images -> (16, 3, 32, 32)
            batch_tensor = torch.cat(grid_images, dim=0)
            
            # Make 4x4 Grid
            grid = make_grid(batch_tensor, nrow=4, padding=2, pad_value=1.0)
            
            # Filename: class_0_scale_1.0.png
            fname = f"class_{class_idx}_cfg_{scale:.1f}.png"
            save_path = os.path.join(mode_dir, fname)
            save_image(grid, save_path)

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize Model Architecture (Shared)
    print(f"Initializing Model (Base Channels: {args.base_channels})...")
    model = DiffusionUNet(num_classes=10, base_channels=args.base_channels).to(device)

    # 2. Define Scale Ranges
    # Flow: 1.0 to 3.2, stride 0.2 -> [1.0, 1.2, ..., 3.2]
    # np.arange excludes the stop value, so we add epsilon
    flow_scales = np.arange(1.0, 3.2 + 0.01, 2.2)
    
    # Diffusion: 1.0 to 4.0, stride 0.3 -> [1.0, 1.3, ..., 4.0]
    diff_scales = np.arange(1.0, 4.0 + 0.01, 3.0)

    # 3. Run Flow Generation
    generate_for_mode(
        model, 
        mode='flow', 
        checkpoint_path=args.flow_checkpoint, 
        scales=flow_scales, 
        args=args, 
        device=device
    )

    # 4. Run Diffusion Generation
    generate_for_mode(
        model, 
        mode='diffusion', 
        checkpoint_path=args.diffusion_checkpoint, 
        scales=diff_scales, 
        args=args, 
        device=device
    )

    print(f"\nDone! All grids saved to {args.output_dir}")

if __name__ == "__main__":
    main()