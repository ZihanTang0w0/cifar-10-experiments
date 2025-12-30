import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# --- Import your modules ---
from model import DiffusionUNet
from diffusion import DiffusionManager
from dataset import get_dataloader

def get_args():
    parser = argparse.ArgumentParser(description="Train Diffusion or Flow Model on CIFAR-10")
    
    # Mode Selection
    parser.add_argument("--mode", type=str, default="diffusion", choices=["diffusion", "flow"], 
                        help="Training mode: 'diffusion' (DDPM) or 'flow' (Rectified Flow)")
    
    # Training Hyperparameters
    parser.add_argument("--run_name", type=str, default="cifar10_experiment", help="Name for saving checkpoints/samples")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--cfg_prob", type=float, default=0.1, help="Probability of dropping class labels for CFG")
    
    # Model Architecture
    parser.add_argument("--base_channels", type=int, default=128, help="Base channel width of U-Net")
    
    # System
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    # Checkpointing
    parser.add_argument("--save_interval", type=int, default=20, help="Save checkpoint/samples every X epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume from")

    return parser.parse_args()

def save_training_config(args, save_dir):
    """Save all training parameters to a JSON file"""
    config_path = os.path.join(save_dir, "training_config.json")
    
    # Convert args to dict and add timestamp
    config = vars(args).copy()
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config['torch_version'] = torch.__version__
    config['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        config['cuda_version'] = torch.version.cuda
        config['gpu_name'] = torch.cuda.get_device_name(0)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Training config saved to: {config_path}")

def save_checkpoint(model, optimizer, epoch, save_dir):
    """Save checkpoint to organized directory structure"""
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    filename = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint: {filename}")

def main():
    args = get_args()
    args.run_name = f"cifar10_{args.mode}_cfg_{args.cfg_prob}_bs_{args.batch_size}_lr_{args.lr}"
    
    # Auto-detect device if user didn't specify 'cpu' but has no CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not found, switching to CPU.")
        device = "cpu"
    else:
        device = args.device

    # Setup organized directory structure
    save_dir = os.path.join("./runs", args.run_name)
    samples_dir = os.path.join(save_dir, "samples")
    logs_dir = os.path.join(save_dir, "logs")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=logs_dir)
    
    # Save training configuration
    save_training_config(args, save_dir)

    print(f"--- Starting Training ---")
    print(f"Mode: {args.mode.upper()} | Device: {device} | Batch: {args.batch_size}")
    print(f"Run Name: {args.run_name}")
    print(f"Save Directory: {save_dir}")

    # 1. Prepare Data
    # Ensure correct root_dir for your environment
    train_loader = get_dataloader(
        root_dir="../data/cifar-10-batches-py", 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        train=True
    )
    
    # 2. Initialize Model & Diffusion Engine
    model = DiffusionUNet(num_classes=10, base_channels=args.base_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Initialize Physics (Flow or Diffusion)
    diffusion = DiffusionManager(mode=args.mode, device=device)

    # 3. Resume Logic (Optional)
    start_epoch = 1
    if args.resume:
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # 4. Training Loop
    criterion = nn.MSELoss() # Both modes use MSE
    
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        total_loss = 0
        batch_count = 0
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # --- CFG Training Logic ---
            if args.cfg_prob > 0:
                mask = torch.rand(labels.shape[0], device=device) < args.cfg_prob
                null_tokens = torch.full_like(labels, 10)
                train_labels = torch.where(mask, null_tokens, labels)
            else:
                train_labels = labels

            # --- Forward Process ---
            t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device).long()
            noisy_images, target = diffusion.q_sample(images, t)
            
            # --- Optimization ---
            optimizer.zero_grad()
            predicted_target = model(noisy_images, t, train_labels)
            loss = criterion(predicted_target, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix(loss=loss.item())
            
            # Log batch loss to TensorBoard
            global_step = (epoch - 1) * len(train_loader) + batch_count
            writer.add_scalar('Loss/batch', loss.item(), global_step)

        # Logging
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.5f}")
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # 5. Periodic Sampling & Saving
        if epoch % args.save_interval == 0:
            print(f"--> Generating samples...")
            # Sample: 4 Airplanes (0), 4 Dogs (5)
            sample_labels = torch.tensor([0]*4 + [5]*4).to(device)
            
            sampled_imgs = diffusion.sample(
                model, 
                n_samples=8, 
                labels=sample_labels, 
                steps=20, 
                cfg_scale=3.0
            )
            
            from torchvision.utils import save_image
            
            save_path = os.path.join(samples_dir, f"epoch_{epoch}.png")
            save_image(sampled_imgs, save_path, nrow=4)
            
            # Log sample images to TensorBoard
            writer.add_images('Samples/generated', sampled_imgs, epoch, dataformats='NCHW')
            
            save_checkpoint(model, optimizer, epoch, save_dir)
    
    # Close TensorBoard writer
    writer.close()
    print(f"\nTraining complete! All outputs saved to: {save_dir}")
    print(f"View logs with: tensorboard --logdir={logs_dir}")

if __name__ == "__main__":
    main()