import torch
import torch.nn.functional as F
from tqdm import tqdm

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

class DiffusionManager:
    def __init__(self, mode='diffusion', timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cuda"):
        """
        Args:
            mode: 'diffusion' (Standard DDPM/DDIM) or 'flow' (Rectified Flow)
        """
        self.mode = mode
        self.timesteps = timesteps
        self.device = device
        
        if self.mode == 'diffusion':
            # --- Diffusion Specific Setup ---
            self.betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(device)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
            
        elif self.mode == 'flow':
            # --- Flow Specific Setup ---
            # Flow doesn't need betas/alphas. It just needs raw time 0.0 -> 1.0
            # We don't precompute anything here.
            pass
        
    def extract(self, a, t, x_shape):
        """
        Helper to extract constants for specific timesteps.
        Inputs:
            a: The full array of constants (e.g., alphas_cumprod) [Timesteps]
            t: The timestep indices [Batch_Size]
            x_shape: The shape of the image tensor [Batch, C, H, W]
        """
        batch_size = t.shape[0]
        
        # 1. Ensure t is on the same device as a
        # If 'a' is on CUDA, 't' must be on CUDA.
        t = t.to(a.device)
        
        # 2. Gather values
        out = a.gather(-1, t)
        
        # 3. Reshape for broadcasting
        # We want shape [Batch, 1, 1, 1] so we can multiply with [Batch, C, H, W]
        return out.reshape(batch_size, 1, 1, 1)
    
    # ==========================================
    # 1. Unified Training Forward Process (Add Noise)
    # ==========================================
    def q_sample(self, x_start, t, noise=None):
        """
        Add noise to the image based on the selected mode.
        Args:
            x_start: Clean image
            t: Timestep (Integer 0-999)
            noise: Optional noise tensor
        Returns:
            noisy_image, target_for_loss
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        if self.mode == 'diffusion':
            # --- Diffusion: Curved Path on Hypersphere ---
            # x_t = sqrt(alpha)*x + sqrt(1-alpha)*eps
            sqrt_alpha = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_sigma = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            
            noisy_img = sqrt_alpha * x_start + sqrt_sigma * noise
            target = noise # Diffusion predicts NOISE
            return noisy_img, target

        elif self.mode == 'flow':
            # --- Flow: Straight Line Interpolation ---
            # Convert integer t (0..1000) to float (0.0..1.0)
            # We treat t=0 as Data, t=1 as Noise (Diffusion Convention)
            t_float = t.view(-1, 1, 1, 1).float() / self.timesteps
            
            # x_t = (1 - t) * x_0 + t * x_1 (noise)
            noisy_img = (1 - t_float) * x_start + t_float * noise
            
            # Flow Target: Velocity (The direction from Data to Noise)
            # v = x_1 - x_0 = noise - x_start
            target = noise - x_start 
            return noisy_img, target

    # ==========================================
    # 2. Unified Sampling (ODE Solvers)
    # ==========================================
    @torch.no_grad()
    def sample(self, model, n_samples, labels, steps=50, cfg_scale=3.0, checkpoint_steps=None):
        """
        Unified sampler. Uses DDIM for Diffusion and Euler for Flow.
        
        Args:
            checkpoint_steps: Optional list of step counts to save intermediate results.
                            If provided, returns dict {step_count: images}, otherwise returns final images only.
        """
        model.eval()
        # Start with pure noise
        img = torch.randn((n_samples, 3, 32, 32), device=self.device)
        
        # Create uniform time steps from T_max down to 0
        # e.g. [999, 979, ..., 0]
        times = torch.linspace(0, self.timesteps - 1, steps=steps + 1).long().to(self.device)
        times = torch.flip(times, [0]) 
        
        print(f"Sampling with {self.mode.upper()} mode ({steps} steps)...")
        
        # Dictionary to store checkpointed images if requested
        checkpointed_images = {} if checkpoint_steps is not None else None
        steps_taken = 0
        
        for i in tqdm(range(len(times) - 1), desc='Sampling'):
            t_now = times[i]
            t_next = times[i+1]
            
            # Broadcast t to batch
            batch_t = torch.full((n_samples,), t_now, device=self.device, dtype=torch.long)
            
            # --- Call the appropriate Solver ---
            if self.mode == 'diffusion':
                img = self.ddim_step(model, img, batch_t, t_now, t_next, labels, cfg_scale)
            
            elif self.mode == 'flow':
                img = self.euler_step(model, img, t_now, t_next, labels, cfg_scale)
            
            steps_taken += 1
            
            # Check if we need to save a checkpoint
            if checkpoint_steps is not None and steps_taken in checkpoint_steps:
                # Clone and process the current state
                checkpoint_img = (img.clone() + 1) * 0.5
                checkpointed_images[steps_taken] = checkpoint_img.clamp(0, 1).cpu()
                
        # Final cleanup
        img = (img + 1) * 0.5
        final_img = img.clamp(0, 1)
        
        # Return based on whether checkpointing was requested
        if checkpoint_steps is not None:
            # Make sure final step is included
            if steps not in checkpointed_images:
                checkpointed_images[steps] = final_img.cpu()
            return checkpointed_images
        else:
            return final_img

    # --- A. DDIM Solver (Diffusion) ---
    def ddim_step(self, model, x, t, t_now, t_next, labels, cfg_scale):
        # 1. CFG Prediction
        noise_pred = self.get_cfg_prediction(model, x, t, labels, cfg_scale)

        # 2. Get Alphas
        alpha_now = self.alphas_cumprod[t_now]
        alpha_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=self.device)

        # 3. Predict x0
        sqrt_alpha_now = torch.sqrt(alpha_now)
        sqrt_sigma_now = torch.sqrt(1.0 - alpha_now)
        x0_pred = (x - sqrt_sigma_now * noise_pred) / sqrt_alpha_now
        
        # 4. Point to x_next
        dir_xt = torch.sqrt(1.0 - alpha_next) * noise_pred
        x_prev = torch.sqrt(alpha_next) * x0_pred + dir_xt
        return x_prev

    # --- B. Euler Solver (Flow) ---
    def euler_step(self, model, x, t_now, t_next, labels, cfg_scale):
        # 1. Calculate time delta (dt)
        # Convert integer steps to float 0..1 range
        t_float_now = t_now / self.timesteps
        t_float_next = t_next / self.timesteps
        dt = t_float_next - t_float_now # This will be negative (e.g. -0.02)
        
        # 2. CFG Prediction (Predicts Velocity 'v')
        # We must pass t as an integer to the model (for embedding)
        batch_t = torch.full((x.shape[0],), t_now, device=self.device, dtype=torch.long)
        velocity_pred = self.get_cfg_prediction(model, x, batch_t, labels, cfg_scale)
        
        # 3. Euler Update
        # x_next = x_now + velocity * dt
        x_prev = x + velocity_pred * dt
        return x_prev

    # --- Helper for CFG ---
    def get_cfg_prediction(self, model, x, t, labels, cfg_scale):
        null_labels = torch.full_like(labels, 10)
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        c_in = torch.cat([labels, null_labels], dim=0)
        
        pred = model(x_in, t_in, c_in)
        pred_cond, pred_uncond = pred.chunk(2, dim=0)
        
        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)