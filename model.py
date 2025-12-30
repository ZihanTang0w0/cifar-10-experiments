import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResBlock(nn.Module):
    def __init__(self, in_channels, emb_dim, out_channels=None, dropout=0.1):
        super().__init__()
        
        # If out_channels is not provided, default to same as input
        self.out_channels = out_channels if out_channels else in_channels
        
        # 1. First Convolution
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, self.out_channels, 3, padding=1) 
            # Note: We change channels here in the first conv
        )
        
        # 2. Embedding Projection
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, self.out_channels), 
        )
        
        # 3. Second Convolution
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        )
        
        # 4. Skip Connection Handling
        # If input channels != output channels, we cannot just add x + h.
        # We must project x to match the shape of h.
        if in_channels != self.out_channels:
            self.skip_connection = nn.Conv2d(in_channels, self.out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, t_emb):
        # A. Main Path
        h = self.in_layers(x)
        
        # B. Inject Embedding
        emb_out = self.emb_layers(t_emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out 
        
        h = self.out_layers(h)
        
        # C. Skip Connection
        # If channels changed, self.skip_connection applies a 1x1 conv to x
        return self.skip_connection(x) + h

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # 1. Norm (Standard GroupNorm)
        self.norm = nn.GroupNorm(32, channels)
        
        # 2. QKV Projection (Combined for efficiency)
        # We project to 3x channels (Query, Key, Value)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        
        # 3. Output Projection
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
        # 4. Initialize weights (Critical for Diffusion convergence)
        # We usually initialize the output layer to zero so the block starts as an identity
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # --- 1. Normalize ---
        h_ = self.norm(x)
        
        # --- 2. Calculate Q, K, V ---
        qkv = self.qkv(h_)
        q, k, v = qkv.chunk(3, dim=1)
        
        # --- 3. Reshape for Multi-Head Attention ---
        # Reshape: [Batch, Channels, H, W] -> [Batch, Heads, H*W, Channels/Heads]
        # This effectively turns the 2D image into a 1D sequence of pixels
        head_dim = c // self.num_heads
        
        q = q.view(b, self.num_heads, head_dim, h*w)
        k = k.view(b, self.num_heads, head_dim, h*w)
        v = v.view(b, self.num_heads, head_dim, h*w)
        
        # --- 4. The Math (Scaled Dot-Product Attention) ---
        # Q * K_transpose
        attn_scores = torch.einsum('b h d i, b h d j -> b h i j', q, k) 
        attn_scores = attn_scores * (head_dim ** -0.5) # Scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Attn * V
        out = torch.einsum('b h i j, b h d j -> b h d i', attn_probs, v)
        
        # --- 5. Reshape Back to Image ---
        out = out.reshape(b, c, h, w)
        
        # --- 6. Output Projection + Skip Connection ---
        return x + self.proj_out(out)
    

class DiffusionUNet(nn.Module):
    def __init__(self, num_classes=10, base_channels=128):
        super().__init__()
        
        # --- 1. Conditioning System (Time + Class) ---
        self.time_emb_dim = base_channels * 4
        
        # Projects standard sinusoidal time to embedding vector
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )
        
        # Class embedding: 10 real classes + 1 null class for CFG = 11
        self.class_emb = nn.Embedding(num_classes + 1, self.time_emb_dim)

        # --- 2. Initial Convolution ---
        self.init_conv = nn.Conv2d(3, base_channels, 3, padding=1)

        # --- 3. Down Sampling Path ---
        # Level 1: 32x32 (128 channels) - No Attn
        self.down1 = nn.ModuleList([
            ResBlock(base_channels, self.time_emb_dim),
            ResBlock(base_channels, self.time_emb_dim)
        ])
        self.down1_pool = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)

        # Level 2: 16x16 (128 -> 256 channels) - With Attn
        self.down2 = nn.ModuleList([
            ResBlock(base_channels, self.time_emb_dim, out_channels=base_channels*2), # Channel increase here
            ResBlock(base_channels*2, self.time_emb_dim),
            SelfAttentionBlock(base_channels*2)
        ])
        self.down2_pool = nn.Conv2d(base_channels*2, base_channels*2, 3, stride=2, padding=1)

        # Level 3: 8x8 (256 channels) - With Attn
        self.down3 = nn.ModuleList([
            ResBlock(base_channels*2, self.time_emb_dim),
            ResBlock(base_channels*2, self.time_emb_dim),
            SelfAttentionBlock(base_channels*2)
        ])
        self.down3_pool = nn.Conv2d(base_channels*2, base_channels*2, 3, stride=2, padding=1)

        # --- 4. Bottleneck ---
        self.bot1 = ResBlock(base_channels*2, self.time_emb_dim)
        self.bot2 = SelfAttentionBlock(base_channels*2)
        self.bot3 = ResBlock(base_channels*2, self.time_emb_dim)

        # --- 5. Up Sampling Path (Requires handling Skip Connections) ---
        # Level 3 Up: 4x4 -> 8x8
        self.up1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1 = nn.ModuleList([
            # In_channels is doubled because of skip connection (256 from bottom + 256 from skip)
            ResBlock(base_channels*4, self.time_emb_dim, out_channels=base_channels*2),
            ResBlock(base_channels*2, self.time_emb_dim),
            SelfAttentionBlock(base_channels*2)
        ])

        # Level 2 Up: 8x8 -> 16x16
        self.up2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.ModuleList([
            # Skip connection from Level 2 is 256 ch. Input is 256. Total in = 512.
            ResBlock(base_channels*4, self.time_emb_dim, out_channels=base_channels*2),
            ResBlock(base_channels*2, self.time_emb_dim),
            SelfAttentionBlock(base_channels*2)
        ])

        # Level 1 Up: 16x16 -> 32x32
        self.up3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.ModuleList([
            # Skip connection from Level 1 is 128 ch. Input is 256. Total in = 384.
            ResBlock(base_channels*2 + base_channels, self.time_emb_dim, out_channels=base_channels),
            ResBlock(base_channels, self.time_emb_dim),
            # No Attention at 32x32
        ])

        # --- 6. Final Output ---
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1) # Map back to RGB
        )

    def forward(self, x, t, class_labels):
        # 1. Conditioning
        # t is shape [Batch], convert to sinusoidal [Batch, Base_Ch]
        t = sinusoidal_embedding(t, 128) 
        t_emb = self.time_mlp(t)
        
        c_emb = self.class_emb(class_labels)
        
        # Additive Conditioning
        cond = t_emb + c_emb 

        # 2. Initial Conv
        x = self.init_conv(x)
        
        # 3. Down Path (Saving skips)
        skips = []
        
        # Down 1
        for layer in self.down1: x = layer(x, cond) # Note: ResBlock needs cond
        skips.append(x)
        x = self.down1_pool(x)
        
        # Down 2
        x = self.down2[0](x, cond) # ResBlock
        x = self.down2[1](x, cond) # ResBlock
        x = self.down2[2](x)       # Attention (No cond needed usually)
        skips.append(x)
        x = self.down2_pool(x)
        
        # Down 3
        x = self.down3[0](x, cond)
        x = self.down3[1](x, cond)
        x = self.down3[2](x)
        skips.append(x)
        x = self.down3_pool(x)

        # 4. Bottleneck
        x = self.bot1(x, cond)
        x = self.bot2(x)
        x = self.bot3(x, cond)

        # 5. Up Path (Concatenate with skips)
        
        # Up 1 (Input 4x4 -> 8x8)
        x = self.up1_upsample(x)
        skip = skips.pop() # Get Level 3 skip
        x = torch.cat([x, skip], dim=1) 
        x = self.up1[0](x, cond)
        x = self.up1[1](x, cond)
        x = self.up1[2](x)

        # Up 2 (Input 8x8 -> 16x16)
        x = self.up2_upsample(x)
        skip = skips.pop() # Get Level 2 skip
        x = torch.cat([x, skip], dim=1)
        x = self.up2[0](x, cond)
        x = self.up2[1](x, cond)
        x = self.up2[2](x)

        # Up 3 (Input 16x16 -> 32x32)
        x = self.up3_upsample(x)
        skip = skips.pop() # Get Level 1 skip
        x = torch.cat([x, skip], dim=1)
        x = self.up3[0](x, cond)
        x = self.up3[1](x, cond)

        return self.out_layers(x)

def sinusoidal_embedding(timesteps, dim):
    """
    Args:
        timesteps: A 1-D Tensor of shape [Batch_Size] containing the timesteps (e.g., [50, 999, ...])
        dim: The target embedding dimension (e.g., 128)
    Returns:
        embedding: Tensor of shape [Batch_Size, dim]
    """
    # 1. Calculate the half-dimension (since we use sin/cos pairs)
    half_dim = dim // 2
    
    # 2. Calculate the frequencies (emb = 10000^(2i/d))
    # Using log-space is numerically more stable: exp(-log(10000) * i / half_dim)
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    
    # 3. Create the arguments for sin/cos
    # Shape: [Batch_Size, 1] * [1, Half_Dim] = [Batch_Size, Half_Dim]
    emb = timesteps[:, None].float() * emb[None, :]
    
    # 4. Concatenate sin and cos
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    
    # 5. Handle odd dimensions (just in case dim is odd, though usually it's even)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        
    return emb