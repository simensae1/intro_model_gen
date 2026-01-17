import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

# ==========================================
# 1. MODEL ARCHITECTURE: DiT (Diffusion Transformer)
# Matches "DiT-B" specified in Section 5.1 [cite: 209]
# ==========================================

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Used for both 't' (current time) and 'd' (step size)[cite: 14, 95].
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTBlock(nn.Module):
    """
    Standard DiT Block with Adaptive Layer Norm (adaLN).
    Conditioning is added via scale and shift parameters.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        # adaLN modulation: 6 parameters (scale/shift for norm1, scale for attn, scale/shift for norm2, scale for mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-Attention
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class FinalLayer(nn.Module):
    """
    Final layer of DiT to map back to patch size.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x

class ShortcutDiT(nn.Module):
    """
    The full Shortcut Model architecture.
    Conditions on: Latents x_t, Timestep t, Step-size d, Class label y.
    """
    def __init__(self, 
                 input_size=32,    # Latent size (e.g. 256/8 = 32) 
                 patch_size=2,     #
                 in_channels=4,    # VAE latent channels
                 hidden_size=768,  # DiT-B Hidden Size
                 depth=12,         # DiT-B Layers
                 num_heads=12,     # DiT-B Heads
                 num_classes=1000, # ImageNet classes
                 learn_sigma=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # Patch Embedding
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (input_size // patch_size) ** 2, hidden_size), requires_grad=False)

        # Condition Embeddings
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.d_embedder = TimestepEmbedder(hidden_size) # Extra embedding for step size 'd' [cite: 39]
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size) # +1 for null class (CFG)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize pos_embed (sin-cos)
        # Simplified for brevity; in production use exact sin-cos init
        nn.init.normal_(self.pos_embed, std=0.02)

        # Zero-out adaLN modulation layers (standard DiT practice)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C) -> (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, d, y):
        """
        x: (N, C, H, W) tensor of spatial inputs (latents)
        t: (N,) tensor of diffusion timesteps
        d: (N,) tensor of step sizes
        y: (N,) tensor of class labels
        """
        # 1. Patchify
        x = self.x_embedder(x) # (N, C, H, W) -> (N, Hidden, H/p, W/p)
        x = x.flatten(2).transpose(1, 2) # (N, L, Hidden)
        x = x + self.pos_embed

        # 2. Combine Conditioning
        # The paper implies conditioning on t and d.
        # "Condition the network... on the desired step size" [cite: 14]
        # We sum embeddings: t_emb + d_emb + y_emb
        t_emb = self.t_embedder(t)
        d_emb = self.d_embedder(d)
        y_emb = self.y_embedder(y)
        c = t_emb + d_emb + y_emb

        # 3. Transformer Blocks
        for block in self.blocks:
            x = block(x, c)

        # 4. Final Layer
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

# ==========================================
# 2. TRAINING LOGIC (Matches Algorithm 1)
# ==========================================

class ShortcutTrainer:
    def __init__(self, model, device, lr=1e-4, weight_decay=0.1, m_steps=128):
        self.model = model.to(device)
        self.model_ema = copy.deepcopy(model).to(device)
        # Weight decay is crucial for stability [cite: 176]
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.M = m_steps 
        self.cfg_dropout = 0.1 # "Class Dropout Probability 0.1"
        self.num_classes = model.y_embedder.num_embeddings - 1

    def update_ema(self, decay=0.999):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(decay).add_(p.data, alpha=(1 - decay))

    def train_step(self, x1, y):
        """
        x1: Data samples (Latents)
        y: Class labels
        """
        x1 = x1.to(self.device)
        y = y.to(self.device)
        batch_size = x1.size(0)
        x0 = torch.randn_like(x1)

        # Apply Classifier-Free Guidance (CFG) during training by dropping labels
        # "Class Dropout Probability 0.1"
        if self.cfg_dropout > 0:
            drop_mask = torch.rand(batch_size, device=self.device) < self.cfg_dropout
            y_input = torch.where(drop_mask, torch.full_like(y, self.num_classes), y)
        else:
            y_input = y

        # Ratio of Empirical to Bootstrap Targets ~ 0.75
        k_split = int(batch_size * 0.25)
        loss_total = 0

        # --- PART 1: Flow Matching (d=0) [cite: 68] ---
        # "Regressing onto empirical samples... sampling t ~ U(0,1)" [cite: 144]
        if batch_size > k_split:
            x1_fm = x1[k_split:]
            x0_fm = x0[k_split:]
            y_fm = y_input[k_split:]

            t_fm = torch.rand(x1_fm.size(0), device=self.device)
            xt_fm = (1 - t_fm.view(-1, 1, 1, 1)) * x0_fm + t_fm.view(-1, 1, 1, 1) * x1_fm
            target_fm = x1_fm - x0_fm # Flow matching target v_t = x1 - x0 [cite: 61]

            d_zeros = torch.zeros_like(t_fm)
            pred_fm = self.model(xt_fm, t_fm, d_zeros, y_fm)
            loss_fm = F.mse_loss(pred_fm, target_fm)
            loss_total += loss_fm

        # --- PART 2: Self-Consistency (d > 0) [cite: 126] ---
        # "Algorithm 1: Shortcut Model Training"
        if k_split > 0:
            x1_sc = x1[:k_split]
            x0_sc = x0[:k_split]
            y_sc = y_input[:k_split]

            # Sample step size d (power of 2)
            max_power = int(np.log2(self.M)) 
            powers = torch.randint(0, max_power, (k_split,), device=self.device) 
            delta = (2 ** powers).float() / self.M 

            # Discrete time sampling [cite: 181]
            max_steps = (self.M // (2 ** powers))
            step_indices = (torch.rand(k_split, device=self.device) * (max_steps - 1)).long()
            t_sc = step_indices.float() * delta

            xt_sc = (1 - t_sc.view(-1, 1, 1, 1)) * x0_sc + t_sc.view(-1, 1, 1, 1) * x1_sc

            with torch.no_grad():
                # Base case handling [cite: 160]
                d_input = delta.clone()
                d_input[d_input == (1.0/self.M)] = 0.0

                # Use EMA for bootstrap targets [cite: 175]
                # First small step
                s_t = self.model_ema(xt_sc, t_sc, d_input, y_sc)
                x_next = xt_sc + s_t * delta.view(-1, 1, 1, 1)

                # Second small step
                s_next = self.model_ema(x_next, t_sc + delta, d_input, y_sc)

                # "s_target <- (s_t + s_t+d) / 2" [cite: 130]
                target_sc = (s_t + s_next) / 2.0

            # Train on the larger shortcut "2d"
            d_target = 2 * delta
            pred_sc = self.model(xt_sc, t_sc, d_target, y_sc)
            loss_sc = F.mse_loss(pred_sc, target_sc)
            loss_total += loss_sc

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        self.update_ema()

        return loss_total.item()

# ==========================================
# 3. SAMPLING (Algorithm 2 with CFG)
# ==========================================

def sample(model, n_samples, steps, num_classes, device, image_size=32, in_channels=4, cfg_scale=1.5):
    """
    Algorithm 2: Sampling with Classifier-Free Guidance.
    "CFG is used when evaluating the shortcut model at d=0"[cite: 171].
    However, standard practice allows CFG at any step by extrapolating noise estimates.
    """
    model.eval()
    x = torch.randn(n_samples, in_channels, image_size, image_size).to(device)
    y = torch.randint(0, num_classes, (n_samples,), device=device) # Random classes
    y_null = torch.full_like(y, num_classes) # Null tokens

    d_val = 1.0 / steps
    t_curr = 0.0

    for _ in range(steps):
        t_tensor = torch.full((n_samples,), t_curr, device=device)
        d_tensor = torch.full((n_samples,), d_val, device=device)

        # Base case logic for inference: if steps=128 (d=1/128), pass d=0 to model
        # "At d -> 0, the shortcut is equivalent to the flow" [cite: 104]
        # In code, we usually pass 0 if d is the minimum unit.
        d_input = d_tensor.clone()
        if steps == 128: # Assuming 128 is M
            d_input[:] = 0.0

        with torch.no_grad():
            if cfg_scale > 1.0:
                # CFG: pred = cond + scale * (cond - uncond)
                out_cond = model(x, t_tensor, d_input, y)
                out_uncond = model(x, t_tensor, d_input, y_null)
                vel = out_uncond + cfg_scale * (out_cond - out_uncond)
            else:
                vel = model(x, t_tensor, d_input, y)

        x = x + vel * d_val
        t_curr += d_val

    return x

# ==========================================
# 4. EXECUTION
# ==========================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model with DiT-B config (approximate for demo)
    # Note: input_size=32 corresponds to 256x256 image latents (downsampling 8)
    model = ShortcutDiT(input_size=32, hidden_size=384, depth=4, num_heads=6).to(device) 
    # (Reduced size for CPU/Colab testing; Use hidden=768, depth=12 for full DiT-B)
    
    trainer = ShortcutTrainer(model, device)

    # Mock Data (Simulating Latents from VAE)
    # "All runs use the latent space of the sd-vae-ft-mse autoencoder" 
    print("Training on mock latent data...")
    for step in range(100):
        # Batch of 64 latents (4 channels, 32x32 spatial)
        mock_x1 = torch.randn(64, 4, 32, 32) 
        mock_y = torch.randint(0, 1000, (64,))
        
        loss = trainer.train_step(mock_x1, mock_y)
        if step % 20 == 0:
            print(f"Step {step}: Loss {loss:.4f}")

    print("\nGenerating 1-step sample...")
    # "Shortcut models... produce high-quality samples in a single... step" [cite: 13]
    samples_1step = sample(trainer.model_ema, n_samples=4, steps=1, num_classes=1000, device=device, cfg_scale=1.0)
    print("1-Step Generation Complete. Shape:", samples_1step.shape)
    
    print("\nGenerating 128-step sample...")
    samples_128step = sample(trainer.model_ema, n_samples=4, steps=128, num_classes=1000, device=device, cfg_scale=1.5)
    print("128-Step Generation Complete. Shape:", samples_128step.shape)