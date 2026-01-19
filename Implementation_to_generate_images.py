import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL


# ==========================================

# MODEL ARCHITECTURE (Shortcut DiT)

# ==========================================


class TimestepEmbedder(nn.Module):
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
        return self.mlp(t_freq)


class DiTBlock(nn.Module):
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
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class FinalLayer(nn.Module):
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
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class ShortcutDiT(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=768, depth=12, num_heads=12, num_classes=1000):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (input_size // patch_size) ** 2, hidden_size))
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.d_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

    def unpatchify(self, x):
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, d, y):
        x = self.x_embedder(x).flatten(2).transpose(1, 2) + self.pos_embed
        c = self.t_embedder(t) + self.d_embedder(d) + self.y_embedder(y)
        for block in self.blocks: x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)


# ==========================================
#  TRAINING 
# ==========================================


class ShortcutTrainer:
    def __init__(self, model, device, lr=1e-4, m_steps=128):
        self.model = model.to(device)
        self.model_ema = copy.deepcopy(model).to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.device, self.M = device, m_steps
        self.num_classes = model.y_embedder.num_embeddings - 1

    def update_ema(self, decay=0.999):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(decay).add_(p.data, alpha=(1 - decay))

    def train_step(self, x1, y):
        x1, y = x1.to(self.device), y.to(self.device)
        x0 = torch.randn_like(x1)
        batch_size = x1.size(0)
        k_split = int(batch_size * 0.25)
        loss_total = 0

        # Flow Matching (d=0)
        if batch_size > k_split:
            x1_fm, x0_fm, y_fm = x1[k_split:], x0[k_split:], y[k_split:]
            t_fm = torch.rand(x1_fm.size(0), device=self.device)
            xt_fm = (1 - t_fm.view(-1, 1, 1, 1)) * x0_fm + t_fm.view(-1, 1, 1, 1) * x1_fm
            pred_fm = self.model(xt_fm, t_fm, torch.zeros_like(t_fm), y_fm)
            loss_total += F.mse_loss(pred_fm, x1_fm - x0_fm)

        # Self-Consistency (d > 0)
        if k_split > 0:
            x1_sc, x0_sc, y_sc = x1[:k_split], x0[:k_split], y[:k_split]
            powers = torch.randint(0, int(np.log2(self.M)), (k_split,), device=self.device)
            delta = (2 ** powers).float() / self.M
            t_sc = (torch.rand(k_split, device=self.device) * (self.M // (2**powers) - 1)).long().float() * delta
            xt_sc = (1 - t_sc.view(-1, 1, 1, 1)) * x0_sc + t_sc.view(-1, 1, 1, 1) * x1_sc

            with torch.no_grad():
                d_in = delta.clone()
                d_in[d_in == 1/self.M] = 0
                s_t = self.model_ema(xt_sc, t_sc, d_in, y_sc)
                x_next = xt_sc + s_t * delta.view(-1, 1, 1, 1)
                s_next = self.model_ema(x_next, t_sc + delta, d_in, y_sc)
                target_sc = (s_t + s_next) / 2.0

            loss_total += F.mse_loss(self.model(xt_sc, t_sc, 2*delta, y_sc), target_sc)

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        self.update_ema()
        return loss_total.item()

# ==========================================
# SAMPLING
# ==========================================


def sample(model, n, steps, num_classes, device, cfg=1.5):
    model.eval()
    x = torch.randn(n, 4, 32, 32, device=device)
    y = torch.randint(0, num_classes, (n,), device=device)
    y_null = torch.full_like(y, num_classes)
    dt = 1.0 / steps
    t_curr = 0.0
    for _ in range(steps):
        t_tr = torch.full((n,), t_curr, device=device)
        d_tr = torch.full((n,), 0.0 if steps == 128 else dt, device=device)
        with torch.no_grad():
            if cfg > 1.0:
                v = model(x, t_tr, d_tr, y_null) + cfg * (model(x, t_tr, d_tr, y) - model(x, t_tr, d_tr, y_null))
            else:
                v = model(x, t_tr, d_tr, y)
        x += v * dt
        t_curr += dt
    return x

# ==========================================
# EXECUTION WITH REAL DATA
# ==========================================


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Load VAE for Latent Compression
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    # 2. Setup Real Data Pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    DATA_PATH = "carbonara"
    if os.path.exists(DATA_PATH):
        full_dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
        target_classes = full_dataset.classes[1]
        class_to_idx = {cls: i for i, cls in enumerate(full_dataset.classes) if cls in target_classes}
        filtered_samples = [
            (path, label) for path, label in full_dataset.samples
            if full_dataset.classes[label] in target_classes
        ]

        # Overwrite the dataset samples and class list
        full_dataset.samples = filtered_samples
        full_dataset.classes = target_classes
        loader = DataLoader(full_dataset, batch_size=16, shuffle=True)
        num_classes = len(full_dataset.classes)
    else:
        print(f"Warning: {DATA_PATH} not found. Falling back to mock data.")
        loader = [(torch.randn(16, 3, 256, 256), torch.randint(0, 10, (16,))) for _ in range(5)]
        num_classes = 10

    # 3. Initialize Shortcut Model
    model = ShortcutDiT(input_size=32, hidden_size=768, depth=12, num_heads=12, num_classes=num_classes).to(device)
    trainer = ShortcutTrainer(model, device)

    # 4. Training Loop
    print("Starting Training...")
    epoch_losses = []
    for epoch in range(70):
        total_epoch_loss = 0
        num_batches = 0
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)

            # Encode images to Latents
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215
            loss = trainer.train_step(latents, labels)
            total_epoch_loss += loss
            num_batches += 1

            if i % 5 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss:.4f}")

        avg_loss = total_epoch_loss / num_batches
        epoch_losses.append(avg_loss)

        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        model.eval()  # Set to eval mode for sampling
        with torch.no_grad():
            # Sample 4 images (128-step for best quality)
            sample_latents = sample(trainer.model_ema, n=4, steps=128, num_classes=num_classes, device=device)
            # Decode using VAE
            sample_pixels = vae.decode(sample_latents / 0.18215).sample
            # Save progress image
            utils.save_image(sample_pixels, f"vis_epoch_{epoch+1}.png", nrow=2, normalize=True, value_range=(-1, 1))
            print(f"Visualization saved: vis_epoch_{epoch+1}.png")
        
        # Save a backup of the weights every epoch 
        torch.save(trainer.model_ema.state_dict(), "latest_model.pt")

    # 5. Generate and Decode Real Images
    print("\nGenerating Samples...")

    with torch.no_grad():
        # 1-Step Jump
        latents_1s = sample(trainer.model_ema, n=4, steps=1, num_classes=num_classes, device=device)
        # 128-Step Refinement
        latents_128s = sample(trainer.model_ema, n=4, steps=128, num_classes=num_classes, device=device)

        # Decode Latents -> Pixels
        imgs_1s = vae.decode(latents_1s / 0.18215).sample
        imgs_128s = vae.decode(latents_128s / 0.18215).sample

        # Save results
        utils.save_image(imgs_1s, "shortcut_1step.png", nrow=2, normalize=True, value_range=(-1, 1))
        utils.save_image(imgs_128s, "shortcut_128step.png", nrow=2, normalize=True, value_range=(-1, 1))
        print("Images saved: shortcut_1step.png and shortcut_128step.png")

    print("Saving Loss Plot...")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 251), epoch_losses, marker='o', label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_loss_plot.png')  
    print("Loss plot saved as training_loss_plot.png")

    # 6. Save the model weights
    model_path = "shortcut_dit_model.pt"
    torch.save({
        'model_state_dict': trainer.model_ema.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'num_classes': num_classes,
    }, model_path)

    print(f"Model saved to {model_path}")
