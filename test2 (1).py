import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- 1. MODEL ARCHITECTURE (ResNet-MLP) ---
# Improved architecture to better handle the conditioning on t and d.


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.linear2(h)
        h = self.dropout(h)
        return x + h


class ShortcutNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, embed_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Embeddings for Time (t) and Step Size (d)
        self.t_emb = SinusoidalPosEmb(embed_dim)
        self.d_emb = SinusoidalPosEmb(embed_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cond_proj = nn.Linear(embed_dim * 2, hidden_dim)

        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(4)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t, d):
        # Embed t and d
        t_feat = self.t_emb(t)
        d_feat = self.d_emb(d)
        cond = torch.cat([t_feat, d_feat], dim=-1)
        cond = self.cond_proj(cond)

        h = self.input_proj(x)
        h = h + cond  # Add condition to features

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        out = self.out_proj(h)
        return out

# --- 2. TRAINING ENGINE (Algorithm 1) ---


class ShortcutTrainer:
    def __init__(self, model, device, lr=1e-4, weight_decay=0.1, m_steps=128):
        self.model = model.to(device)
        self.model_ema = copy.deepcopy(model).to(device)
        # Weight decay is crucial for stability
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.M = m_steps  # M = 128 [cite: 149]

    def update_ema(self, decay=0.999):
        # EMA parameters used for bootstrap targets
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(decay).add_(p.data, alpha=(1 - decay))

    def train_step(self, x1):
        x1 = x1.view(x1.size(0), -1).to(self.device)
        x0 = torch.randn_like(x1)
        batch_size = x1.size(0)

        # Ratio of Empirical (Flow Matching) to Bootstrap (Consistency) targets is ~0.75 / 0.25
        k_split = int(batch_size * 0.25)

        loss_total = 0

        # --- PART 1: Flow Matching (d=0) [cite: 68] ---
        # "Regressing onto empirical samples" [cite: 141]
        x1_fm = x1[k_split:]
        x0_fm = x0[k_split:]

        t_fm = torch.rand(x1_fm.size(0), device=self.device)
        # Interpolation x_t
        xt_fm = (1 - t_fm.view(-1, 1)) * x0_fm + t_fm.view(-1, 1) * x1_fm
        # Target velocity v_t = x1 - x0
        target_fm = x1_fm - x0_fm

        # Train model with d=0
        d_zeros = torch.zeros_like(t_fm)
        pred_fm = self.model(xt_fm, t_fm, d_zeros)
        loss_fm = F.mse_loss(pred_fm, target_fm)
        loss_total += loss_fm

        # --- PART 2: Self-Consistency (d > 0) [cite: 126] ---
        # "Enforcing self-consistency" via Algorithm 1
        if k_split > 0:
            x1_sc = x1[:k_split]
            x0_sc = x0[:k_split]

            # 1. Sample the size of the shortcut we want to LEARN (2 * delta)
            # We assume possible step sizes are 1/M, 2/M, 4/M ... up to 1.
            # We sample the 'sub-step' delta from {1/M, ..., 1/2}.
            max_power = int(np.log2(self.M))  # e.g., log2(128) = 7
            powers = torch.randint(0, max_power, (k_split,), device=self.device)  # 0 to 6

            # delta is the size of the smaller steps we take to build the target
            delta = (2 ** powers).float() / self.M  # 1/128, 2/128, ...

            # 2. Sample t. Discrete time sampling: t must be multiple of delta
            # Max steps available for this delta is M / (2^power)
            max_steps = (self.M // (2 ** powers))
            step_indices = (torch.rand(k_split, device=self.device) * (max_steps - 1)).long()
            t_sc = step_indices.float() * delta

            # Create x_t using the ODE (linear interpolation of noise and data)
            xt_sc = (1 - t_sc.view(-1, 1)) * x0_sc + t_sc.view(-1, 1) * x1_sc

            with torch.no_grad():
                # Handling the base case:
                # "When d is at the smallest value... we instead query the model at d=0"
                # If delta is 1/M, the input 'd' to the model for the sub-step should be 0.
                d_input = delta.clone()
                d_input[d_input == (1.0/self.M)] = 0.0

                # First small step: s_t <- s_theta(x_t, t, d)
                s_t = self.model_ema(xt_sc, t_sc, d_input)

                # Euler update: x_{t+d} <- x_t + s_t * d
                x_next = xt_sc + s_t * delta.view(-1, 1)

                # Second small step: s_{t+d} <- s_theta(x_{t+d}, t+d, d)
                s_next = self.model_ema(x_next, t_sc + delta, d_input)

                # Self-consistency target: average of the two velocities
                target_sc = (s_t + s_next) / 2.0

            # Train the model to predict this "double step" in a single pass
            # The model is queried at (x_t, t, 2*delta)
            d_target = 2 * delta
            pred_sc = self.model(xt_sc, t_sc, d_target)

            loss_sc = F.mse_loss(pred_sc, target_sc)
            loss_total += loss_sc

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        self.update_ema()

        return loss_total.item()

# --- 3. MAIN EXECUTION ---


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters from Table 3 [cite: 469] (adapted for MNIST)
    model = ShortcutNet(input_dim=784, hidden_dim=512, embed_dim=128).to(device)
    trainer = ShortcutTrainer(model, device, lr=1e-4, weight_decay=0.1, m_steps=128)

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=True)

    print("Starting Shortcut Model Training...")
    epochs = 200

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(loader):
            loss = trainer.train_step(data)
            total_loss += loss

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss:.4f}")

        print(f"==> Epoch {epoch} Average Loss: {total_loss/len(loader):.4f}")

    # --- EVALUATION ---
    # We compare 1-step, 4-step, and 128-step generation as shown in Figure 1 [cite: 27]
    print("Generating samples...")
    model.eval()

    # Use EMA weights for evaluation
    eval_model = trainer.model_ema

    n_samples = 10
    x_init = torch.randn(n_samples, 784).to(device)

    # Helper for sampling loop
    def sample_fn(model, x, steps):
        curr_x = x.clone()
        d_val = 1.0 / steps
        t_curr = 0.0

        # Algorithm 2: Sampling [cite: 155]
        for _ in range(steps):
            # Input tensors
            t_tensor = torch.full((x.size(0),), t_curr, device=x.device)
            d_tensor = torch.full((x.size(0),), d_val, device=x.device)

            with torch.no_grad():
                vel = model(curr_x, t_tensor, d_tensor)

            curr_x = curr_x + vel * d_val
            t_curr += d_val

        return curr_x

    # 1. One Step Generation (d=1.0)
    img_1step = sample_fn(eval_model, x_init, steps=1)

    # 2. Four Step Generation (d=0.25) [cite: 22]
    img_4step = sample_fn(eval_model, x_init, steps=4)

    # 3. 128 Step Generation (d=1/128) [cite: 21]
    # Note: For very small steps, the paper implies d converges to flow.
    # We pass 1/128 to the model.
    img_128step = sample_fn(eval_model, x_init, steps=128)

    # --- VISUALIZATION ---
    def process_img(tensor):
        img = tensor.detach().cpu().numpy()
        img = img * 0.5 + 0.5
        return np.clip(img, 0, 1).reshape(-1, 28, 28)

    vis_1 = process_img(img_1step)
    vis_4 = process_img(img_4step)
    vis_128 = process_img(img_128step)

    plt.figure(figsize=(10, 6))
    for i in range(n_samples):
        # 1 Step
        plt.subplot(3, n_samples, i + 1)
        plt.imshow(vis_1[i], cmap='gray')
        if i == 0: plt.ylabel("1 Step", fontsize=12, fontweight='bold')
        plt.axis('off')

        # 4 Steps
        plt.subplot(3, n_samples, n_samples + i + 1)
        plt.imshow(vis_4[i], cmap='gray')
        if i == 0: plt.ylabel("4 Steps", fontsize=12, fontweight='bold')
        plt.axis('off')

        # 128 Steps
        plt.subplot(3, n_samples, 2 * n_samples + i + 1)
        plt.imshow(vis_128[i], cmap='gray')
        if i == 0: plt.ylabel("128 Steps", fontsize=12, fontweight='bold')
        plt.axis('off')

    plt.suptitle(f"Shortcut Models: Consistency across budgets\n(Epoch {epochs})", fontsize=14)
    plt.tight_layout()
    plt.savefig("shortcut_results200epochs1024batch.png")
    plt.show()
