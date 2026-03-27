import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import copy


class DeepCausalMediation(nn.Module):
    def __init__(self, k_clusters=10, latent_dim=16,
                 input_channels=1, spatial_dim=(64, 64)):
        super().__init__()
        self.K = k_clusters
        self.latent_dim = latent_dim

        # Z-Encoder (unchanged)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.cluster_head = nn.Linear(32, k_clusters)
        self.cluster_embeddings = nn.Embedding(k_clusters, latent_dim)

        # P(B_k | X): mediator model (unchanged)
        self.mediator_model = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, k_clusters)
        )

        # Direct effect: X -> scalar shift δ(X)
        # Only sees X; captures everything not routed through Z
        self.direct_head = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)          # outputs δ (mean shift)
        )

        # Mediated effect: B_k -> scalar shift γ(B_k)
        # Only sees the cluster embedding; no X allowed here
        self.mediated_head = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 1)          # outputs γ_k
        )

        # Shared log-variance head (depends on X for heteroskedasticity)
        self.logvar_head = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def encode_z(self, z, tau=1.0, hard=False):
        z_feat = self.encoder(z)
        logits = self.cluster_head(z_feat)
        q_k_z = F.gumbel_softmax(logits, tau=tau, hard=hard)
        z_tilde = torch.matmul(q_k_z, self.cluster_embeddings.weight)
        return z_tilde, q_k_z

    def forward(self, x, z, tau=1.0):
        # Mediator path
        p_k_x_logits = self.mediator_model(x)
        p_k_x = F.softmax(p_k_x_logits, dim=-1)

        # Encoder path
        z_tilde, q_k_z = self.encode_z(z, tau=tau)

        # --- Decomposed outcome ---
        delta   = self.direct_head(x)            # direct effect of X
        gamma   = self.mediated_head(z_tilde)     # mediated effect through B_k
        mu      = delta + gamma                   # additive combination
        log_var = self.logvar_head(x)             # variance depends on X

        return mu, log_var, q_k_z, p_k_x


@torch.no_grad()
@torch.no_grad()
def estimate_pn(model, x_c, x_f, threshold, device='cpu'):
    """
    P(Y > τ | do(X=x)) = Σ_k P(B_k|X=x) · P(Y > τ | X=x, B_k)
    
    With additive decomposition μ(x, B_k) = δ(x) + γ(B_k):
    this integral has a clean analytic form per cluster.
    
    PN = max(0, 1 - P_c / P_f)  [lower bound under monotonicity + exogeneity]
    """
    model.eval()

    def get_p_y_do_x(x_val):
        x_t = torch.tensor([[x_val]], dtype=torch.float32).to(device)

        # --- Direct effect (fixed for this x) ---
        delta   = model.direct_head(x_t)          # shape (1,1)
        log_var = model.logvar_head(x_t)
        sigma   = torch.exp(0.5 * log_var).item()

        # --- Cluster distribution P(B_k | X=x) ---
        p_k_x = F.softmax(
            model.mediator_model(x_t), dim=-1
        ).squeeze()                                # shape (K,)

        prob_exceed = 0.0
        for k in range(model.K):
            emb = model.cluster_embeddings.weight[k:k+1]  # (1, latent_dim)

            # Mediated contribution for cluster k
            gamma_k = model.mediated_head(emb).item()     # scalar

            # Total mean for this cluster
            mu_k = delta.item() + gamma_k

            # P(Y > τ | X=x, B_k) — analytic Gaussian tail
            dist = Normal(
                torch.tensor(mu_k),
                torch.tensor(sigma)
            )
            p_exceed_k = 1.0 - dist.cdf(
                torch.tensor(float(threshold))
            )
            prob_exceed += p_exceed_k.item() * p_k_x[k].item()

        return prob_exceed

    p_c = get_p_y_do_x(x_c)
    p_f = get_p_y_do_x(x_f)

    # Guard against zero denominator
    if p_f < 1e-9:
        raise ValueError(
            f"P(Y>τ | do(X={x_f})) ≈ 0. "
            "The threshold may be too extreme for the factual treatment. "
            "PN is undefined (0/0 regime)."
        )

    pn = max(0.0, 1.0 - p_c / p_f)
    return p_c, p_f, pn
    


def train_causal_model(
    X, Y, Z,
    model,
    epochs=200,
    batch_size=32,
    lr=1e-3,
    patience=15,
    val_split=0.2,
    tau_start=1.0,
    tau_min=0.1,
    lambda_kl_start=0.0,   # start at 0, anneal up — lets NLL stabilise first
    lambda_kl_max=1.0,
    lambda_kl_warmup=0.3,  # fraction of epochs to reach lambda_kl_max
    grad_clip=1.0,         #  max gradient norm
    seed=42,
    num_workers=0,         # set >0 if not on Windows and data loading is a bottleneck
):
    # --- Reproducibility ---
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # cosine LR schedule — decays smoothly to lr/10 over training.
    # Works well with Adam: keeps large steps early, fine-tunes late.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 10
    )

    # --- Data preparation ---
    X_t = torch.tensor(X, dtype=torch.float32).view(-1, 1)
    Y_t = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

    # guard against Z already having a channel dimension
    Z_arr = np.array(Z)
    if Z_arr.ndim == 3:          # (N, H, W) → add channel dim
        Z_t = torch.tensor(Z_arr, dtype=torch.float32).unsqueeze(1)
    elif Z_arr.ndim == 4:        # (N, C, H, W) → already correct
        Z_t = torch.tensor(Z_arr, dtype=torch.float32)
    else:
        raise ValueError(f"Z must be 3-D (N,H,W) or 4-D (N,C,H,W), got shape {Z_arr.shape}")

    dataset = TensorDataset(X_t, Y_t, Z_t)

    num_samples = len(dataset)
    indices = list(range(num_samples))
    rng = np.random.default_rng(seed)   # seeded RNG for the split
    rng.shuffle(indices)
    split = int(np.floor(val_split * num_samples))
    train_idx, val_idx = indices[split:], indices[:split]

    pin = device.type == "cuda"   # pin_memory only makes sense with CUDA
    train_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        pin_memory=pin, num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=SubsetRandomSampler(val_idx),
        pin_memory=pin, num_workers=num_workers
    )

    # --- Annealing schedules ---
    # use multiplicative tau decay so the schedule is
    # independent of how many epochs actually run (early stopping safe).
    # tau_{t+1} = max(tau_min, tau_t * decay_factor)
    tau_decay_factor = (tau_min / tau_start) ** (1.0 / max(1, int(epochs * 0.7)))

    lambda_kl_warmup_epochs = max(1, int(epochs * lambda_kl_warmup))

    # keep best weights in memory — no disk I/O on every improvement
    best_state  = copy.deepcopy(model.state_dict())
    best_val_loss      = float("inf")
    epochs_no_improve  = 0
    tau = tau_start

    # history dict so the caller can inspect loss curves
    history = {
        "train_nll":    [],
        "train_kl":     [],
        "train_total":  [],
        "val_nll":      [],
        "val_kl":       [],
        "val_total":    [],
        "tau":          [],
        "lambda_kl":    [],
        "lr":           [],
    }

    for epoch in range(epochs):

        # --- Lambda KL schedule: linear warm-up from 0 to lambda_kl_max ---
        # Starting at 0 lets the NLL (outcome model) stabilise before the
        # KL term forces the mediator to align — avoids early collapse.
        lambda_kl = min(
            lambda_kl_max,
            lambda_kl_max * (epoch + 1) / lambda_kl_warmup_epochs
        )

        # --- Training ---
        model.train()
        train_nll_acc = train_kl_acc = 0.0

        for batch_x, batch_y, batch_z in train_loader:
            batch_x = batch_x.to(device, non_blocking=pin)
            batch_y = batch_y.to(device, non_blocking=pin)
            batch_z = batch_z.to(device, non_blocking=pin)

            optimizer.zero_grad()

            mu, log_var, q_k_z, p_k_x = model(batch_x, batch_z, tau=tau)

            # retrieve individual loss components for logging
            nll, kl = compute_loss_components(mu, log_var, batch_y, q_k_z, p_k_x)
            loss = nll + lambda_kl * kl

            loss.backward()

            # gradient clipping — important with Gumbel-softmax
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_nll_acc += nll.item()
            train_kl_acc  += kl.item()

        scheduler.step()

        # --- Validation ---
        # pass tau=0 (hard assignment, no Gumbel noise) so val loss
        # is deterministic and comparable across epochs.
        model.eval()
        val_nll_acc = val_kl_acc = 0.0

        with torch.no_grad():
            for batch_x, batch_y, batch_z in val_loader:
                batch_x = batch_x.to(device, non_blocking=pin)
                batch_y = batch_y.to(device, non_blocking=pin)
                batch_z = batch_z.to(device, non_blocking=pin)

                mu, log_var, q_k_z, p_k_x = model(batch_x, batch_z, tau=0.0)
                nll, kl = compute_loss_components(mu, log_var, batch_y, q_k_z, p_k_x)
                val_nll_acc += nll.item()
                val_kl_acc  += kl.item()

        n_train = len(train_loader)
        n_val   = len(val_loader)
        avg_train_total = (train_nll_acc + lambda_kl * train_kl_acc) / n_train
        avg_val_total   = (val_nll_acc   + lambda_kl * val_kl_acc)   / n_val

        # --- Logging ---
        history["train_nll"].append(train_nll_acc / n_train)
        history["train_kl"].append(train_kl_acc   / n_train)
        history["train_total"].append(avg_train_total)
        history["val_nll"].append(val_nll_acc     / n_val)
        history["val_kl"].append(val_kl_acc       / n_val)
        history["val_total"].append(avg_val_total)
        history["tau"].append(tau)
        history["lambda_kl"].append(lambda_kl)
        history["lr"].append(scheduler.get_last_lr()[0])

        # multiplicative tau decay — schedule survives early stopping
        tau = max(tau_min, tau * tau_decay_factor)

        # --- Early stopping on val total loss ---
        if avg_val_total < best_val_loss:
            best_val_loss     = avg_val_total
            best_state        = copy.deepcopy(model.state_dict())  # in-memory
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                model.load_state_dict(best_state)
                break

    else:
        model.load_state_dict(best_state)

    return model, history


def compute_loss_components(mu, log_var, y_true, q_k_z, p_k_x):
    """Returns NLL and KL as separate tensors for logging and weighted sum."""
    precision = torch.exp(-log_var)
    nll = 0.5 * (precision * (y_true - mu) ** 2 + log_var).mean()
    kl = F.kl_div(p_k_x.log(), q_k_z.detach(), reduction="batchmean")

    return nll, kl