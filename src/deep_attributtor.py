import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

class DeepCausalMediation(nn.Module):
    def __init__(self, k_clusters=10, latent_dim=16, input_channels=1, spatial_dim=(64, 64)):
        super().__init__()
        self.K = k_clusters
        self.latent_dim = latent_dim
        
        # 1. Z-Encoder (2D CNN)
        # Input shape: (Batch, Channels, H, W)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 2. Clustering & Latent Space
        self.cluster_head = nn.Linear(32, k_clusters)
        self.cluster_embeddings = nn.Embedding(k_clusters, latent_dim)
        
        # 3. Mediator Predictor: P(B_k | X)
        self.mediator_model = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, k_clusters)
        )
        
        # 4. Outcome Model: P(Y | X, B_k) -> Outputs Mean and Log-Variance
        self.outcome_model = nn.Sequential(
            nn.Linear(1 + latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )

    def encode_z(self, z, tau=1.0, hard=False):
        """Encodes Z and returns soft cluster assignments and the latent vector."""
        z_feat = self.encoder(z)
        logits = self.cluster_head(z_feat)
        # Gumbel-Softmax for differentiable discrete clustering
        q_k_z = F.gumbel_softmax(logits, tau=tau, hard=hard)
        z_tilde = torch.matmul(q_k_z, self.cluster_embeddings.weight)
        return z_tilde, q_k_z

    def forward(self, x, z, tau=1.0):
        # Mediator path (from X)
        p_k_x_logits = self.mediator_model(x)
        p_k_x = F.softmax(p_k_x_logits, dim=-1)
        
        # Encoder path (from Z)
        z_tilde, q_k_z = self.encode_z(z, tau=tau)
        
        # Outcome path (from X and Z_latent)
        y_input = torch.cat([x, z_tilde], dim=-1)
        y_params = self.outcome_model(y_input)
        mu, log_var = y_params[:, 0:1], y_params[:, 1:2]
        
        return mu, log_var, q_k_z, p_k_x
    
def compute_loss(mu, log_var, y_true, q_k_z, p_k_x, lambda_kl=1.0):
    # 1. Gaussian NLL for Y
    precision = torch.exp(-log_var)
    nll = 0.5 * (precision * (y_true - mu)**2 + log_var).mean()
    
    # 2. KL Divergence for Mediation: forces P(B|X) to match Q(B|Z)
    # Note: p_k_x is predicted from X, q_k_z is 'ground truth' from Z
    kl_loss = F.kl_div(p_k_x.log(), q_k_z, reduction='batchmean')
    
    return nll + lambda_kl * kl_loss

@torch.no_grad()
def estimate_pn(model, x_c, x_f, threshold, device='cpu'):
    """
    Computes PN = 1 - 1/RR
    where RR = P(Y > th | do(x_c)) / P(Y > th | do(x_f))
    """
    model.eval()
    
    def get_p_y_do_x(x_val_scalar):
        x_t = torch.tensor([[x_val_scalar]], dtype=torch.float32).to(device)
        
        # Get cluster distribution from Mediator model
        p_k_x = F.softmax(model.mediator_model(x_t), dim=-1).squeeze()
        
        prob_exceed_total = 0.0
        # Marginalize over all possible clusters K
        for k in range(model.K):
            emb = model.cluster_embeddings.weight[k:k+1]
            y_params = model.outcome_model(torch.cat([x_t, emb], dim=-1))
            mu, sigma = y_params[:, 0], torch.exp(0.5 * y_params[:, 1])
            
            # Normal CDF for P(Y > threshold)
            dist = Normal(mu, sigma)
            p_y_gt_th = 1.0 - dist.cdf(torch.tensor([threshold]).to(device))
            prob_exceed_total += p_y_gt_th.item() * p_k_x[k].item()
            
        return prob_exceed_total

    p_c = get_p_y_do_x(x_c)
    p_f = get_p_y_do_x(x_f)
    
    return p_c, p_f
    

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
    lambda_kl=1.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # --- 1. Data Preparation ---
    # Ensure shapes are (N, 1) for X, Y and (N, 1, H, W) for Z
    X_t = torch.tensor(X, dtype=torch.float32).view(-1, 1)
    Y_t = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
    Z_t = torch.tensor(Z, dtype=torch.float32).unsqueeze(1) # Adds channel dim
    
    dataset = TensorDataset(X_t, Y_t, Z_t)
    
    # Split for Validation
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split = int(np.floor(val_split * num_samples))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))
    
    # --- 2. Early Stopping & Annealing Setup ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    tau = tau_start
    tau_decay = (tau_start - tau_min) / (epochs * 0.7) # Decay over 70% of epochs
    
    torch.save(model.state_dict(), "models/best_causal_model.pt")
    
    # print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch_x, batch_y, batch_z in train_loader:
            batch_x, batch_y, batch_z = batch_x.to(device), batch_y.to(device), batch_z.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            mu, log_var, q_k_z, p_k_x = model(batch_x, batch_z, tau=tau)
            
            # Loss computation
            loss = compute_loss(mu, log_var, batch_y, q_k_z, p_k_x, lambda_kl=lambda_kl)
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        # --- 3. Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y, batch_z in val_loader:
                batch_x, batch_y, batch_z = batch_x.to(device), batch_y.to(device), batch_z.to(device)
                mu, log_var, q_k_z, p_k_x = model(batch_x, batch_z, tau=tau)
                loss = compute_loss(mu, log_var, batch_y, q_k_z, p_k_x, lambda_kl=lambda_kl)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Anneal Temperature
        tau = max(tau_min, tau - tau_decay)
        
        # if (epoch + 1) % 10 == 0 or epoch == 0:
            # print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Tau: {tau:.2f}")

        # --- 4. Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/best_causal_model.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                # print(f"Early stopping triggered at epoch {epoch+1}")
                model.load_state_dict(torch.load("models/best_causal_model.pt"))
                break
                
    return model