import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm

# ---------------------------------------------------------
# GPU Device
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", DEVICE)

# =========================================================
# 1. DATA LOADING (YOU EDIT ONLY THIS FUNCTION)
# =========================================================
def load_data():
    """
    Replace this section with your Challenge dataset.

    MUST return:
        train_set, valid_set, test_set

    Each must be a torch Dataset that returns (X, y):
        X shape: (channels, time)
        y shape: scalar

    Remove the NotImplementedError and put your dataset code.
    """

    raise NotImplementedError("Replace load_data() content with your dataset loader.")

# Load the dataset
train_set, valid_set, test_set = load_data()

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False)


# =========================================================
# 2. MODELS
# =========================================================

# ---------------------- A. CNN BASELINE (EEGNet) ----------------------
class EEGNet(nn.Module):
    """Simplified EEGNet for regression"""
    def __init__(self, n_chans, n_times):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, (1, 7), padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (n_chans, 5), padding=(0, 2)),
            nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Linear(32 * n_times, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)           # (B, 1, chans, time)
        f = self.features(x)
        f = f.flatten(1)
        return self.regressor(f).squeeze(-1)


# ---------------------- B. Transformer BASELINE (EEGConformer-like) ----------------------
class SimpleEEGConformer(nn.Module):
    """Transformer encoder over time dimension"""
    def __init__(self, n_chans, n_times, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(n_chans, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x: (B, chans, time)
        x = x.permute(0, 2, 1)        # (B, time, chans)
        x = self.input_proj(x)        # (B, time, d_model)
        h = self.transformer(x)
        pooled = h.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


# ---------------------- C. DANN (CNN + Transformer + GRL) ----------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None

class GRL(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return GradReverse.apply(x, self.lambd)


class DANNModel(nn.Module):
    def __init__(self, n_chans, n_times, n_domains, d_model=64, lambd=0.5):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, (1, 7), padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (n_chans, 5), padding=(0, 2)),
            nn.ReLU()
        )

        # Transformer encoder
        self.proj = nn.Linear(32, d_model)
        encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)

        # Heads
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        self.grl = GRL(lambd=lambd)
        self.domain_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_domains)
        )

    def forward(self, x, return_domain=False):
        # x: (B, chans, time)
        x = x.unsqueeze(1)                  # (B,1,C,T)
        f = self.cnn(x).squeeze(2)          # (B,32,T)
        f = f.permute(0, 2, 1)              # (B,T,32)
        h = self.proj(f)                    # (B,T,d)
        h = self.transformer(h)
        emb = h.mean(dim=1)

        y = self.reg_head(emb).squeeze(-1)

        if not return_domain:
            return y

        d_emb = self.grl(emb)
        dom = self.domain_head(d_emb)
        return y, dom


# =========================================================
# 3. TRAINING UTILITIES
# =========================================================
def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2))

def train_regression(model, train_loader, valid_loader, epochs, lr):
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_rmse = float("inf")
    os.makedirs("results", exist_ok=True)
    name = model.__class__.__name__

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs} | Model: {name}")
        # --- Train ---
        model.train()
        tr_loss, tr_rmse = 0, 0
        n = 0
        for X, y in tqdm(train_loader, desc="Train"):
            X, y = X.to(DEVICE).float(), y.to(DEVICE).float().view(-1)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_rmse += rmse(pred, y).item()
            n += 1

        # --- Validate ---
        model.eval()
        va_loss, va_rmse = 0, 0
        n2 = 0
        with torch.no_grad():
            for X, y in tqdm(valid_loader, desc="Valid"):
                X, y = X.to(DEVICE).float(), y.to(DEVICE).float().view(-1)
                pred = model(X)
                loss = loss_fn(pred, y)
                va_loss += loss.item()
                va_rmse += rmse(pred, y).item()
                n2 += 1

        tr_loss /= n
        tr_rmse /= n
        va_loss /= n2
        va_rmse /= n2

        print(f"Train RMSE: {tr_rmse:.4f} | Valid RMSE: {va_rmse:.4f}")

        # --- Save best ---
        if va_rmse < best_rmse:
            best_rmse = va_rmse
            torch.save(model.state_dict(), f"results/{name}_best.pt")
            print(f"Saved best {name}: RMSE={best_rmse:.4f}")

    return best_rmse


# =========================================================
# 4. TRAIN ALL THREE MODELS
# =========================================================
# Infer sizes from one batch
sample_X, _ = next(iter(train_loader))
_, n_chans, n_times = sample_X.shape

print("Detected data shape:", sample_X.shape)

# ---------------- CNN ----------------
cnn = EEGNet(n_chans, n_times)
cnn_rmse = train_regression(cnn, train_loader, valid_loader, epochs=30, lr=1e-3)

# ---------------- Transformer ----------------
trans = SimpleEEGConformer(n_chans, n_times)
trans_rmse = train_regression(trans, train_loader, valid_loader, epochs=30, lr=1e-3)

# ---------------- DANN ----------------
domains = len({s for _, _, in train_set})
dann = DANNModel(n_chans, n_times, n_domains=domains, lambd=0.5)
# For DANN, use special loop

print("\nTraining DANN...")
dann = dann.to(DEVICE)
opt = AdamW(dann.parameters(), lr=1e-3)
dom_loss_fn = nn.CrossEntropyLoss()
reg_loss_fn = nn.MSELoss()

best_dann_rmse = float("inf")

for epoch in range(1, 31):
    print(f"\nEpoch {epoch}/30 | Model: DANN")
    dann.train()
    tr_rmse, n = 0, 0
    for X, y in tqdm(train_loader):
        X = X.to(DEVICE).float()
        y = y.to(DEVICE).float().view(-1)
        domains_batch = torch.zeros_like(y).long()  # TODO replace with actual subject IDs

        opt.zero_grad()
        y_hat, dom_logits = dann(X, return_domain=True)

        loss_reg = reg_loss_fn(y_hat, y)
        loss_dom = dom_loss_fn(dom_logits, domains_batch)
        loss = loss_reg + 0.5 * loss_dom

        loss.backward()
        opt.step()
        tr_rmse += rmse(y_hat, y).item()
        n += 1
    tr_rmse /= n

    # Validate
    dann.eval()
    va_rmse, n2 = 0, 0
    with torch.no_grad():
        for X, y in valid_loader:
            X = X.to(DEVICE).float()
            y = y.to(DEVICE).float().view(-1)
            pred = dann(X)
            va_rmse += rmse(pred, y).item()
            n2 += 1
    va_rmse /= n2

    print(f"Train RMSE={tr_rmse:.4f} | Valid RMSE={va_rmse:.4f}")

    if va_rmse < best_dann_rmse:
        best_dann_rmse = va_rmse
        torch.save(dann.state_dict(), "results/DANN_best.pt")
        print("Saved best DANN.")

print("\nALL TRAINING COMPLETE")
print("CNN Best RMSE:", cnn_rmse)
print("Transformer Best RMSE:", trans_rmse)
print("DANN Best RMSE:", best_dann_rmse)
