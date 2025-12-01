import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

# Braindecode / EEGDash imports (same stack you already use)
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

# PyG imports
from torch_geometric.nn import GCNConv, GATConv


# ============================================================
# CONFIG & RESULTS FOLDER
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", DEVICE)

EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3

EPOCH_LEN_S = 2.0
SFREQ = 100
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0
ANCHOR = "stimulus_anchor"

# This mirrors your Pioneer path layout: data_merged/release_1 under the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data_merged"
RELEASE_ID = 1
RELEASE_DIR = DATA_DIR / f"release_{RELEASE_ID}"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = PROJECT_ROOT / f"gnn_results_{TIMESTAMP}"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\n📁 All GNN results will be saved to: {RESULTS_DIR}/")
print("=" * 70)


# ============================================================
# 1. GRAPH CONSTRUCTION
# ============================================================
def build_functional_connectivity_graph(eeg_data, threshold=0.3):
    """
    Build graph from functional connectivity (correlation) between channels.

    Args:
        eeg_data: (n_samples, 129, 200) numpy array
        threshold: |corr| threshold for edges
    """
    eeg_channels = eeg_data[:, :128, :]  # (N, 128, 200)

    # Concatenate time across samples: (N, 128, 200) -> (N*200, 128)
    signals = eeg_channels.transpose(0, 2, 1).reshape(-1, 128)

    corr_matrix = np.corrcoef(signals.T)  # (128, 128)

    adj_matrix = np.abs(corr_matrix) > threshold
    np.fill_diagonal(adj_matrix, False)
    adj_matrix = np.triu(adj_matrix, k=1)

    edge_index = np.array(np.where(adj_matrix))
    edge_weights = corr_matrix[adj_matrix]

    num_edges = edge_index.shape[1]
    num_possible = 128 * 127 // 2

    print("📊 Functional Connectivity Graph")
    print(f"  threshold       : {threshold}")
    print(f"  edges           : {num_edges:,} / {num_possible:,}")
    print(f"  edge density    : {num_edges / num_possible:.2%}")
    print(f"  mean |corr|     : {np.abs(edge_weights).mean():.3f}")
    print(f"  max  |corr|     : {np.abs(edge_weights).max():.3f}")

    return edge_index, edge_weights, corr_matrix


def convert_to_pytorch(edge_index, edge_weights=None):
    edge_index_t = torch.from_numpy(edge_index).long()
    if edge_weights is not None:
        edge_weights_t = torch.from_numpy(edge_weights).float()
        return edge_index_t, edge_weights_t
    return edge_index_t, None


# ============================================================
# 2. DATA LOADING (data_merged on Pioneer)
# ============================================================
def load_windows_and_targets():
    print("=" * 70)
    print("Loading EEG windows and targets (data_merged / release_1)")
    print("=" * 70)

    print(f"📁 Loading from: {RELEASE_DIR}")
    if not RELEASE_DIR.exists():
        raise FileNotFoundError(f"Release dir not found: {RELEASE_DIR}")

    dataset_ccd = EEGChallengeDataset(
        task="contrastChangeDetection",
        release=f"R{RELEASE_ID}",
        cache_dir=RELEASE_DIR,
        mini=False,
        download=False,
    )
    print(f"✅ Loaded {len(dataset_ccd.datasets)} recordings")

    transformation_offline = [
        Preprocessor(
            annotate_trials_with_target,
            target_field="rt_from_stimulus",
            epoch_length=EPOCH_LEN_S,
            require_stimulus=True,
            require_response=True,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]
    preprocess(dataset_ccd, transformation_offline, n_jobs=1)

    dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)

    single_windows = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )

    single_windows = add_extras_columns(
        single_windows,
        dataset,
        desc=ANCHOR,
        keys=(
            "target",
            "rt_from_stimulus",
            "rt_from_trialstart",
            "stimulus_onset",
            "response_onset",
            "correct",
            "response_type",
        ),
    )

    meta = single_windows.get_metadata()
    if "rt_from_stimulus" not in meta.columns:
        raise RuntimeError("rt_from_stimulus not found in metadata.")

    targets = meta["rt_from_stimulus"].values.astype(float)

    print(f"✅ Created {len(single_windows)} windows")
    print(
        f"RT stats: min={np.nanmin(targets):.2f}, max={np.nanmax(targets):.2f}, "
        f"mean={np.nanmean(targets):.2f}"
    )

    return single_windows, targets, meta


def group_by_subject(single_windows, meta):
    print("\nGrouping windows by subject...")
    if "subject" not in meta.columns:
        raise RuntimeError("No 'subject' column in metadata.")

    subjects = meta["subject"].values
    from collections import defaultdict

    subj_to_indices = defaultdict(list)
    for idx in range(len(single_windows)):
        subj = subjects[idx]
        subj = str(subj) if not pd.isna(subj) else f"unknown_{idx}"
        subj_to_indices[subj].append(idx)

    print(f"✅ Found {len(subj_to_indices)} subjects, {len(single_windows)} windows.")
    return subj_to_indices


def split_by_subject(subj_to_indices, val_frac=0.1, test_frac=0.1, seed=2025):
    all_subjects = list(subj_to_indices.keys())
    rng = check_random_state(seed)

    # First split off test+val
    train_subj, temp_subj = train_test_split(
        all_subjects, test_size=(val_frac + test_frac), random_state=rng
    )
    val_subj, test_subj = train_test_split(
        temp_subj,
        test_size=test_frac / (val_frac + test_frac),
        random_state=rng,
    )

    def collect_indices(subj_list):
        idxs = []
        for s in subj_list:
            idxs.extend(subj_to_indices[s])
        return sorted(idxs)

    train_idx = collect_indices(train_subj)
    val_idx = collect_indices(val_subj)
    test_idx = collect_indices(test_subj)

    print("\nSubject-level split:")
    print(f"  Train subjects: {len(train_subj)}  windows: {len(train_idx)}")
    print(f"  Val subjects  : {len(val_subj)}  windows: {len(val_idx)}")
    print(f"  Test subjects : {len(test_subj)}  windows: {len(test_idx)}")

    assert not set(train_subj) & set(val_subj)
    assert not set(train_subj) & set(test_subj)
    assert not set(val_subj) & set(test_subj)

    return train_idx, val_idx, test_idx


class EEGWindowDataset(Dataset):
    def __init__(self, windows, targets, indices):
        self.windows = windows
        self.targets = targets
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        win_idx = self.indices[idx]
        X = self.windows[win_idx][0]  # (129, 200), tensor or np
        y = self.targets[win_idx]
        if isinstance(X, torch.Tensor):
            X = X.float()
        else:
            X = torch.from_numpy(X).float()
        y = torch.tensor(y, dtype=torch.float32)
        return X, y, win_idx


# ============================================================
# 3. MODEL (Dual-Branch GRU + GNN)
# ============================================================
class DualBranchEEGModel(nn.Module):
    def __init__(
        self,
        n_channels=128,
        n_times=200,
        gru_hidden_dim=64,
        gru_num_layers=2,
        gnn_hidden_dim=64,
        gnn_num_layers=2,
        use_gat=False,
        fusion_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times
        self.use_gat = use_gat

        # GRU branch
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.gru_proj = nn.Linear(gru_hidden_dim * 2, gru_hidden_dim)

        # GNN branch
        if use_gat:
            heads = 4
            self.gnn_layers = nn.ModuleList()
            self.gnn_layers.append(GATConv(n_times, gnn_hidden_dim, heads=heads, dropout=dropout))
            for _ in range(gnn_num_layers - 1):
                self.gnn_layers.append(
                    GATConv(gnn_hidden_dim * heads, gnn_hidden_dim, heads=heads, dropout=dropout)
                )
            self.gat_final_proj = nn.Linear(gnn_hidden_dim * heads, gnn_hidden_dim)
            self.gnn_bn = nn.ModuleList(
                [nn.BatchNorm1d(gnn_hidden_dim * heads) for _ in range(gnn_num_layers)]
            )
        else:
            self.gnn_layers = nn.ModuleList()
            self.gnn_layers.append(GCNConv(n_times, gnn_hidden_dim))
            for _ in range(gnn_num_layers - 1):
                self.gnn_layers.append(GCNConv(gnn_hidden_dim, gnn_hidden_dim))
            self.gnn_bn = nn.ModuleList(
                [nn.BatchNorm1d(gnn_hidden_dim) for _ in range(gnn_num_layers)]
            )

        # Fusion + prediction
        self.fusion = nn.Sequential(
            nn.Linear(gru_hidden_dim + gnn_hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weights=None):
        # x: (B, 129, 200)
        B = x.size(0)
        eeg = x[:, :128, :]  # (B, 128, 200)

        # GRU branch
        eeg_time = eeg.reshape(B * self.n_channels, self.n_times, 1)
        gru_out, _ = self.gru(eeg_time)
        gru_last = gru_out[:, -1, :]  # (B*128, 2*hidden)
        gru_feat = self.gru_proj(gru_last)
        gru_feat = self.dropout(gru_feat)
        gru_feat = gru_feat.reshape(B, self.n_channels, -1)  # (B, 128, H_gru)

        # GNN branch: node features are time-series (length n_times)
        node_features = eeg.reshape(B * self.n_channels, self.n_times)
        gnn_feat = node_features
        if self.use_gat:
            for i, (layer, bn) in enumerate(zip(self.gnn_layers, self.gnn_bn)):
                gnn_feat = layer(gnn_feat, edge_index)
                gnn_feat = bn(gnn_feat)
                gnn_feat = torch.relu(gnn_feat)
                if i < len(self.gnn_layers) - 1:
                    gnn_feat = self.dropout(gnn_feat)
            gnn_feat = self.gat_final_proj(gnn_feat)
        else:
            for i, (layer, bn) in enumerate(zip(self.gnn_layers, self.gnn_bn)):
                gnn_feat = layer(gnn_feat, edge_index, edge_weights)
                gnn_feat = bn(gnn_feat)
                gnn_feat = torch.relu(gnn_feat)
                if i < len(self.gnn_layers) - 1:
                    gnn_feat = self.dropout(gnn_feat)

        gnn_feat = gnn_feat.reshape(B, self.n_channels, -1)  # (B, 128, H_gnn)

        # Fusion
        fused = torch.cat([gru_feat, gnn_feat], dim=-1)
        fused = self.fusion(fused)  # (B, 128, F)
        pooled = fused.mean(dim=1)  # (B, F)

        out = self.predictor(pooled)
        return out.squeeze(-1)


# ============================================================
# 4. TRAINING UTILITIES (checkpoints, error distributions, curves)
# ============================================================
def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2))


def train_one_epoch(model, loader, optimizer, edge_index, edge_weights):
    model.train()
    total_rmse = 0.0
    total_loss = 0.0
    steps = 0
    for X, y, _ in tqdm(loader, desc="Train"):
        X = X.to(DEVICE).float()
        y = y.to(DEVICE).float().view(-1)

        optimizer.zero_grad()
        pred = model(X, edge_index, edge_weights)
        loss = nn.MSELoss()(pred, y)
        loss.backward()
        optimizer.step()

        total_rmse += rmse(pred, y).item()
        total_loss += loss.item()
        steps += 1

    return total_rmse / steps, total_loss / steps


def eval_epoch(model, loader, edge_index, edge_weights):
    model.eval()
    total_rmse = 0.0
    steps = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y, _ in tqdm(loader, desc="Valid"):
            X = X.to(DEVICE).float()
            y = y.to(DEVICE).float().view(-1)
            pred = model(X, edge_index, edge_weights)
            total_rmse += rmse(pred, y).item()
            steps += 1
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return total_rmse / steps, all_preds, all_targets


def analyze_error_distribution(errors, name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    errors = np.array(errors).flatten()

    stats_dict = {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "median": float(np.median(errors)),
        "q25": float(np.percentile(errors, 25)),
        "q75": float(np.percentile(errors, 75)),
        "skewness": float(stats.skew(errors)),
        "kurtosis": float(stats.kurtosis(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
    }

    with open(save_dir / f"{name}_error_stats.json", "w") as f:
        json.dump(stats_dict, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram
    axes[0, 0].hist(errors, bins=50, alpha=0.7, edgecolor="black")
    axes[0, 0].axvline(0, color="r", linestyle="--", label="Zero error")
    axes[0, 0].set_xlabel("Error (pred - true)")
    axes[0, 0].set_ylabel("Freq")
    axes[0, 0].set_title(f"{name} - Error Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # QQ plot
    stats.probplot(errors, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title(f"{name} - Q-Q Plot")
    axes[0, 1].grid(True, alpha=0.3)

    # Boxplot
    axes[1, 0].boxplot(errors, vert=True)
    axes[1, 0].set_ylabel("Error")
    axes[1, 0].set_title(f"{name} - Boxplot")
    axes[1, 0].grid(True, alpha=0.3)

    # CDF
    s = np.sort(errors)
    axes[1, 1].plot(s, np.arange(len(s)) / len(s))
    axes[1, 1].set_xlabel("Error")
    axes[1, 1].set_ylabel("CDF")
    axes[1, 1].set_title(f"{name} - CDF")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f"{name}_error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    return stats_dict


def train_gnn():
    # 1) Load windows/targets from data_merged path
    single_windows, targets, meta = load_windows_and_targets()

    # 2) Build functional connectivity graph using a subset (for speed)
    #    Use up to 2000 windows to estimate correlations
    max_samples_for_graph = min(2000, len(single_windows))
    sample_stack = []
    for i in range(max_samples_for_graph):
        X = single_windows[i][0]
        X = X.numpy() if isinstance(X, torch.Tensor) else X
        sample_stack.append(X)
    sample_stack = np.stack(sample_stack, axis=0)  # (N, 129, 200)

    edge_index_np, edge_weights_np, _ = build_functional_connectivity_graph(
        sample_stack, threshold=0.3
    )
    edge_index_t, edge_weights_t = convert_to_pytorch(edge_index_np, edge_weights_np)
    edge_index_t = edge_index_t.to(DEVICE)
    edge_weights_t = edge_weights_t.to(DEVICE)

    # 3) Subject split and datasets
    subj_to_indices = group_by_subject(single_windows, meta)
    train_idx, val_idx, test_idx = split_by_subject(subj_to_indices)

    train_ds = EEGWindowDataset(single_windows, targets, train_idx)
    val_ds = EEGWindowDataset(single_windows, targets, val_idx)
    test_ds = EEGWindowDataset(single_windows, targets, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4) Model, optimizer
    model = DualBranchEEGModel(
        n_channels=128,
        n_times=200,
        gru_hidden_dim=64,
        gnn_hidden_dim=64,
        use_gat=False,  # functional connectivity + GCN
        dropout=0.3,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)

    # Results subdir
    model_dir = RESULTS_DIR / "DualBranch_GNN"
    os.makedirs(model_dir, exist_ok=True)

    best_rmse = float("inf")
    best_epoch = 0
    train_rmse_hist = []
    val_rmse_hist = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        tr_rmse, tr_loss = train_one_epoch(model, train_loader, optimizer, edge_index_t, edge_weights_t)
        va_rmse, va_preds, va_targets = eval_epoch(model, val_loader, edge_index_t, edge_weights_t)

        print(f"Train RMSE={tr_rmse:.4f} | Val RMSE={va_rmse:.4f}")
        train_rmse_hist.append(tr_rmse)
        val_rmse_hist.append(va_rmse)

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_rmse": tr_rmse,
                "val_rmse": va_rmse,
            }
            torch.save(ckpt, model_dir / f"checkpoint_epoch_{epoch}.pt")
            print(f"Saved checkpoint at epoch {epoch}")

        # Best model
        if va_rmse < best_rmse:
            best_rmse = va_rmse
            best_epoch = epoch
            torch.save(model.state_dict(), model_dir / "best_model.pt")
            print(f"Saved best model (epoch {epoch}, RMSE={best_rmse:.4f})")

    # Save training history & curves
    history = {
        "epochs": EPOCHS,
        "train_rmse": train_rmse_hist,
        "val_rmse": val_rmse_hist,
        "best_rmse": best_rmse,
        "best_epoch": best_epoch,
    }
    with open(model_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Curves
    epochs_axis = list(range(1, EPOCHS + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, train_rmse_hist, label="Train RMSE", marker="o")
    plt.plot(epochs_axis, val_rmse_hist, label="Val RMSE", marker="s")
    plt.axvline(best_epoch, color="r", linestyle="--", label=f"Best (epoch {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("DualBranch GNN - Training / Validation RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Final evaluation on test set + error distributions
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=DEVICE))
    test_rmse, test_preds, test_targets = eval_epoch(model, test_loader, edge_index_t, edge_weights_t)
    print(f"\nTEST RMSE (best model @ epoch {best_epoch}): {test_rmse:.4f}")

    # Save predictions/targets
    np.save(model_dir / "val_predictions.npy", va_preds)
    np.save(model_dir / "val_targets.npy", va_targets)
    np.save(model_dir / "test_predictions.npy", test_preds)
    np.save(model_dir / "test_targets.npy", test_targets)

    val_errors = va_preds - va_targets
    test_errors = test_preds - test_targets

    analyze_error_distribution(val_errors, "DualBranch_GNN_val", model_dir)
    analyze_error_distribution(test_errors, "DualBranch_GNN_test", model_dir)

    print("\nAll done. Check results under:")
    print(f"  {model_dir}")


if __name__ == "__main__":
    train_gnn()


