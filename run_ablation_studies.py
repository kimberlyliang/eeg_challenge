"""
Ablation Studies Script - Overnight Run
Actually runs ablation experiments on different hyperparameters.
"""

import os
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import json
import matplotlib.pyplot as plt
from datetime import datetime
import copy

# Braindecode imports
from braindecode.models import EEGNetv4
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events

# EEGDash imports
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", DEVICE)

# Reduced epochs for faster ablation studies
ABLATION_EPOCHS = 20
BATCH_SIZE_BASE = 64
LR_BASE = 1e-3

# Data configuration (same as main script)
EPOCH_LEN_S = 2.0
SFREQ = 100
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0
ANCHOR = "stimulus_anchor"

# Results directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ABLATION_DIR = f"ablation_results_{TIMESTAMP}"
os.makedirs(ABLATION_DIR, exist_ok=True)

print(f"\n📁 Ablation results will be saved to: {ABLATION_DIR}/")
print("=" * 70)

# Ablation configurations to test
ABLATION_CONFIG = {
    'learning_rate': {
        'enabled': True,
        'values': [5e-4, 1e-3, 2e-3, 5e-3],
    },
    'batch_size': {
        'enabled': True,
        'values': [32, 64, 128],
    },
    'dropout_rate': {
        'enabled': True,
        'values': [0.0, 0.1, 0.3, 0.5],
    },
    'transformer_layers': {
        'enabled': True,
        'values': [1, 2, 3, 4],
    },
    'transformer_heads': {
        'enabled': True,
        'values': [2, 4, 8],
    },
    'd_model': {
        'enabled': True,
        'values': [32, 64, 128],
    },
}

# ============================================================
# DATA LOADING (Reuse from main script)
# ============================================================
print("=" * 70)
print("Loading Dataset")
print("=" * 70)

data_dir = Path("data")
available_releases = []

if data_dir.exists():
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.startswith("release_"):
            release_num = item.name.split("_")[1]
            available_releases.append(int(release_num))

available_releases.sort()
print(f"Available releases: {available_releases}")

if not available_releases:
    raise FileNotFoundError("No release folders found in data/")

# Load datasets from all available releases
print("\n🔄 Loading datasets from all available releases...")
all_release_datasets = []

for release_id in available_releases:
    release_dir = Path(f"data/release_{release_id}")
    if not release_dir.exists():
        continue
    
    try:
        dataset = EEGChallengeDataset(
            task="contrastChangeDetection",
            release=f"R{release_id}",
            cache_dir=release_dir,
            mini=False
        )
        if len(dataset.datasets) > 0:
            all_release_datasets.append(dataset)
            print(f"   ✅ Loaded {len(dataset.datasets)} recordings from Release R{release_id}")
    except Exception as e:
        print(f"   ❌ Failed to load Release R{release_id}: {str(e)[:100]}")
        continue

if not all_release_datasets:
    raise ValueError("No datasets could be loaded from any release!")

dataset_ccd = BaseConcatDataset(all_release_datasets)
total_recordings = sum(len(ds.datasets) for ds in all_release_datasets)
print(f"✅ Combined dataset: {total_recordings} total recordings")

# Preprocessing
print("\nPreprocessing...")
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
    keys=("target", "rt_from_stimulus", "rt_from_trialstart",
          "stimulus_onset", "response_onset", "correct", "response_type")
)

# Train/Val/Test Split
meta_information = single_windows.get_metadata()
valid_frac = 0.1
test_frac = 0.1
seed = 2025

subjects = meta_information["subject"].unique()
rng = check_random_state(seed)

train_subj, temp_subj = train_test_split(
    subjects, test_size=(valid_frac + test_frac), random_state=rng
)
valid_subj, test_subj = train_test_split(
    temp_subj, test_size=test_frac / (valid_frac + test_frac), random_state=rng
)

subject_split = single_windows.split("subject")

train_set = []
valid_set = []
test_set = []

for s in subject_split:
    if s in train_subj:
        train_set.append(subject_split[s])
    elif s in valid_subj:
        valid_set.append(subject_split[s])
    elif s in test_subj:
        test_set.append(subject_split[s])

train_set = BaseConcatDataset(train_set)
valid_set = BaseConcatDataset(valid_set)
test_set = BaseConcatDataset(test_set)

print(f"Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")

# Detect dimensions
sample_X, _, _ = next(iter(DataLoader(train_set, batch_size=1)))
_, n_chans, n_times = sample_X.shape
print(f"Input shape: n_chans={n_chans}, n_times={n_times}")

# ============================================================
# MODEL ARCHITECTURES
# ============================================================
class CNNOnly(nn.Module):
    """CNN-only baseline"""
    def __init__(self, n_chans, n_times, n_outputs=1):
        super().__init__()
        self.cnn = EEGNetv4(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
    
    def forward(self, x):
        return self.cnn(x).squeeze(-1)


class CNNTransformer(nn.Module):
    """CNN + Transformer with configurable dropout"""
    def __init__(self, n_chans, n_times, n_outputs=1, d_model=64, nhead=4, 
                 num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.cnn = EEGNetv4(n_chans=n_chans, n_times=n_times, n_outputs=d_model)
        if hasattr(self.cnn, 'classifier'):
            self.cnn.classifier = nn.Identity()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, n_outputs)
        )
    
    def forward(self, x):
        cnn_feat = self.cnn(x).unsqueeze(1)
        trans_out = self.transformer(cnn_feat)
        pooled = trans_out.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


# ============================================================
# TRAINING UTILITIES
# ============================================================
def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2))


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_rmse = 0
    steps = 0
    for X, y, _ in tqdm(loader, desc="Train", leave=False):
        X = X.to(DEVICE).float()
        y = y.to(DEVICE).float().view(-1)
        optimizer.zero_grad()
        pred = model(X).view(-1)
        loss = nn.MSELoss()(pred, y)
        loss.backward()
        optimizer.step()
        total_rmse += rmse(pred, y).item()
        steps += 1
    return total_rmse / steps


def eval_epoch(model, loader):
    model.eval()
    total_rmse = 0
    steps = 0
    with torch.no_grad():
        for X, y, _ in tqdm(loader, desc="Valid", leave=False):
            X = X.to(DEVICE).float()
            y = y.to(DEVICE).float().view(-1)
            pred = model(X).view(-1)
            total_rmse += rmse(pred, y).item()
            steps += 1
    return total_rmse / steps


def train_ablation_model(name, model, train_loader, valid_loader, epochs, lr):
    """Simplified training for ablation studies with full history"""
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    best_rmse = float("inf")
    best_epoch = 0
    train_rmses = []
    val_rmses = []
    train_losses = []
    
    # Create model directory
    model_dir = os.path.join(ABLATION_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        tr_rmse = train_one_epoch(model, train_loader, optimizer)
        va_rmse = eval_epoch(model, valid_loader)
        
        train_rmses.append(tr_rmse)
        val_rmses.append(va_rmse)
        
        if va_rmse < best_rmse:
            best_rmse = va_rmse
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            torch.save(model.state_dict(), os.path.join(ABLATION_DIR, f"{name}_best.pt"))
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_rmse': tr_rmse,
                'val_rmse': va_rmse,
            }
            torch.save(checkpoint, os.path.join(model_dir, f"checkpoint_epoch_{epoch}.pt"))
    
    # Save training history
    history = {
        'train_rmses': train_rmses,
        'val_rmses': val_rmses,
        'best_rmse': best_rmse,
        'best_epoch': best_epoch,
        'epochs': epochs,
        'learning_rate': lr,
    }
    with open(os.path.join(model_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        epochs_list = list(range(1, epochs + 1))
        ax.plot(epochs_list, train_rmses, label='Train RMSE', marker='o', markersize=3)
        ax.plot(epochs_list, val_rmses, label='Val RMSE', marker='s', markersize=3)
        ax.axvline(best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{name} - Training Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "training_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save training curves for {name}: {e}")
    
    return best_rmse, train_rmses, val_rmses, best_epoch


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================
def run_ablation_learning_rate():
    """Ablation: Learning rate variations"""
    print("\n" + "="*70)
    print("ABLATION: Learning Rate")
    print("="*70)
    
    results = []
    batch_size = BATCH_SIZE_BASE
    
    for lr in ABLATION_CONFIG['learning_rate']['values']:
        exp_name = f"CNN_only_lr_{lr}"
        print(f"\n--- Testing LR={lr} ---")
        
        model = CNNOnly(n_chans=n_chans, n_times=n_times)
        train_loader_ablation = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader_ablation = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        test_loader_ablation = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        best_val_rmse, train_rmses, val_rmses, best_epoch = train_ablation_model(
            exp_name, model, train_loader_ablation, valid_loader_ablation,
            ABLATION_EPOCHS, lr
        )
        
        # Test evaluation
        model.load_state_dict(torch.load(os.path.join(ABLATION_DIR, f"{exp_name}_best.pt")))
        test_rmse = eval_epoch(model, test_loader_ablation)
        
        results.append({
            'experiment_name': exp_name,
            'learning_rate': lr,
            'best_val_rmse': float(best_val_rmse),
            'test_rmse': float(test_rmse),
            'best_epoch': int(best_epoch),
            'final_train_rmse': float(train_rmses[-1]),
            'final_val_rmse': float(val_rmses[-1]),
            'train_rmses': [float(x) for x in train_rmses],
            'val_rmses': [float(x) for x in val_rmses],
        })
        print(f"  Val RMSE: {best_val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    
    return results


def run_ablation_batch_size():
    """Ablation: Batch size variations"""
    print("\n" + "="*70)
    print("ABLATION: Batch Size")
    print("="*70)
    
    results = []
    lr = LR_BASE
    
    for batch_size in ABLATION_CONFIG['batch_size']['values']:
        exp_name = f"CNN_only_bs_{batch_size}"
        print(f"\n--- Testing Batch Size={batch_size} ---")
        
        model = CNNOnly(n_chans=n_chans, n_times=n_times)
        train_loader_ablation = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader_ablation = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        test_loader_ablation = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        best_val_rmse, train_rmses, val_rmses, best_epoch = train_ablation_model(
            exp_name, model, train_loader_ablation, valid_loader_ablation,
            ABLATION_EPOCHS, lr
        )
        
        # Test evaluation
        model.load_state_dict(torch.load(os.path.join(ABLATION_DIR, f"{exp_name}_best.pt")))
        test_rmse = eval_epoch(model, test_loader_ablation)
        
        results.append({
            'experiment_name': exp_name,
            'batch_size': batch_size,
            'best_val_rmse': float(best_val_rmse),
            'test_rmse': float(test_rmse),
            'best_epoch': int(best_epoch),
            'final_train_rmse': float(train_rmses[-1]),
            'final_val_rmse': float(val_rmses[-1]),
            'train_rmses': [float(x) for x in train_rmses],
            'val_rmses': [float(x) for x in val_rmses],
        })
        print(f"  Val RMSE: {best_val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    
    return results


def run_ablation_dropout_rate():
    """Ablation: Dropout rate in model"""
    print("\n" + "="*70)
    print("ABLATION: Dropout Rate")
    print("="*70)
    
    results = []
    lr = LR_BASE
    batch_size = BATCH_SIZE_BASE
    
    for dropout_rate in ABLATION_CONFIG['dropout_rate']['values']:
        exp_name = f"CNN_Transformer_dropout_{dropout_rate}"
        print(f"\n--- Testing Dropout={dropout_rate} ---")
        
        model = CNNTransformer(n_chans=n_chans, n_times=n_times, n_outputs=1, 
                              dropout_rate=dropout_rate)
        train_loader_ablation = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader_ablation = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        test_loader_ablation = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        best_val_rmse, train_rmses, val_rmses, best_epoch = train_ablation_model(
            exp_name, model, train_loader_ablation, valid_loader_ablation,
            ABLATION_EPOCHS, lr
        )
        
        # Test evaluation
        model.load_state_dict(torch.load(os.path.join(ABLATION_DIR, f"{exp_name}_best.pt")))
        test_rmse = eval_epoch(model, test_loader_ablation)
        
        results.append({
            'experiment_name': exp_name,
            'dropout_rate': dropout_rate,
            'best_val_rmse': float(best_val_rmse),
            'test_rmse': float(test_rmse),
            'best_epoch': int(best_epoch),
            'final_train_rmse': float(train_rmses[-1]),
            'final_val_rmse': float(val_rmses[-1]),
            'train_rmses': [float(x) for x in train_rmses],
            'val_rmses': [float(x) for x in val_rmses],
        })
        print(f"  Val RMSE: {best_val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    
    return results


def run_ablation_transformer_layers():
    """Ablation: Number of transformer layers"""
    print("\n" + "="*70)
    print("ABLATION: Transformer Layers")
    print("="*70)
    
    results = []
    lr = LR_BASE
    batch_size = BATCH_SIZE_BASE
    dropout_rate = 0.5
    
    for num_layers in ABLATION_CONFIG['transformer_layers']['values']:
        exp_name = f"CNN_Transformer_layers_{num_layers}"
        print(f"\n--- Testing {num_layers} Transformer Layers ---")
        
        model = CNNTransformer(n_chans=n_chans, n_times=n_times, n_outputs=1,
                              dropout_rate=dropout_rate, num_layers=num_layers)
        train_loader_ablation = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader_ablation = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        test_loader_ablation = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        best_val_rmse, train_rmses, val_rmses, best_epoch = train_ablation_model(
            exp_name, model, train_loader_ablation, valid_loader_ablation,
            ABLATION_EPOCHS, lr
        )
        
        model.load_state_dict(torch.load(os.path.join(ABLATION_DIR, f"{exp_name}_best.pt")))
        test_rmse = eval_epoch(model, test_loader_ablation)
        
        results.append({
            'experiment_name': exp_name,
            'num_layers': num_layers,
            'best_val_rmse': float(best_val_rmse),
            'test_rmse': float(test_rmse),
            'best_epoch': int(best_epoch),
            'final_train_rmse': float(train_rmses[-1]),
            'final_val_rmse': float(val_rmses[-1]),
            'train_rmses': [float(x) for x in train_rmses],
            'val_rmses': [float(x) for x in val_rmses],
        })
        print(f"  Val RMSE: {best_val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    
    return results


def run_ablation_transformer_heads():
    """Ablation: Number of attention heads"""
    print("\n" + "="*70)
    print("ABLATION: Transformer Attention Heads")
    print("="*70)
    
    results = []
    lr = LR_BASE
    batch_size = BATCH_SIZE_BASE
    dropout_rate = 0.5
    
    for nhead in ABLATION_CONFIG['transformer_heads']['values']:
        exp_name = f"CNN_Transformer_heads_{nhead}"
        print(f"\n--- Testing {nhead} Attention Heads ---")
        
        model = CNNTransformer(n_chans=n_chans, n_times=n_times, n_outputs=1,
                              dropout_rate=dropout_rate, nhead=nhead)
        train_loader_ablation = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader_ablation = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        test_loader_ablation = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        best_val_rmse, train_rmses, val_rmses, best_epoch = train_ablation_model(
            exp_name, model, train_loader_ablation, valid_loader_ablation,
            ABLATION_EPOCHS, lr
        )
        
        model.load_state_dict(torch.load(os.path.join(ABLATION_DIR, f"{exp_name}_best.pt")))
        test_rmse = eval_epoch(model, test_loader_ablation)
        
        results.append({
            'experiment_name': exp_name,
            'nhead': nhead,
            'best_val_rmse': float(best_val_rmse),
            'test_rmse': float(test_rmse),
            'best_epoch': int(best_epoch),
            'final_train_rmse': float(train_rmses[-1]),
            'final_val_rmse': float(val_rmses[-1]),
            'train_rmses': [float(x) for x in train_rmses],
            'val_rmses': [float(x) for x in val_rmses],
        })
        print(f"  Val RMSE: {best_val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    
    return results


def run_ablation_d_model():
    """Ablation: Transformer dimension"""
    print("\n" + "="*70)
    print("ABLATION: Transformer Dimension (d_model)")
    print("="*70)
    
    results = []
    lr = LR_BASE
    batch_size = BATCH_SIZE_BASE
    dropout_rate = 0.5
    
    for d_model in ABLATION_CONFIG['d_model']['values']:
        exp_name = f"CNN_Transformer_dmodel_{d_model}"
        print(f"\n--- Testing d_model={d_model} ---")
        
        model = CNNTransformer(n_chans=n_chans, n_times=n_times, n_outputs=1,
                              dropout_rate=dropout_rate, d_model=d_model)
        train_loader_ablation = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader_ablation = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        test_loader_ablation = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        best_val_rmse, train_rmses, val_rmses, best_epoch = train_ablation_model(
            exp_name, model, train_loader_ablation, valid_loader_ablation,
            ABLATION_EPOCHS, lr
        )
        
        model.load_state_dict(torch.load(os.path.join(ABLATION_DIR, f"{exp_name}_best.pt")))
        test_rmse = eval_epoch(model, test_loader_ablation)
        
        results.append({
            'experiment_name': exp_name,
            'd_model': d_model,
            'best_val_rmse': float(best_val_rmse),
            'test_rmse': float(test_rmse),
            'best_epoch': int(best_epoch),
            'final_train_rmse': float(train_rmses[-1]),
            'final_val_rmse': float(val_rmses[-1]),
            'train_rmses': [float(x) for x in train_rmses],
            'val_rmses': [float(x) for x in val_rmses],
        })
        print(f"  Val RMSE: {best_val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    
    return results


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def create_ablation_comparison_plots(all_results):
    """Create comprehensive comparison plots for all ablation studies"""
    print("\n" + "="*70)
    print("Creating Comparison Visualizations")
    print("="*70)
    
    # Count how many studies we have
    num_studies = len([k for k, v in all_results.items() if v])
    
    # Create grid based on number of studies
    if num_studies <= 4:
        rows, cols = 2, 2
    elif num_studies <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    # 1. Bar chart comparing all experiments
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if num_studies == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot 1: Learning Rate Comparison
    if 'learning_rate' in all_results and all_results['learning_rate']:
        lr_results = all_results['learning_rate']
        lr_values = [r['learning_rate'] for r in lr_results]
        val_rmses = [r['best_val_rmse'] for r in lr_results]
        test_rmses = [r['test_rmse'] for r in lr_results]
        
        x_pos = np.arange(len(lr_values))
        width = 0.35
        axes[plot_idx].bar(x_pos - width/2, val_rmses, width, label='Val RMSE', alpha=0.8)
        axes[plot_idx].bar(x_pos + width/2, test_rmses, width, label='Test RMSE', alpha=0.8)
        axes[plot_idx].set_xlabel('Learning Rate')
        axes[plot_idx].set_ylabel('RMSE')
        axes[plot_idx].set_title('Learning Rate Ablation')
        axes[plot_idx].set_xticks(x_pos)
        axes[plot_idx].set_xticklabels([f'{lr:.0e}' for lr in lr_values])
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 2: Batch Size Comparison
    if 'batch_size' in all_results and all_results['batch_size']:
        bs_results = all_results['batch_size']
        bs_values = [r['batch_size'] for r in bs_results]
        val_rmses = [r['best_val_rmse'] for r in bs_results]
        test_rmses = [r['test_rmse'] for r in bs_results]
        
        x_pos = np.arange(len(bs_values))
        width = 0.35
        axes[plot_idx].bar(x_pos - width/2, val_rmses, width, label='Val RMSE', alpha=0.8)
        axes[plot_idx].bar(x_pos + width/2, test_rmses, width, label='Test RMSE', alpha=0.8)
        axes[plot_idx].set_xlabel('Batch Size')
        axes[plot_idx].set_ylabel('RMSE')
        axes[plot_idx].set_title('Batch Size Ablation')
        axes[plot_idx].set_xticks(x_pos)
        axes[plot_idx].set_xticklabels(bs_values)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 3: Dropout Rate Comparison
    if 'dropout_rate' in all_results and all_results['dropout_rate']:
        dr_results = all_results['dropout_rate']
        dr_values = [r['dropout_rate'] for r in dr_results]
        val_rmses = [r['best_val_rmse'] for r in dr_results]
        test_rmses = [r['test_rmse'] for r in dr_results]
        
        x_pos = np.arange(len(dr_values))
        width = 0.35
        axes[plot_idx].bar(x_pos - width/2, val_rmses, width, label='Val RMSE', alpha=0.8)
        axes[plot_idx].bar(x_pos + width/2, test_rmses, width, label='Test RMSE', alpha=0.8)
        axes[plot_idx].set_xlabel('Dropout Rate')
        axes[plot_idx].set_ylabel('RMSE')
        axes[plot_idx].set_title('Dropout Rate Ablation')
        axes[plot_idx].set_xticks(x_pos)
        axes[plot_idx].set_xticklabels(dr_values)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 4: Transformer Layers
    if 'transformer_layers' in all_results and all_results['transformer_layers']:
        tl_results = all_results['transformer_layers']
        tl_values = [r['num_layers'] for r in tl_results]
        val_rmses = [r['best_val_rmse'] for r in tl_results]
        test_rmses = [r['test_rmse'] for r in tl_results]
        
        x_pos = np.arange(len(tl_values))
        width = 0.35
        axes[plot_idx].bar(x_pos - width/2, val_rmses, width, label='Val RMSE', alpha=0.8)
        axes[plot_idx].bar(x_pos + width/2, test_rmses, width, label='Test RMSE', alpha=0.8)
        axes[plot_idx].set_xlabel('Number of Layers')
        axes[plot_idx].set_ylabel('RMSE')
        axes[plot_idx].set_title('Transformer Layers Ablation')
        axes[plot_idx].set_xticks(x_pos)
        axes[plot_idx].set_xticklabels(tl_values)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 5: Transformer Heads
    if 'transformer_heads' in all_results and all_results['transformer_heads']:
        th_results = all_results['transformer_heads']
        th_values = [r['nhead'] for r in th_results]
        val_rmses = [r['best_val_rmse'] for r in th_results]
        test_rmses = [r['test_rmse'] for r in th_results]
        
        x_pos = np.arange(len(th_values))
        width = 0.35
        axes[plot_idx].bar(x_pos - width/2, val_rmses, width, label='Val RMSE', alpha=0.8)
        axes[plot_idx].bar(x_pos + width/2, test_rmses, width, label='Test RMSE', alpha=0.8)
        axes[plot_idx].set_xlabel('Number of Heads')
        axes[plot_idx].set_ylabel('RMSE')
        axes[plot_idx].set_title('Transformer Heads Ablation')
        axes[plot_idx].set_xticks(x_pos)
        axes[plot_idx].set_xticklabels(th_values)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 6: d_model
    if 'd_model' in all_results and all_results['d_model']:
        dm_results = all_results['d_model']
        dm_values = [r['d_model'] for r in dm_results]
        val_rmses = [r['best_val_rmse'] for r in dm_results]
        test_rmses = [r['test_rmse'] for r in dm_results]
        
        x_pos = np.arange(len(dm_values))
        width = 0.35
        axes[plot_idx].bar(x_pos - width/2, val_rmses, width, label='Val RMSE', alpha=0.8)
        axes[plot_idx].bar(x_pos + width/2, test_rmses, width, label='Test RMSE', alpha=0.8)
        axes[plot_idx].set_xlabel('d_model')
        axes[plot_idx].set_ylabel('RMSE')
        axes[plot_idx].set_title('Transformer Dimension Ablation')
        axes[plot_idx].set_xticks(x_pos)
        axes[plot_idx].set_xticklabels(dm_values)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    # Overall Best Comparison (always last)
    all_experiments = []
    all_val_rmses = []
    all_test_rmses = []
    
    for study_name, study_results in all_results.items():
        for result in study_results:
            all_experiments.append(result['experiment_name'])
            all_val_rmses.append(result['best_val_rmse'])
            all_test_rmses.append(result['test_rmse'])
    
    if all_experiments and plot_idx < len(axes):
        # Show top 10 best
        sorted_indices = np.argsort(all_val_rmses)[:10]
        top_experiments = [all_experiments[i] for i in sorted_indices]
        top_val_rmses = [all_val_rmses[i] for i in sorted_indices]
        top_test_rmses = [all_test_rmses[i] for i in sorted_indices]
        
        x_pos = np.arange(len(top_experiments))
        width = 0.35
        axes[plot_idx].barh(x_pos - width/2, top_val_rmses, width, label='Val RMSE', alpha=0.8)
        axes[plot_idx].barh(x_pos + width/2, top_test_rmses, width, label='Test RMSE', alpha=0.8)
        axes[plot_idx].set_yticks(x_pos)
        axes[plot_idx].set_yticklabels([name.replace('CNN_only_', '').replace('CNN_Transformer_', '') 
                                        for name in top_experiments], fontsize=8)
        axes[plot_idx].set_xlabel('RMSE')
        axes[plot_idx].set_title('Top 10 Best Experiments')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_DIR, "ablation_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Learning curves comparison for each study
    for study_name, study_results in all_results.items():
        if not study_results:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot training curves
        for result in study_results:
            epochs = list(range(1, len(result['train_rmses']) + 1))
            axes[0].plot(epochs, result['train_rmses'], label=result['experiment_name'], 
                        alpha=0.7, linewidth=1.5)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Train RMSE')
        axes[0].set_title(f'{study_name.replace("_", " ").title()} - Training Curves')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        # Plot validation curves
        for result in study_results:
            epochs = list(range(1, len(result['val_rmses']) + 1))
            axes[1].plot(epochs, result['val_rmses'], label=result['experiment_name'],
                        alpha=0.7, linewidth=1.5)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Val RMSE')
        axes[1].set_title(f'{study_name.replace("_", " ").title()} - Validation Curves')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ABLATION_DIR, f"{study_name}_curves.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✅ All visualizations saved!")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ABLATION STUDIES - OVERNIGHT RUN")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {ABLATION_DIR}")
    print(f"Epochs per experiment: {ABLATION_EPOCHS}")
    print("="*70)
    
    all_results = {}
    
    # Run ablation studies
    if ABLATION_CONFIG['learning_rate']['enabled']:
        all_results['learning_rate'] = run_ablation_learning_rate()
    
    if ABLATION_CONFIG['batch_size']['enabled']:
        all_results['batch_size'] = run_ablation_batch_size()
    
    if ABLATION_CONFIG['dropout_rate']['enabled']:
        all_results['dropout_rate'] = run_ablation_dropout_rate()
    
    if ABLATION_CONFIG.get('transformer_layers', {}).get('enabled', False):
        all_results['transformer_layers'] = run_ablation_transformer_layers()
    
    if ABLATION_CONFIG.get('transformer_heads', {}).get('enabled', False):
        all_results['transformer_heads'] = run_ablation_transformer_heads()
    
    if ABLATION_CONFIG.get('d_model', {}).get('enabled', False):
        all_results['d_model'] = run_ablation_d_model()
    
    # Create visualizations
    create_ablation_comparison_plots(all_results)
    
    # Save all results
    results_file = os.path.join(ABLATION_DIR, "ablation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': ABLATION_CONFIG,
            'ablation_epochs': ABLATION_EPOCHS,
            'device': DEVICE,
            'results': all_results
        }, f, indent=2)
    
    # Create comprehensive summary document
    summary_file = os.path.join(ABLATION_DIR, "ABLATION_SUMMARY.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ABLATION STUDIES SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Epochs per experiment: {ABLATION_EPOCHS}\n")
        f.write(f"Results directory: {ABLATION_DIR}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        # Overall best
        all_best_results = []
        for study_name, study_results in all_results.items():
            if study_results:
                best = min(study_results, key=lambda x: x['best_val_rmse'])
                all_best_results.append((study_name, best))
        
        if all_best_results:
            f.write("BEST CONFIGURATION FROM EACH STUDY:\n")
            f.write("-" * 70 + "\n")
            for study_name, best in all_best_results:
                f.write(f"{study_name.replace('_', ' ').title()}:\n")
                f.write(f"  Experiment: {best['experiment_name']}\n")
                f.write(f"  Val RMSE: {best['best_val_rmse']:.4f}\n")
                f.write(f"  Test RMSE: {best['test_rmse']:.4f}\n")
                f.write(f"  Best Epoch: {best['best_epoch']}\n")
                f.write("\n")
        
        # Overall best
        if all_best_results:
            overall_best = min(all_best_results, key=lambda x: x[1]['best_val_rmse'])
            f.write("=" * 70 + "\n")
            f.write("OVERALL BEST CONFIGURATION\n")
            f.write("=" * 70 + "\n")
            f.write(f"Study: {overall_best[0].replace('_', ' ').title()}\n")
            f.write(f"Experiment: {overall_best[1]['experiment_name']}\n")
            f.write(f"Val RMSE: {overall_best[1]['best_val_rmse']:.4f}\n")
            f.write(f"Test RMSE: {overall_best[1]['test_rmse']:.4f}\n")
            f.write(f"Best Epoch: {overall_best[1]['best_epoch']}\n\n")
        
        # Detailed results for each study
        for study_name, study_results in all_results.items():
            f.write("=" * 70 + "\n")
            f.write(f"{study_name.replace('_', ' ').upper()} DETAILED RESULTS\n")
            f.write("=" * 70 + "\n")
            f.write(f"{'Experiment':<45} {'Val RMSE':<12} {'Test RMSE':<12} {'Best Epoch':<12}\n")
            f.write("-" * 70 + "\n")
            for result in study_results:
                f.write(f"{result['experiment_name']:<45} {result['best_val_rmse']:<12.4f} "
                       f"{result['test_rmse']:<12.4f} {result['best_epoch']:<12}\n")
            
            if study_results:
                best = min(study_results, key=lambda x: x['best_val_rmse'])
                worst = max(study_results, key=lambda x: x['best_val_rmse'])
                f.write(f"\nBest: {best['experiment_name']} (Val RMSE: {best['best_val_rmse']:.4f})\n")
                f.write(f"Worst: {worst['experiment_name']} (Val RMSE: {worst['best_val_rmse']:.4f})\n")
                f.write(f"Improvement: {worst['best_val_rmse'] - best['best_val_rmse']:.4f} "
                       f"({(worst['best_val_rmse'] - best['best_val_rmse']) / worst['best_val_rmse'] * 100:.2f}%)\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("SAVED FILES\n")
        f.write("=" * 70 + "\n")
        f.write("For each experiment:\n")
        f.write("  - <experiment_name>/best_model.pt: Best model weights\n")
        f.write("  - <experiment_name>/checkpoint_epoch_*.pt: Periodic checkpoints\n")
        f.write("  - <experiment_name>/training_history.json: Complete training history\n")
        f.write("  - <experiment_name>/training_curves.png: Training/validation curves\n")
        f.write("\nSummary files:\n")
        f.write("  - ablation_results.json: Complete results (JSON)\n")
        f.write("  - ablation_summary.json: Summary with best configurations\n")
        f.write("  - ABLATION_SUMMARY.txt: This summary file\n")
        f.write("  - ablation_comparison.png: Comparison plots\n")
        f.write("  - <study_name>_curves.png: Learning curves for each study\n")
        f.write("  - README.txt: Directory structure guide\n")
    
    # Create JSON summary with best configurations
    summary_json = {
        'timestamp': datetime.now().isoformat(),
        'device': DEVICE,
        'ablation_epochs': ABLATION_EPOCHS,
        'config': ABLATION_CONFIG,
        'best_configurations': {},
        'overall_best': None,
        'results': all_results
    }
    
    # Find best from each study
    for study_name, study_results in all_results.items():
        if study_results:
            best = min(study_results, key=lambda x: x['best_val_rmse'])
            summary_json['best_configurations'][study_name] = best
    
    # Overall best
    if summary_json['best_configurations']:
        overall_best = min(summary_json['best_configurations'].items(), 
                          key=lambda x: x[1]['best_val_rmse'])
        summary_json['overall_best'] = {
            'study': overall_best[0],
            'experiment': overall_best[1]['experiment_name'],
            'val_rmse': overall_best[1]['best_val_rmse'],
            'test_rmse': overall_best[1]['test_rmse']
        }
    
    summary_json_file = os.path.join(ABLATION_DIR, "ablation_summary.json")
    with open(summary_json_file, 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    # Create README
    readme_file = os.path.join(ABLATION_DIR, "README.txt")
    with open(readme_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ABLATION STUDIES RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directory: {ABLATION_DIR}\n\n")
        f.write("This directory contains all results from the ablation studies.\n\n")
        f.write("STRUCTURE:\n")
        f.write("  - <experiment_name>/     : Individual experiment directories\n")
        f.write("    - best_model.pt        : Best model weights\n")
        f.write("    - checkpoint_epoch_*.pt: Periodic checkpoints\n")
        f.write("    - training_history.json: Complete training history\n")
        f.write("    - training_curves.png  : Training/validation curves\n")
        f.write("  - *_best.pt              : Quick access to best models\n")
        f.write("  - ablation_results.json  : Complete results (JSON)\n")
        f.write("  - ablation_summary.json  : Summary with best configurations\n")
        f.write("  - ABLATION_SUMMARY.txt   : Human-readable summary\n")
        f.write("  - ablation_comparison.png: Comparison plots\n")
        f.write("  - <study_name>_curves.png: Learning curves per study\n")
        f.write("\n")
        f.write("KEY FILES TO REVIEW:\n")
        f.write("  1. ABLATION_SUMMARY.txt - Start here for overview\n")
        f.write("  2. ablation_comparison.png - Visual comparison\n")
        f.write("  3. ablation_summary.json - Best configurations\n")
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDIES COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 All results saved to: {ABLATION_DIR}/")
    print(f"\n📄 Key files to review in the morning:")
    print(f"  ✅ {summary_file}")
    print(f"  ✅ {summary_json_file}")
    print(f"  ✅ {os.path.join(ABLATION_DIR, 'ablation_comparison.png')}")
    print(f"  ✅ {readme_file}")
    print(f"  ✅ {results_file}")
    
    # Print summary
    for study_name, study_results in all_results.items():
        print(f"\n{study_name.upper()} Results:")
        print(f"{'Experiment':<40} {'Val RMSE':<12} {'Test RMSE':<12}")
        print("-" * 70)
        for result in study_results:
            print(f"{result['experiment_name']:<40} {result['best_val_rmse']:<12.4f} {result['test_rmse']:<12.4f}")
        
        # Find best
        if study_results:
            best = min(study_results, key=lambda x: x['best_val_rmse'])
            print(f"\n✅ Best: {best['experiment_name']} (Val RMSE: {best['best_val_rmse']:.4f})")
    
    # Overall best
    if summary_json['overall_best']:
        print("\n" + "="*70)
        print("🏆 OVERALL BEST CONFIGURATION")
        print("="*70)
        print(f"Study: {summary_json['overall_best']['study']}")
        print(f"Experiment: {summary_json['overall_best']['experiment']}")
        print(f"Val RMSE: {summary_json['overall_best']['val_rmse']:.4f}")
        print(f"Test RMSE: {summary_json['overall_best']['test_rmse']:.4f}")
    
    print("\n" + "="*70)
    print(f"✅ All results saved! Check {ABLATION_DIR}/ in the morning!")
    print("="*70)
