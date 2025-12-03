#!/usr/bin/env python3
"""
Standardized Training Script for ALL 21 Models

This script ensures all models are trained on identical data splits:
- Training: Releases 1, 2, 3, 4, 6, 7, 8, 9, 10 (all releases 1-10 except 5)
- Validation: Release 5
  - For NNs/epoch-based: Early stopping
  - For linear/tree models: Hyperparameter tuning
- Testing: Release 11

All models receive:
- Same data splits
- Consistent regularization approaches
- Same hyperparameter tuning methodology

Models included (21 total):
- Linear: LinearRegression, Ridge, Lasso
- Tree-based: RandomForest, XGBoost
- Simple CNNs: CNN1D, SimpleEEGNet, EEGNet (Custom), CNN_only (EEGNetv4)
- Moderate CNNs: EEGNeX (Custom), EEGNeX (Braindecode), EEGMiner, Deep4Net
- Attention-based: ATCNet
- Transformers: SimpleEEGConformer, EEGConformer
- Hybrid: CNN_Transformer, HybridEEGRegressor
- Domain Adaptation: DANN, DANNModel, CNNTransformerDANN
- GNN: DualBranchEEGModel (GRU + GNN)
- Braindecode Models: Labram
"""

import os
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import copy
import json
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# Braindecode imports
from braindecode.models import (
    EEGNetv4, EEGConformer, EEGNeX, EEGMiner, Deep4Net, 
    ATCNet, Labram
)
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events

# PyG imports for GNN models
try:
    from torch_geometric.nn import GCNConv, GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: PyTorch Geometric not available. GNN models will be skipped.")

# EEGDash imports
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

# XGBoost (optional)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not available. XGBRegressor will be skipped.")

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", DEVICE)

# Training hyperparameters (will be tuned for NNs)
EPOCHS = 50  # Max epochs (early stopping will cut this short)
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5  # L2 regularization
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4

# Data configuration
EPOCH_LEN_S = 2.0
SFREQ = 100
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0
ANCHOR = "stimulus_anchor"

# STANDARDIZED DATA SPLIT
TRAIN_RELEASES = [1, 2, 3, 4, 6, 7, 8, 9, 10]  # All except 5
VAL_RELEASE = 5  # Validation for early stopping / hyperparameter tuning
TEST_RELEASE = 11  # Final testing

# Results directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"standardized_results_{TIMESTAMP}"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\n📁 Results will be saved to: {RESULTS_DIR}/")
print("=" * 70)
print("STANDARDIZED DATA SPLIT:")
print(f"  Training: Releases {TRAIN_RELEASES}")
print(f"  Validation: Release {VAL_RELEASE}")
print(f"  Testing: Release {TEST_RELEASE}")
print("=" * 70)

# ============================================================
# 1. DATA LOADING WITH STANDARDIZED SPLIT
# ============================================================
def load_release(release_id, data_dir="data_merged"):
    """Load a specific release - tries multiple possible data directory locations"""
    # Try multiple possible data directory locations, prioritizing data_merged
    # if data_dir == "data_merged":
    #     possible_dirs = [
    #         Path(f"data_merged/release_{release_id}"),
    #         Path(f"data/release_{release_id}"),
    #         Path(f"data/merged/release_{release_id}"),
    #     ]
    # else:
    #     possible_dirs = [
    #         Path(f"{data_dir}/release_{release_id}"),
    #         Path(f"data_merged/release_{release_id}"),
    #         Path(f"data/release_{release_id}"),
    #     ]
    
    release_dir = Path(f"{data_dir}/release_{release_id}")
    if not release_dir.exists():
        print(f"⚠️  No data directory found for Release {release_id}. Tried: {release_dir}")
        return None
    
    try:
        dataset = EEGChallengeDataset(
            task="contrastChangeDetection",
            release=f"R{release_id}",
            cache_dir=str(release_dir),
            mini=False
        )
        if len(dataset.datasets) > 0:
            print(f"   ✅ Loaded from: {release_dir}")
            return dataset
        else:
            print(f"⚠️  Release {release_id} loaded but has no datasets")
    except Exception as e:
        print(f"⚠️  Failed to load Release {release_id} from {release_dir}: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return None
    return None


def preprocess_and_window_dataset(dataset, release_id):
    """Preprocess and create windows for a dataset"""
    print(f"\n📊 Preprocessing Release {release_id}...")
    
    # Preprocessing
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
    preprocess(dataset, transformation_offline, n_jobs=1)
    
    # Keep only recordings with stimulus anchors
    dataset_filtered = keep_only_recordings_with(ANCHOR, dataset)
    
    # Create windows
    single_windows = create_windows_from_events(
        dataset_filtered,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )
    
    # Add metadata
    single_windows = add_extras_columns(
        single_windows,
        dataset_filtered,
        desc=ANCHOR,
        keys=("target", "rt_from_stimulus", "rt_from_trialstart",
              "stimulus_onset", "response_onset", "correct", "response_type")
    )
    
    return single_windows


print("\n" + "=" * 70)
print("LOADING DATA WITH STANDARDIZED SPLIT")
print("=" * 70)

# Load training releases
print(f"\n📥 Loading TRAINING releases: {TRAIN_RELEASES}")
train_datasets = []
for release_id in TRAIN_RELEASES:
    dataset = load_release(release_id)
    if dataset is not None:
        train_datasets.append(dataset)
        print(f"   ✅ Release {release_id}: {len(dataset.datasets)} recordings")

if not train_datasets:
    raise ValueError("No training datasets could be loaded!")

train_dataset_combined = BaseConcatDataset(train_datasets)
train_windows = preprocess_and_window_dataset(train_dataset_combined, f"Train_{TRAIN_RELEASES}")
print(f"✅ Training windows: {len(train_windows)}")

# Load validation release
print(f"\n📥 Loading VALIDATION release: {VAL_RELEASE}")
val_dataset = load_release(VAL_RELEASE)
if val_dataset is None:
    raise ValueError(f"Validation release {VAL_RELEASE} could not be loaded!")

val_windows = preprocess_and_window_dataset(val_dataset, VAL_RELEASE)
print(f"✅ Validation windows: {len(val_windows)}")

# Load test release
print(f"\n📥 Loading TEST release: {TEST_RELEASE}")
test_dataset = load_release(TEST_RELEASE)
if test_dataset is None:
    print(f"⚠️  Warning: Test release {TEST_RELEASE} not found. Will skip testing.")
    test_windows = None
else:
    test_windows = preprocess_and_window_dataset(test_dataset, TEST_RELEASE)
    print(f"✅ Test windows: {len(test_windows)}")

# Create dataloaders for neural networks
train_loader = DataLoader(train_windows, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_windows, batch_size=BATCH_SIZE, shuffle=False)
if test_windows is not None:
    test_loader = DataLoader(test_windows, batch_size=BATCH_SIZE, shuffle=False)
else:
    test_loader = None

# Detect input dimensions
sample_X, _, _ = next(iter(train_loader))
_, n_chans, n_times = sample_X.shape
print(f"\n📐 Detected input shape: n_chans={n_chans}, n_times={n_times}")

# ============================================================
# 2. FEATURE EXTRACTION FOR LINEAR/TREE MODELS
# ============================================================
def extract_features_from_window(window_np, fs=100.0):
    """
    Extract features from EEG window (same as submission_5/submission.py)
    Returns: feature vector of length 1161 (129 channels * 9 features per channel)
    """
    from scipy.signal import welch
    from scipy.stats import skew, kurtosis
    
    def bandpower(data, fs, fmin, fmax):
        """Compute bandpower using Welch's method"""
        f, Pxx = welch(data, fs=fs, nperseg=min(256, len(data)), nfft=1024)
        band = (f >= fmin) & (f <= fmax)
        return np.trapz(Pxx[band], f[band])
    
    # Basic stats
    means = window_np.mean(axis=1)
    stds = window_np.std(axis=1) + 1e-8
    skews = skew(window_np, axis=1, bias=False, nan_policy='omit')
    kurts = kurtosis(window_np, axis=1, fisher=True, bias=False, nan_policy='omit')
    
    # Frequency bands
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }
    
    # Bandpower for each channel
    band_feats = []
    for ch in range(window_np.shape[0]):
        ch_data = window_np[ch, :]
        ch_bandpowers = [bandpower(ch_data, fs, fmin, fmax) for fmin, fmax in bands.values()]
        band_feats.append(ch_bandpowers)
    band_feats = np.array(band_feats)
    
    # Combine: mean, std, skew, kurt, 5 bandpowers = 9 features per channel
    feats = np.concatenate([
        means[:, None],
        stds[:, None],
        skews[:, None],
        kurts[:, None],
        band_feats
    ], axis=1).reshape(-1)
    
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def extract_features_from_dataset(windows):
    """Extract features from all windows in dataset"""
    print("🔧 Extracting features from dataset...")
    features = []
    targets = []
    
    for i in tqdm(range(len(windows)), desc="Extracting features"):
        window_data = windows[i][0]  # (n_chans, n_times)
        target = windows[i][1]  # scalar
        
        # Convert to numpy if needed
        if isinstance(window_data, torch.Tensor):
            window_np = window_data.numpy()
        else:
            window_np = np.array(window_data)
        
        feat = extract_features_from_window(window_np, fs=SFREQ)
        features.append(feat)
        targets.append(float(target))
    
    return np.array(features), np.array(targets)


# Extract features for linear/tree models
print("\n" + "=" * 70)
print("EXTRACTING FEATURES FOR LINEAR/TREE MODELS")
print("=" * 70)

X_train, y_train = extract_features_from_dataset(train_windows)
X_val, y_val = extract_features_from_dataset(val_windows)
if test_windows is not None:
    X_test, y_test = extract_features_from_dataset(test_windows)
else:
    X_test, y_test = None, None

print(f"✅ Training features: {X_train.shape}")
print(f"✅ Validation features: {X_val.shape}")
if X_test is not None:
    print(f"✅ Test features: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
if X_test is not None:
    X_test_scaled = scaler.transform(X_test)

# ============================================================
# 3. MODEL DEFINITIONS
# ============================================================

# ============================================================
# 3. MODEL DEFINITIONS - ALL MODELS
# ============================================================

# Simple CNNs
class CNN1D(nn.Module):
    """Simple 1D CNN baseline"""
    def __init__(self, n_chans=129, n_outputs=1, n_times=200):
        super().__init__()
        self.conv1 = nn.Conv1d(n_chans, 64, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=10, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        self.fc_input_size = 256 * (n_times // 8)
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_outputs)
        )
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(-1)


class SimpleEEGNet(nn.Module):
    """Simplified EEGNet"""
    def __init__(self, n_chans=129, n_outputs=1, n_times=200, sfreq=100):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, 8, (1, 64), padding=(0, 32))
        self.temporal_bn = nn.BatchNorm2d(8)
        
        self.spatial_conv = nn.Conv2d(8, 16, (n_chans, 1), bias=False)
        self.spatial_bn = nn.BatchNorm2d(16)
        self.spatial_activation = nn.ELU()
        self.spatial_pool = nn.AvgPool2d((1, 4))
        
        self.feature_conv = nn.Conv2d(16, 32, (1, 16), padding=(0, 8))
        self.feature_bn = nn.BatchNorm2d(32)
        self.feature_activation = nn.ELU()
        self.feature_pool = nn.AvgPool2d((1, 8))
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * (n_times // 32), 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_outputs)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.spatial_activation(x)
        x = self.spatial_pool(x)
        x = self.feature_conv(x)
        x = self.feature_bn(x)
        x = self.feature_activation(x)
        x = self.feature_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(-1)


class EEGNet(nn.Module):
    """Custom simplified EEGNet"""
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
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        f = self.features(x)
        f = f.flatten(1)
        return self.regressor(f).squeeze(-1)


class CNNOnly(nn.Module):
    """CNN-only baseline using EEGNetv4"""
    def __init__(self, n_chans, n_times, n_outputs=1):
        super().__init__()
        self.cnn = EEGNetv4(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
    
    def forward(self, x):
        return self.cnn(x).squeeze(-1)


# Moderate CNNs
class EEGNeX_Custom(nn.Module):
    """Custom EEGNeX implementation"""
    def __init__(self, n_chans=129, n_outputs=1, n_times=200, sfreq=100):
        super().__init__()
        self.temporal_conv = nn.Conv1d(n_chans, 32, kernel_size=15, padding=7)
        self.temporal_bn = nn.BatchNorm1d(32)
        self.temporal_activation = nn.ELU()
        self.temporal_pool = nn.AvgPool1d(kernel_size=2)
        
        self.spatial_conv = nn.Conv1d(32, 64, kernel_size=1)
        self.spatial_bn = nn.BatchNorm1d(64)
        self.spatial_activation = nn.ELU()
        
        self.feature_conv1 = nn.Conv1d(64, 128, kernel_size=10, padding=4)
        self.feature_bn1 = nn.BatchNorm1d(128)
        self.feature_activation1 = nn.ELU()
        self.feature_pool1 = nn.AvgPool1d(kernel_size=2)
        
        self.feature_conv2 = nn.Conv1d(128, 256, kernel_size=10, padding=4)
        self.feature_bn2 = nn.BatchNorm1d(256)
        self.feature_activation2 = nn.ELU()
        self.feature_pool2 = nn.AvgPool1d(kernel_size=2)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_outputs)
        )
    
    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = self.temporal_activation(x)
        x = self.temporal_pool(x)
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.spatial_activation(x)
        x = self.feature_conv1(x)
        x = self.feature_bn1(x)
        x = self.feature_activation1(x)
        x = self.feature_pool1(x)
        x = self.feature_conv2(x)
        x = self.feature_bn2(x)
        x = self.feature_activation2(x)
        x = self.feature_pool2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(-1)


# Transformer Models
class SimpleEEGConformer(nn.Module):
    """Simple Transformer encoder"""
    def __init__(self, n_chans, n_times, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(n_chans, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        h = self.transformer(x)
        pooled = h.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


class CNNTransformer(nn.Module):
    """CNN + Transformer"""
    def __init__(self, n_chans, n_times, n_outputs=1, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = EEGNetv4(n_chans=n_chans, n_times=n_times, n_outputs=d_model)
        if hasattr(self.cnn, 'classifier'):
            self.cnn.classifier = nn.Identity()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, 
            dim_feedforward=d_model*4, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_outputs)
        )
    
    def forward(self, x):
        cnn_feat = self.cnn(x).unsqueeze(1)
        trans_out = self.transformer(cnn_feat)
        pooled = trans_out.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


# Hybrid Models
class HybridEEGRegressor(nn.Module):
    """CNN-Transformer hybrid"""
    def __init__(self, n_chans=129, n_times=200, n_outputs=1, sfreq=100,
                 n_filters_temporal=40, filter_time_length=25, n_filters_spatial=40,
                 embed_dim=128, num_heads=8, num_layers=4, dropout=0.3, fusion_dim=256):
        super().__init__()
        # CNN branch
        self.conv_time = nn.Conv2d(1, n_filters_temporal, (1, filter_time_length),
                                   padding=(0, filter_time_length // 2), bias=False)
        self.conv_spatial = nn.Conv2d(n_filters_temporal, n_filters_spatial, (n_chans, 1), bias=False)
        self.batch_norm = nn.BatchNorm2d(n_filters_spatial)
        self.activation = nn.ELU()
        self.pool_time = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout_cnn = nn.Dropout(dropout)
        self.cnn_out_dim = n_filters_spatial * (n_times // 15)
        
        # Transformer branch
        self.input_projection = nn.Linear(n_chans, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_times, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_out_dim + embed_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim // 2, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(64, n_outputs)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        # CNN branch
        x_cnn = x.unsqueeze(1)
        x_cnn = self.conv_time(x_cnn)
        x_cnn = self.conv_spatial(x_cnn)
        x_cnn = self.batch_norm(x_cnn)
        x_cnn = self.activation(x_cnn)
        x_cnn = self.pool_time(x_cnn)
        x_cnn = self.dropout_cnn(x_cnn)
        x_cnn = x_cnn.view(batch_size, -1)
        
        # Transformer branch
        x_trans = x.permute(0, 2, 1)
        x_trans = self.input_projection(x_trans) + self.pos_encoding
        x_trans = self.transformer(x_trans)
        x_trans = x_trans.mean(dim=1)
        
        # Fusion
        fused = torch.cat([x_cnn, x_trans], dim=1)
        fused = self.fusion(fused)
        return self.regression_head(fused).squeeze(-1)


# Domain Adaptation Models
class GRL(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return GradReverse.apply(x, self.lambd)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.clone()
    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


class DANN(nn.Module):
    """DANN with EEGNetv4 backend"""
    def __init__(self, n_chans, n_times, n_domains, feature_dim=64, lambd=0.5):
        super().__init__()
        self.feature = EEGNetv4(n_chans=n_chans, n_times=n_times, n_outputs=feature_dim)
        if hasattr(self.feature, 'classifier'):
            self.feature.classifier = nn.Identity()
        
        self.reg_head = nn.Sequential(
            nn.Linear(feature_dim, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(10, 1)
        )
        
        self.grl = GRL(lambd)
        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(10, n_domains)
        )
    
    def forward(self, x, return_domain=False):
        feat = self.feature(x)
        y_hat = self.reg_head(feat).squeeze(-1)
        if not return_domain:
            return y_hat
        dom = self.domain_head(self.grl(feat))
        return y_hat, dom


class DANNModel(nn.Module):
    """DANN with CNN+Transformer backend"""
    def __init__(self, n_chans, n_times, n_domains, d_model=64, lambd=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, (1, 7), padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (n_chans, 5), padding=(0, 2)),
            nn.ReLU()
        )
        self.proj = nn.Linear(32, d_model)
        encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model, 1)
        )
        
        self.grl = GRL(lambd=lambd)
        self.domain_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model, n_domains)
        )
    
    def forward(self, x, return_domain=False):
        x = x.unsqueeze(1)
        f = self.cnn(x).squeeze(2)
        f = f.permute(0, 2, 1)
        h = self.proj(f)
        h = self.transformer(h)
        emb = h.mean(dim=1)
        y = self.reg_head(emb).squeeze(-1)
        if not return_domain:
            return y
        d_emb = self.grl(emb)
        dom = self.domain_head(d_emb)
        return y, dom


class CNNTransformerDANN(nn.Module):
    """CNN + Transformer + DANN"""
    def __init__(self, n_chans, n_times, n_domains, feature_dim=64, lambd=0.5):
        super().__init__()
        self.cnn = EEGNetv4(n_chans=n_chans, n_times=n_times, n_outputs=feature_dim)
        if hasattr(self.cnn, 'classifier'):
            self.cnn.classifier = nn.Identity()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=4, batch_first=True, dim_feedforward=feature_dim*4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.reg_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 1)
        )
        
        self.grl = GRL(lambd)
        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, n_domains)
        )
    
    def forward(self, x, return_domain=False):
        cnn_feat = self.cnn(x).unsqueeze(1)
        trans_out = self.transformer(cnn_feat)
        pooled = trans_out.mean(dim=1)
        y_hat = self.reg_head(pooled).squeeze(-1)
        if not return_domain:
            return y_hat
        dom = self.domain_head(self.grl(pooled))
        return y_hat, dom


# Graph construction utility (used by GNN models)
def build_functional_connectivity_graph(eeg_data, threshold=0.3):
    """Build graph from functional connectivity"""
    eeg_channels = eeg_data[:, :128, :]
    signals = eeg_channels.transpose(0, 2, 1).reshape(-1, 128)
    corr_matrix = np.corrcoef(signals.T)
    adj_matrix = np.abs(corr_matrix) > threshold
    np.fill_diagonal(adj_matrix, False)
    adj_matrix = np.triu(adj_matrix, k=1)
    edge_index = np.array(np.where(adj_matrix))
    edge_weights = corr_matrix[adj_matrix]
    return edge_index, edge_weights

# GNN Models (if PyG available)
if HAS_PYG:
    
    class DualBranchEEGModel(nn.Module):
        """GRU + GNN hybrid"""
        def __init__(self, n_channels=128, n_times=200, gru_hidden_dim=64, gru_num_layers=2,
                     gnn_hidden_dim=64, gnn_num_layers=2, use_gat=False, fusion_dim=128, dropout=0.3):
            super().__init__()
            self.n_channels = n_channels
            self.n_times = n_times
            self.use_gat = use_gat
            
            # GRU branch
            self.gru = nn.GRU(input_size=1, hidden_size=gru_hidden_dim, num_layers=gru_num_layers,
                             batch_first=True, dropout=dropout if gru_num_layers > 1 else 0.0, bidirectional=True)
            self.gru_proj = nn.Linear(gru_hidden_dim * 2, gru_hidden_dim)
            
            # GNN branch
            if use_gat:
                heads = 4
                self.gnn_layers = nn.ModuleList()
                self.gnn_layers.append(GATConv(n_times, gnn_hidden_dim, heads=heads, dropout=dropout))
                for _ in range(gnn_num_layers - 1):
                    self.gnn_layers.append(GATConv(gnn_hidden_dim * heads, gnn_hidden_dim, heads=heads, dropout=dropout))
                self.gat_final_proj = nn.Linear(gnn_hidden_dim * heads, gnn_hidden_dim)
                self.gnn_bn = nn.ModuleList([nn.BatchNorm1d(gnn_hidden_dim * heads) for _ in range(gnn_num_layers)])
            else:
                self.gnn_layers = nn.ModuleList()
                self.gnn_layers.append(GCNConv(n_times, gnn_hidden_dim))
                for _ in range(gnn_num_layers - 1):
                    self.gnn_layers.append(GCNConv(gnn_hidden_dim, gnn_hidden_dim))
                self.gnn_bn = nn.ModuleList([nn.BatchNorm1d(gnn_hidden_dim) for _ in range(gnn_num_layers)])
            
            # Fusion
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
            B = x.size(0)
            eeg = x[:, :128, :]
            
            # GRU branch
            eeg_time = eeg.reshape(B * self.n_channels, self.n_times, 1)
            gru_out, _ = self.gru(eeg_time)
            gru_last = gru_out[:, -1, :]
            gru_feat = self.gru_proj(gru_last)
            gru_feat = self.dropout(gru_feat)
            gru_feat = gru_feat.reshape(B, self.n_channels, -1)
            
            # GNN branch
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
            
            gnn_feat = gnn_feat.reshape(B, self.n_channels, -1)
            
            # Fusion
            fused = torch.cat([gru_feat, gnn_feat], dim=-1)
            fused = self.fusion(fused)
            pooled = fused.mean(dim=1)
            out = self.predictor(pooled)
            return out.squeeze(-1)


# ============================================================
# 4. TRAINING UTILITIES
# ============================================================
def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2))


def nrmse(a, b):
    """Normalized RMSE: RMSE / mean(|targets|)"""
    rmse_val = torch.sqrt(torch.mean((a - b) ** 2))
    mean_target = torch.mean(torch.abs(b))
    return rmse_val / (mean_target + 1e-8)


def train_neural_network(model, train_loader, val_loader, model_name, epochs=EPOCHS, 
                         lr=LR, weight_decay=WEIGHT_DECAY, patience=EARLY_STOPPING_PATIENCE, return_domain=False):
    """Train neural network with early stopping"""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    
    best_rmse = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    train_history = {'rmse': [], 'nrmse': [], 'loss': [], 'epoch': []}
    val_history = {'rmse': [], 'nrmse': [], 'errors': [], 'predictions': [], 'targets': []}
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_rmse = 0
        train_nrmse = 0
        train_loss = 0
        steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            X, y, _ = batch[0], batch[1], batch[2] if len(batch) > 2 else None
            X = X.to(DEVICE).float()
            y = y.to(DEVICE).float().view(-1)
            
            optimizer.zero_grad()
            if return_domain:
                pred = model(X, return_domain=False).view(-1)
            else:
                pred = model(X).view(-1)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            
            train_rmse += rmse(pred, y).item()
            train_nrmse += nrmse(pred, y).item()
            train_loss += loss.item()
            steps += 1
        
        train_rmse /= steps
        train_nrmse /= steps
        train_loss /= steps
        
        # Validation
        model.eval()
        val_rmse = 0
        val_nrmse = 0
        val_steps = 0
        val_preds = []
        val_targets = []
        val_errors = []
        
        with torch.no_grad():
            for batch in val_loader:
                X, y, _ = batch[0], batch[1], batch[2] if len(batch) > 2 else None
                X = X.to(DEVICE).float()
                y = y.to(DEVICE).float().view(-1)
                if return_domain:
                    pred = model(X, return_domain=False).view(-1)
                else:
                    pred = model(X).view(-1)
                val_rmse += rmse(pred, y).item()
                val_nrmse += nrmse(pred, y).item()
                val_steps += 1
                val_preds.append(pred.cpu().numpy())
                val_targets.append(y.cpu().numpy())
                val_errors.append((pred - y).cpu().numpy())
        
        val_rmse /= val_steps
        val_nrmse /= val_steps
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_errors = np.concatenate(val_errors)
        
        train_history['rmse'].append(train_rmse)
        train_history['nrmse'].append(train_nrmse)
        train_history['loss'].append(train_loss)
        train_history['epoch'].append(epoch)
        val_history['rmse'].append(val_rmse)
        val_history['nrmse'].append(val_nrmse)
        val_history['errors'].append(val_errors)
        val_history['predictions'].append(val_preds)
        val_history['targets'].append(val_targets)
        
        print(f"Epoch {epoch}: Train RMSE={train_rmse:.4f} | Val RMSE={val_rmse:.4f} | Val NRMSE={val_nrmse:.4f}")
        
        # Early stopping
        if val_rmse < best_rmse - EARLY_STOPPING_MIN_DELTA:
            best_rmse = val_rmse
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{model_name}_best.pt"))
            print(f"  ✅ New best! (RMSE={best_rmse:.4f}, NRMSE={val_nrmse:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  ⏹️  Early stopping at epoch {epoch}")
                break
    
    # Load best model and save final checkpoint
    best_model_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        # Save final model weights (ensure it's saved)
        torch.save(model.state_dict(), best_model_path)
        print(f"  💾 Model weights saved to: {best_model_path}")
    else:
        # If no best model was saved (shouldn't happen), save current model
        torch.save(model.state_dict(), best_model_path)
        print(f"  💾 Model weights saved to: {best_model_path}")
    
    # Also save full model checkpoint for easier loading
    full_checkpoint = {
        'model_state_dict': model.state_dict(),
        'best_rmse': best_rmse,
        'best_epoch': best_epoch,
        'train_history': train_history,
        'val_history': val_history,
        'model_name': model_name,
        'hyperparameters': {
            'lr': lr,
            'weight_decay': weight_decay,
            'batch_size': BATCH_SIZE,
            'epochs': epochs,
            'patience': patience
        }
    }
    checkpoint_path = os.path.join(RESULTS_DIR, f"{model_name}_checkpoint.pt")
    torch.save(full_checkpoint, checkpoint_path)
    print(f"  💾 Full checkpoint saved to: {checkpoint_path}")
    
    # Plot training curves
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs_list = train_history['epoch']
        axes[0].plot(epochs_list, train_history['rmse'], label='Train RMSE', marker='o')
        axes[0].plot(epochs_list, val_history['rmse'], label='Val RMSE', marker='s')
        axes[0].axvline(best_epoch, color='r', linestyle='--', label=f'Best (epoch {best_epoch})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title(f'{model_name} - RMSE over Epochs')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs_list, train_history['nrmse'], label='Train NRMSE', marker='o')
        axes[1].plot(epochs_list, val_history['nrmse'], label='Val NRMSE', marker='s')
        axes[1].axvline(best_epoch, color='r', linestyle='--', label=f'Best (epoch {best_epoch})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('NRMSE')
        axes[1].set_title(f'{model_name} - NRMSE over Epochs')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_training_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  ⚠️  Warning: Could not save training curves: {e}")
    
    # Error distribution analysis on best validation errors
    if len(val_history['errors']) > 0:
        best_val_errors = val_history['errors'][best_epoch - 1]
        analyze_error_distribution(best_val_errors, f"{model_name}_val", RESULTS_DIR)
    
    return best_rmse, best_epoch, train_history, val_history


def analyze_error_distribution(errors, name, save_dir):
    """Analyze and visualize error distributions"""
    os.makedirs(save_dir, exist_ok=True)
    errors = np.array(errors).flatten()
    
    stats_dict = {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "median": float(np.median(errors)),
        "q25": float(np.percentile(errors, 25)),
        "q75": float(np.percentile(errors, 75)),
        "skewness": float(skew(errors)),
        "kurtosis": float(kurtosis(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
    }
    
    with open(os.path.join(save_dir, f"{name}_error_stats.json"), "w") as f:
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
    plt.savefig(os.path.join(save_dir, f"{name}_error_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    return stats_dict


def train_linear_model(model_class, model_name, param_grid=None):
    """Train linear/tree model with hyperparameter tuning on validation set"""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    if param_grid is None:
        # Default: just train without tuning
        model = model_class()
        model.fit(X_train_scaled, y_train)
        val_pred = model.predict(X_val_scaled)
        val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        best_params = None
    else:
        # Grid search on validation set
        print(f"🔍 Hyperparameter tuning with {len(param_grid)} combinations...")
        model = GridSearchCV(
            model_class(), 
            param_grid, 
            cv=3,  # 3-fold CV on training set
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train_scaled, y_train)
        best_params = model.best_params_
        val_pred = model.predict(X_val_scaled)
        val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        print(f"  ✅ Best params: {best_params}")
        print(f"  ✅ Best CV score: {-model.best_score_:.4f}")
    
    print(f"  ✅ Validation RMSE: {val_rmse:.4f}")
    
    # Save model (pickle for sklearn models)
    import pickle
    model_path = os.path.join(RESULTS_DIR, f"{model_name}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  💾 Model saved to: {model_path}")
    
    # Also save scaler (needed for inference)
    scaler_path = os.path.join(RESULTS_DIR, f"{model_name}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  💾 Scaler saved to: {scaler_path}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'best_params': best_params,
        'val_rmse': val_rmse,
        'n_features': X_train_scaled.shape[1],
        'n_train_samples': X_train_scaled.shape[0],
        'n_val_samples': X_val_scaled.shape[0]
    }
    metadata_path = os.path.join(RESULTS_DIR, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  💾 Metadata saved to: {metadata_path}")
    
    return val_rmse, best_params, model


# ============================================================
# 5. TRAIN ALL MODELS WITH STANDARDIZED SETUP
# ============================================================
print("\n" + "=" * 70)
print("TRAINING ALL MODELS WITH STANDARDIZED SETUP")
print("=" * 70)

results = {}

# ============================================================
# LINEAR MODELS
# ============================================================
print("\n" + "=" * 70)
print("LINEAR MODELS")
print("=" * 70)

# LinearRegression
results['LinearRegression'] = train_linear_model(
    LinearRegression, 
    'LinearRegression'
)

# Ridge with hyperparameter tuning
ridge_params = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
results['Ridge'] = train_linear_model(
    Ridge,
    'Ridge',
    param_grid=ridge_params
)

# Lasso with hyperparameter tuning
lasso_params = {'alpha': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]}
results['Lasso'] = train_linear_model(
    Lasso,
    'Lasso',
    param_grid=lasso_params
)

# ============================================================
# TREE-BASED MODELS
# ============================================================
print("\n" + "=" * 70)
print("TREE-BASED MODELS")
print("=" * 70)

# RandomForest with hyperparameter tuning
rf_params = {
    'n_estimators': [200, 400, 600],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
results['RandomForest'] = train_linear_model(
    RandomForestRegressor,
    'RandomForest',
    param_grid=rf_params
)

# XGBoost with hyperparameter tuning
if HAS_XGB:
    xgb_params = {
        'n_estimators': [300, 500, 700],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'reg_lambda': [0.5, 1.0, 2.0]
    }
    results['XGBoost'] = train_linear_model(
        XGBRegressor,
        'XGBoost',
        param_grid=xgb_params
    )

# ============================================================
# NEURAL NETWORK MODELS
# ============================================================
print("\n" + "=" * 70)
print("NEURAL NETWORK MODELS")
print("=" * 70)

# Simple CNNs
print("\n--- Simple CNNs ---")
cnn1d_model = CNN1D(n_chans=n_chans, n_times=n_times)
results['CNN1D'] = train_neural_network(
    cnn1d_model, train_loader, val_loader, 'CNN1D'
)

simple_eegnet_model = SimpleEEGNet(n_chans=n_chans, n_times=n_times)
results['SimpleEEGNet'] = train_neural_network(
    simple_eegnet_model, train_loader, val_loader, 'SimpleEEGNet'
)

eegnet_custom_model = EEGNet(n_chans=n_chans, n_times=n_times)
results['EEGNet_Custom'] = train_neural_network(
    eegnet_custom_model, train_loader, val_loader, 'EEGNet_Custom'
)

cnn_only_model = CNNOnly(n_chans=n_chans, n_times=n_times)
results['CNN_only'] = train_neural_network(
    cnn_only_model, train_loader, val_loader, 'CNN_only'
)

# Moderate CNNs
print("\n--- Moderate CNNs ---")
eegnex_custom_model = EEGNeX_Custom(n_chans=n_chans, n_times=n_times)
results['EEGNeX_Custom'] = train_neural_network(
    eegnex_custom_model, train_loader, val_loader, 'EEGNeX_Custom'
)

# Braindecode models
try:
    eegnex_braindecode = EEGNeX(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ)
    results['EEGNeX_Braindecode'] = train_neural_network(
        eegnex_braindecode, train_loader, val_loader, 'EEGNeX_Braindecode'
    )
except Exception as e:
    print(f"⚠️  Could not train EEGNeX (Braindecode): {e}")

try:
    eegminer_model = EEGMiner(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ)
    results['EEGMiner'] = train_neural_network(
        eegminer_model, train_loader, val_loader, 'EEGMiner'
    )
except Exception as e:
    print(f"⚠️  Could not train EEGMiner: {e}")

try:
    deep4net_model = Deep4Net(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ)
    results['Deep4Net'] = train_neural_network(
        deep4net_model, train_loader, val_loader, 'Deep4Net'
    )
except Exception as e:
    print(f"⚠️  Could not train Deep4Net: {e}")

# Attention-based CNN
print("\n--- Attention-based CNN ---")
try:
    atcnet_model = ATCNet(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ)
    results['ATCNet'] = train_neural_network(
        atcnet_model, train_loader, val_loader, 'ATCNet'
    )
except Exception as e:
    print(f"⚠️  Could not train ATCNet: {e}")

# Transformer Models
print("\n--- Transformer Models ---")
simple_conformer_model = SimpleEEGConformer(n_chans=n_chans, n_times=n_times)
results['SimpleEEGConformer'] = train_neural_network(
    simple_conformer_model, train_loader, val_loader, 'SimpleEEGConformer'
)

conformer_model = EEGConformer(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ)
results['EEGConformer'] = train_neural_network(
    conformer_model, train_loader, val_loader, 'EEGConformer'
)

# Hybrid Models
print("\n--- Hybrid Models ---")
cnn_trans_model = CNNTransformer(n_chans=n_chans, n_times=n_times, dropout=0.3)
results['CNN_Transformer'] = train_neural_network(
    cnn_trans_model, train_loader, val_loader, 'CNN_Transformer'
)

hybrid_model = HybridEEGRegressor(n_chans=n_chans, n_times=n_times, dropout=0.3)
results['HybridEEGRegressor'] = train_neural_network(
    hybrid_model, train_loader, val_loader, 'HybridEEGRegressor'
)

# GNN Models (requires graph construction)
if HAS_PYG:
    print("\n--- GNN Models ---")
    # Build functional connectivity graph from training data
    print("🔧 Building functional connectivity graph...")
    sample_data = []
    for i in range(min(2000, len(train_windows))):
        X = train_windows[i][0]
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        sample_data.append(X)
    sample_data = np.stack(sample_data, axis=0)  # (N, 129, 200)
    
    edge_index_np, edge_weights_np = build_functional_connectivity_graph(sample_data, threshold=0.3)
    edge_index_t = torch.from_numpy(edge_index_np).long().to(DEVICE)
    edge_weights_t = torch.from_numpy(edge_weights_np).float().to(DEVICE) if edge_weights_np is not None else None
    
    print(f"   Graph: {edge_index_t.shape[1]} edges")
    
    # Create custom training function for GNN models
    def train_gnn_model(model, train_loader, val_loader, model_name, edge_index, edge_weights):
        """Train GNN model with graph structure"""
        print(f"\n{'='*70}")
        print(f"Training {model_name}")
        print(f"{'='*70}")
        
        model = model.to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.MSELoss()
        
        best_rmse = float("inf")
        best_epoch = 0
        epochs_no_improve = 0
        train_history = {'rmse': [], 'nrmse': [], 'loss': [], 'epoch': []}
        val_history = {'rmse': [], 'nrmse': [], 'errors': [], 'predictions': [], 'targets': []}
        
        for epoch in range(1, EPOCHS + 1):
            # Training
            model.train()
            train_rmse = 0
            train_nrmse = 0
            train_loss = 0
            steps = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
                X, y, _ = batch[0], batch[1], batch[2] if len(batch) > 2 else None
                X = X.to(DEVICE).float()
                y = y.to(DEVICE).float().view(-1)
                
                optimizer.zero_grad()
                pred = model(X, edge_index_t, edge_weights_t).view(-1)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                
                train_rmse += rmse(pred, y).item()
                train_nrmse += nrmse(pred, y).item()
                train_loss += loss.item()
                steps += 1
            
            train_rmse /= steps
            train_nrmse /= steps
            train_loss /= steps
            
            # Validation
            model.eval()
            val_rmse = 0
            val_nrmse = 0
            val_steps = 0
            val_preds = []
            val_targets = []
            val_errors = []
            
            with torch.no_grad():
                for batch in val_loader:
                    X, y, _ = batch[0], batch[1], batch[2] if len(batch) > 2 else None
                    X = X.to(DEVICE).float()
                    y = y.to(DEVICE).float().view(-1)
                    pred = model(X, edge_index_t, edge_weights_t).view(-1)
                    val_rmse += rmse(pred, y).item()
                    val_nrmse += nrmse(pred, y).item()
                    val_steps += 1
                    val_preds.append(pred.cpu().numpy())
                    val_targets.append(y.cpu().numpy())
                    val_errors.append((pred - y).cpu().numpy())
            
            val_rmse /= val_steps
            val_nrmse /= val_steps
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            val_errors = np.concatenate(val_errors)
            
            train_history['rmse'].append(train_rmse)
            train_history['nrmse'].append(train_nrmse)
            train_history['loss'].append(train_loss)
            train_history['epoch'].append(epoch)
            val_history['rmse'].append(val_rmse)
            val_history['nrmse'].append(val_nrmse)
            val_history['errors'].append(val_errors)
            val_history['predictions'].append(val_preds)
            val_history['targets'].append(val_targets)
            
            print(f"Epoch {epoch}: Train RMSE={train_rmse:.4f} | Val RMSE={val_rmse:.4f} | Val NRMSE={val_nrmse:.4f}")
            
            # Early stopping
            if val_rmse < best_rmse - EARLY_STOPPING_MIN_DELTA:
                best_rmse = val_rmse
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{model_name}_best.pt"))
                print(f"  ✅ New best! (RMSE={best_rmse:.4f}, NRMSE={val_nrmse:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"  ⏹️  Early stopping at epoch {epoch}")
                    break
        
        # Load best model and save final checkpoint
        best_model_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pt")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            # Save final model weights (ensure it's saved)
            torch.save(model.state_dict(), best_model_path)
            print(f"  💾 Model weights saved to: {best_model_path}")
        else:
            # If no best model was saved (shouldn't happen), save current model
            torch.save(model.state_dict(), best_model_path)
            print(f"  💾 Model weights saved to: {best_model_path}")
        
        # Also save full model checkpoint for easier loading
        full_checkpoint = {
            'model_state_dict': model.state_dict(),
            'best_rmse': best_rmse,
            'best_epoch': best_epoch,
            'train_history': train_history,
            'val_history': val_history,
            'model_name': model_name,
            'hyperparameters': {
                'lr': LR,
                'weight_decay': WEIGHT_DECAY,
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'patience': EARLY_STOPPING_PATIENCE
            }
        }
        checkpoint_path = os.path.join(RESULTS_DIR, f"{model_name}_checkpoint.pt")
        torch.save(full_checkpoint, checkpoint_path)
        print(f"  💾 Full checkpoint saved to: {checkpoint_path}")
        
        # Plot training curves
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            epochs_list = train_history['epoch']
            axes[0].plot(epochs_list, train_history['rmse'], label='Train RMSE', marker='o')
            axes[0].plot(epochs_list, val_history['rmse'], label='Val RMSE', marker='s')
            axes[0].axvline(best_epoch, color='r', linestyle='--', label=f'Best (epoch {best_epoch})')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('RMSE')
            axes[0].set_title(f'{model_name} - RMSE over Epochs')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(epochs_list, train_history['nrmse'], label='Train NRMSE', marker='o')
            axes[1].plot(epochs_list, val_history['nrmse'], label='Val NRMSE', marker='s')
            axes[1].axvline(best_epoch, color='r', linestyle='--', label=f'Best (epoch {best_epoch})')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('NRMSE')
            axes[1].set_title(f'{model_name} - NRMSE over Epochs')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_training_curves.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  ⚠️  Warning: Could not save training curves: {e}")
        
        # Error distribution analysis
        if len(val_history['errors']) > 0:
            best_val_errors = val_history['errors'][best_epoch - 1]
            analyze_error_distribution(best_val_errors, f"{model_name}_val", RESULTS_DIR)
        
        return best_rmse, best_epoch, train_history, val_history
    
    dualbranch_model = DualBranchEEGModel(
        n_channels=128, n_times=n_times, dropout=0.3, use_gat=False
    )
    results['DualBranchEEGModel'] = train_gnn_model(
        dualbranch_model, train_loader, val_loader, 'DualBranchEEGModel',
        edge_index_t, edge_weights_t
    )

# Domain Adaptation Models
print("\n--- Domain Adaptation Models ---")
# Get number of domains (subjects)
unique_subjects = train_windows.get_metadata()["subject"].unique()
n_domains = len(unique_subjects)
print(f"   Number of domains (subjects): {n_domains}")

# Get feature dimension
temp_model = EEGNetv4(n_chans=n_chans, n_times=n_times, n_outputs=1)
with torch.no_grad():
    dummy_input = torch.randn(1, n_chans, n_times)
    try:
        if hasattr(temp_model, 'classifier'):
            original_classifier = temp_model.classifier
            temp_model.classifier = nn.Identity()
            feat_output = temp_model(dummy_input)
            feature_dim = feat_output.shape[1]
            temp_model.classifier = original_classifier
        else:
            feature_dim = 64
    except:
        feature_dim = 64

dann_model = DANN(n_chans=n_chans, n_times=n_times, n_domains=n_domains, feature_dim=feature_dim)
results['DANN'] = train_neural_network(
    dann_model, train_loader, val_loader, 'DANN', return_domain=False
)

dannmodel_model = DANNModel(n_chans=n_chans, n_times=n_times, n_domains=n_domains)
results['DANNModel'] = train_neural_network(
    dannmodel_model, train_loader, val_loader, 'DANNModel', return_domain=False
)

cnn_trans_dann_model = CNNTransformerDANN(
    n_chans=n_chans, n_times=n_times, n_domains=n_domains, feature_dim=feature_dim
)
results['CNNTransformerDANN'] = train_neural_network(
    cnn_trans_dann_model, train_loader, val_loader, 'CNNTransformerDANN', return_domain=False
)

# Labram Model
print("\n--- Labram Model ---")
try:
    labram_model = Labram(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ)
    results['Labram'] = train_neural_network(
        labram_model, train_loader, val_loader, 'Labram'
    )
except Exception as e:
    print(f"⚠️  Could not train Labram: {e}")

# ============================================================
# 6. FINAL TESTING ON RELEASE 11
# ============================================================
if test_loader is not None:
    print("\n" + "=" * 70)
    print("FINAL TESTING ON RELEASE 11")
    print("=" * 70)
    
    test_results = {}
    
    # Test neural networks
    neural_network_models = {
        'CNN1D': lambda: CNN1D(n_chans=n_chans, n_times=n_times),
        'SimpleEEGNet': lambda: SimpleEEGNet(n_chans=n_chans, n_times=n_times),
        'EEGNet_Custom': lambda: EEGNet(n_chans=n_chans, n_times=n_times),
        'CNN_only': lambda: CNNOnly(n_chans=n_chans, n_times=n_times),
        'EEGNeX_Custom': lambda: EEGNeX_Custom(n_chans=n_chans, n_times=n_times),
        'EEGNeX_Braindecode': lambda: EEGNeX(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ),
        'EEGMiner': lambda: EEGMiner(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ),
        'Deep4Net': lambda: Deep4Net(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ),
        'ATCNet': lambda: ATCNet(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ),
        'SimpleEEGConformer': lambda: SimpleEEGConformer(n_chans=n_chans, n_times=n_times),
        'EEGConformer': lambda: EEGConformer(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ),
        'CNN_Transformer': lambda: CNNTransformer(n_chans=n_chans, n_times=n_times, dropout=0.3),
        'HybridEEGRegressor': lambda: HybridEEGRegressor(n_chans=n_chans, n_times=n_times, dropout=0.3),
        'DANN': lambda: DANN(n_chans=n_chans, n_times=n_times, n_domains=n_domains, feature_dim=feature_dim),
        'DANNModel': lambda: DANNModel(n_chans=n_chans, n_times=n_times, n_domains=n_domains),
        'CNNTransformerDANN': lambda: CNNTransformerDANN(n_chans=n_chans, n_times=n_times, n_domains=n_domains, feature_dim=feature_dim),
        'Labram': lambda: Labram(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ),
    }
    
    for model_name, model_constructor in neural_network_models.items():
        model_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pt")
        if os.path.exists(model_path):
            print(f"\n🧪 Testing {model_name}...")
            try:
                model = model_constructor()
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model = model.to(DEVICE)
                model.eval()
                
                test_preds = []
                test_targets = []
                
                with torch.no_grad():
                    for batch in test_loader:
                        X, y, _ = batch[0], batch[1], batch[2] if len(batch) > 2 else None
                        X = X.to(DEVICE).float()
                        y = y.to(DEVICE).float().view(-1)
                        
                        # Handle models with special forward signatures
                        if model_name in ['DANN', 'DANNModel', 'CNNTransformerDANN']:
                            pred = model(X, return_domain=False).view(-1)
                        elif model_name == 'DualBranchEEGModel' and HAS_PYG:
                            pred = model(X, edge_index_t, edge_weights_t).view(-1)
                        else:
                            pred = model(X).view(-1)
                        
                        test_preds.append(pred.cpu().numpy())
                        test_targets.append(y.cpu().numpy())
                
                test_preds = np.concatenate(test_preds)
                test_targets = np.concatenate(test_targets)
                test_rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2))
                test_nrmse = test_rmse / (np.mean(np.abs(test_targets)) + 1e-8)
                test_results[model_name] = {'rmse': test_rmse, 'nrmse': test_nrmse}
                print(f"  ✅ Test RMSE: {test_rmse:.4f} | Test NRMSE: {test_nrmse:.4f}")
                
                # Error distribution analysis
                test_errors = test_preds - test_targets
                analyze_error_distribution(test_errors, f"{model_name}_test", RESULTS_DIR)
            except Exception as e:
                print(f"  ❌ Error testing {model_name}: {e}")
    
    # Test GNN models separately if available
    if HAS_PYG:
        gnn_model_path = os.path.join(RESULTS_DIR, "DualBranchEEGModel_best.pt")
        if os.path.exists(gnn_model_path):
            print(f"\n🧪 Testing DualBranchEEGModel...")
            try:
                model = DualBranchEEGModel(n_channels=128, n_times=n_times, dropout=0.3, use_gat=False)
                model.load_state_dict(torch.load(gnn_model_path, map_location=DEVICE))
                model = model.to(DEVICE)
                model.eval()
                
                test_preds = []
                test_targets = []
                
                with torch.no_grad():
                    for batch in test_loader:
                        X, y, _ = batch[0], batch[1], batch[2] if len(batch) > 2 else None
                        X = X.to(DEVICE).float()
                        y = y.to(DEVICE).float().view(-1)
                        pred = model(X, edge_index_t, edge_weights_t).view(-1)
                        test_preds.append(pred.cpu().numpy())
                        test_targets.append(y.cpu().numpy())
                
                test_preds = np.concatenate(test_preds)
                test_targets = np.concatenate(test_targets)
                test_rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2))
                test_nrmse = test_rmse / (np.mean(np.abs(test_targets)) + 1e-8)
                test_results['DualBranchEEGModel'] = {'rmse': test_rmse, 'nrmse': test_nrmse}
                print(f"  ✅ Test RMSE: {test_rmse:.4f} | Test NRMSE: {test_nrmse:.4f}")
                
                # Error distribution analysis
                test_errors = test_preds - test_targets
                analyze_error_distribution(test_errors, "DualBranchEEGModel_test", RESULTS_DIR)
            except Exception as e:
                print(f"  ❌ Error testing DualBranchEEGModel: {e}")
    
    # Test linear/tree models
    for model_name in ['LinearRegression', 'Ridge', 'Lasso', 'RandomForest']:
        if HAS_XGB:
            if model_name == 'XGBoost':
                continue  # Will handle separately
        
        model_path = os.path.join(RESULTS_DIR, f"{model_name}_model.pkl")
        if os.path.exists(model_path):
            print(f"\n🧪 Testing {model_name}...")
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            test_pred = model.predict(X_test_scaled)
            test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
            test_nrmse = test_rmse / (np.mean(np.abs(y_test)) + 1e-8)
            test_results[model_name] = {'rmse': test_rmse, 'nrmse': test_nrmse}
            print(f"  ✅ Test RMSE: {test_rmse:.4f} | Test NRMSE: {test_nrmse:.4f}")
            
            # Error distribution analysis
            test_errors = test_pred - y_test
            analyze_error_distribution(test_errors, f"{model_name}_test", RESULTS_DIR)
    
    if HAS_XGB:
        model_path = os.path.join(RESULTS_DIR, "XGBoost_model.pkl")
        if os.path.exists(model_path):
            print(f"\n🧪 Testing XGBoost...")
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            test_pred = model.predict(X_test_scaled)
            test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
            test_nrmse = test_rmse / (np.mean(np.abs(y_test)) + 1e-8)
            test_results['XGBoost'] = {'rmse': test_rmse, 'nrmse': test_nrmse}
            print(f"  ✅ Test RMSE: {test_rmse:.4f} | Test NRMSE: {test_nrmse:.4f}")
            
            # Error distribution analysis
            test_errors = test_pred - y_test
            analyze_error_distribution(test_errors, "XGBoost_test", RESULTS_DIR)

# ============================================================
# 7. SAVE RESULTS SUMMARY
# ============================================================
summary = {
    'data_split': {
        'train_releases': TRAIN_RELEASES,
        'val_release': VAL_RELEASE,
        'test_release': TEST_RELEASE,
        'train_samples': len(train_windows),
        'val_samples': len(val_windows),
        'test_samples': len(test_windows) if test_windows is not None else 0,
    },
    'hyperparameters': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'weight_decay': WEIGHT_DECAY,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'early_stopping_min_delta': EARLY_STOPPING_MIN_DELTA,
    },
    'validation_results': {},
    'test_results': test_results if 'test_results' in locals() else {},
    'timestamp': TIMESTAMP,
}

# Add validation results
for model_name, result in results.items():
    if isinstance(result, tuple) and len(result) >= 2:
        if isinstance(result[0], float):  # Neural network
            val_hist = result[3] if len(result) > 3 else {}
            best_nrmse = val_hist.get('nrmse', [0])[-1] if val_hist and 'nrmse' in val_hist and len(val_hist['nrmse']) > 0 else None
            summary['validation_results'][model_name] = {
                'best_rmse': float(result[0]),
                'best_epoch': int(result[1]) if len(result) > 1 else None,
                'best_nrmse': float(best_nrmse) if best_nrmse is not None else None,
            }
        else:  # Linear/tree model
            summary['validation_results'][model_name] = {
                'val_rmse': float(result[0]),
                'best_params': result[1] if result[1] is not None else {},
            }

with open(os.path.join(RESULTS_DIR, "results_summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
print("\nValidation Results (Release 5):")
for model_name, result in summary['validation_results'].items():
    if 'best_rmse' in result:
        # Try to get NRMSE from history if available
        val_nrmse = result.get('best_nrmse', 'N/A')
        if val_nrmse != 'N/A':
            print(f"  {model_name:20s}: RMSE = {result['best_rmse']:.4f} | NRMSE = {val_nrmse:.4f} (epoch {result['best_epoch']})")
        else:
            print(f"  {model_name:20s}: RMSE = {result['best_rmse']:.4f} (epoch {result['best_epoch']})")
    else:
        print(f"  {model_name:20s}: RMSE = {result['val_rmse']:.4f}")

if 'test_results' in summary and summary['test_results']:
    print("\nTest Results (Release 11):")
    for model_name, test_result in summary['test_results'].items():
        if isinstance(test_result, dict):
            print(f"  {model_name:20s}: RMSE = {test_result['rmse']:.4f} | NRMSE = {test_result['nrmse']:.4f}")
        else:
            print(f"  {model_name:20s}: RMSE = {test_result:.4f}")

print(f"\n✅ All results saved to: {RESULTS_DIR}/")
print("=" * 70)

# ============================================================
# 8. SAVE SUMMARY OF ALL SAVED MODELS
# ============================================================
print("\n" + "=" * 70)
print("SAVED MODEL FILES SUMMARY")
print("=" * 70)

saved_models = {
    'neural_networks': [],
    'linear_tree_models': [],
    'scalers': [],
    'checkpoints': [],
    'metadata': []
}

# Check for neural network models
neural_network_names = [
    'CNN1D', 'SimpleEEGNet', 'EEGNet_Custom', 'CNN_only', 'EEGNeX_Custom',
    'EEGNeX_Braindecode', 'EEGMiner', 'Deep4Net', 'ATCNet', 'SimpleEEGConformer',
    'EEGConformer', 'CNN_Transformer', 'HybridEEGRegressor', 'DANN', 'DANNModel',
    'CNNTransformerDANN', 'Labram', 'DualBranchEEGModel'
]

for model_name in neural_network_names:
    best_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pt")
    checkpoint_path = os.path.join(RESULTS_DIR, f"{model_name}_checkpoint.pt")
    if os.path.exists(best_path):
        saved_models['neural_networks'].append(f"{model_name}_best.pt")
    if os.path.exists(checkpoint_path):
        saved_models['checkpoints'].append(f"{model_name}_checkpoint.pt")

# Check for linear/tree models
linear_tree_names = ['LinearRegression', 'Ridge', 'Lasso', 'RandomForest']
if HAS_XGB:
    linear_tree_names.append('XGBoost')

for model_name in linear_tree_names:
    model_path = os.path.join(RESULTS_DIR, f"{model_name}_model.pkl")
    scaler_path = os.path.join(RESULTS_DIR, f"{model_name}_scaler.pkl")
    metadata_path = os.path.join(RESULTS_DIR, f"{model_name}_metadata.json")
    if os.path.exists(model_path):
        saved_models['linear_tree_models'].append(f"{model_name}_model.pkl")
    if os.path.exists(scaler_path):
        saved_models['scalers'].append(f"{model_name}_scaler.pkl")
    if os.path.exists(metadata_path):
        saved_models['metadata'].append(f"{model_name}_metadata.json")

print(f"\n📦 Neural Network Models (weights): {len(saved_models['neural_networks'])}")
for fname in sorted(saved_models['neural_networks']):
    print(f"   ✅ {fname}")

print(f"\n📦 Neural Network Checkpoints (full): {len(saved_models['checkpoints'])}")
for fname in sorted(saved_models['checkpoints']):
    print(f"   ✅ {fname}")

print(f"\n📦 Linear/Tree Models: {len(saved_models['linear_tree_models'])}")
for fname in sorted(saved_models['linear_tree_models']):
    print(f"   ✅ {fname}")

print(f"\n📦 Scalers (for linear/tree models): {len(saved_models['scalers'])}")
for fname in sorted(saved_models['scalers']):
    print(f"   ✅ {fname}")

print(f"\n📦 Metadata Files: {len(saved_models['metadata'])}")
for fname in sorted(saved_models['metadata']):
    print(f"   ✅ {fname}")

total_saved = (len(saved_models['neural_networks']) + 
               len(saved_models['checkpoints']) + 
               len(saved_models['linear_tree_models']) + 
               len(saved_models['scalers']) + 
               len(saved_models['metadata']))

print(f"\n✅ Total files saved: {total_saved}")
print(f"📁 All files in: {RESULTS_DIR}/")
print("=" * 70)

