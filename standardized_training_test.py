#!/usr/bin/env python3
"""
Standardized Testing Script for ALL 21 Models

TESTING MODE: This script only loads test data and tests existing models.
No training is performed. Models are loaded from previous result directories.

This script ensures all models are tested on identical data splits:
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
import warnings
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
from scipy import stats
from scipy.stats import skew, kurtosis
import copy
import json
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import logging

# Suppress expected warnings during feature extraction
# (precision loss in skew/kurtosis is expected when data has low variance)
warnings.filterwarnings('ignore', category=RuntimeWarning,
                       message='.*Precision loss.*|.*invalid value.*|.*divide by zero.*')

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

# TESTING MODE: Set to True to skip all training and only test existing models
TESTING_MODE_ONLY = True

# Results directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"standardized_results_{TIMESTAMP}"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set up logging
log_file = os.path.join(RESULTS_DIR, "test_results.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ],
    force=True  # Force reconfiguration
)
logger = logging.getLogger(__name__)
logger.info("=" * 70)
logger.info("TESTING MODE - Test Results Log")
logger.info(f"Results directory: {RESULTS_DIR}")
logger.info("=" * 70)
# Force flush to ensure log is written immediately
for handler in logger.handlers:
    handler.flush()

# Use only specific result directories
print("\n" + "=" * 70)
print("USING SPECIFIED RESULT DIRECTORIES")
print("=" * 70)

# Only use these two directories
PREVIOUS_RESULT_DIRS = [
    "standardized_results_20251205_204816",
    "standardized_results_20251212_130126"
]

# Verify directories exist
existing_dirs = []
for dir_name in PREVIOUS_RESULT_DIRS:
    if Path(dir_name).exists():
        existing_dirs.append(dir_name)
        print(f"  ✅ {dir_name}")
    else:
        print(f"  ⚠️  {dir_name} (not found)")

PREVIOUS_RESULT_DIRS = existing_dirs

if len(PREVIOUS_RESULT_DIRS) == 0:
    print("  ❌ No specified directories found. Will only use current directory.")
else:
    print(f"\n✅ Will scan {len(PREVIOUS_RESULT_DIRS)} directory(ies) for existing models.")
print("=" * 70)

print(f"\n📁 Results will be saved to: {RESULTS_DIR}/")
print("=" * 70)
print("STANDARDIZED DATA SPLIT:")
print(f"  Training: Releases {TRAIN_RELEASES}")
print(f"  Validation: Release {VAL_RELEASE}")
print(f"  Testing: Release {TEST_RELEASE}")
print("=" * 70)
print("\n💡 NOTE: You may see warnings about 'Missing: _eeg.bdf' files.")
print("   This is EXPECTED and can be safely ignored.")
print("   EEGChallengeDataset checks for ALL tasks, but we only have")
print("   contrastChangeDetection task files. The missing files are for other tasks.")
print("=" * 70)

# ============================================================
# 1. DATA LOADING WITH STANDARDIZED SPLIT
# ============================================================
def load_release(release_id, task="contrastChangeDetection"):
    """
    Load data from a specific release folder (same approach as eeg_conformer_reg.py)
    
    Args:
        release_id (int): Release number (1-11)
        task (str): Task name
    
    Returns:
        EEGChallengeDataset: Loaded dataset or None if not found
    """
    release_dir = Path(f"data_new_new/release_{release_id}")
    
    if not release_dir.exists():
        print(f"⚠️  Release {release_id} folder not found: {release_dir.resolve()}")
        return None
    
    # Check if any *-bdf folders exist in the release directory
    bdf_folders = list(release_dir.glob("*-bdf"))
    if not bdf_folders:
        print(f"⚠️  No *-bdf folders found in {release_dir.resolve()}")
        return None
    
    print(f"Loading Release R{release_id} from: {release_dir.resolve()}")
    print(f"   Found dataset folder(s): {[f.name for f in bdf_folders]}")
    
    try:
        dataset = EEGChallengeDataset(
            task=task,
            release=f"R{release_id}",
            cache_dir=release_dir,
            mini=False,  # Always use full dataset, not mini
            download=False  # Never download - use only local data
        )
        
        if len(dataset.datasets) > 0:
            print(f"✅ Loaded {len(dataset.datasets)} recordings from Release R{release_id}")
            return dataset
        else:
            print(f"⚠️  Release {release_id} loaded but has no datasets")
            return None
    except ValueError as e:
        error_msg = str(e)
        if "does not exist" in error_msg or "Offline mode" in error_msg:
            # Extract the expected folder name from the error if possible
            import re
            match = re.search(r'/([^/]+-bdf)', error_msg)
            if match:
                expected_folder = match.group(1)
                actual_folders = [f.name for f in bdf_folders]
                print(f"   ❌ EEGChallengeDataset expects folder: {expected_folder}")
                print(f"   But found folders: {actual_folders}")
                if expected_folder not in actual_folders and len(actual_folders) > 0:
                    # Try using the actual folder that exists by creating a symlink or trying alternative approach
                    print(f"   💡 Attempting to work around folder name mismatch...")
                    actual_folder = actual_folders[0]
                    actual_folder_path = release_dir / actual_folder
                    
                    # Try creating a symlink to the expected folder name
                    try:
                        expected_folder_path = release_dir / expected_folder
                        actual_folder_path = release_dir / actual_folder
                        
                        if not expected_folder_path.exists() and actual_folder_path.exists():
                            # Create symlink from expected name to actual folder
                            print(f"   🔗 Creating symlink: {expected_folder} -> {actual_folder}")
                            print(f"   💡 TIP: If this hangs, you can create the symlink manually:")
                            print(f"      cd {release_dir}")
                            print(f"      ln -s {actual_folder} {expected_folder}")
                            try:
                                expected_folder_path.symlink_to(actual_folder, target_is_directory=True)
                                print(f"   ✅ Symlink created successfully")
                                
                                # Verify symlink exists
                                if expected_folder_path.exists():
                                    print(f"   ✅ Symlink verified: {expected_folder_path} -> {expected_folder_path.readlink()}")
                                else:
                                    print(f"   ⚠️  Warning: Symlink created but path doesn't exist")
                            except (OSError, PermissionError) as symlink_err:
                                print(f"   ⚠️  Could not create symlink (may need permissions): {str(symlink_err)[:200]}")
                        elif expected_folder_path.exists() and expected_folder_path.is_symlink():
                            print(f"   ✅ Symlink already exists: {expected_folder_path} -> {expected_folder_path.readlink()}")
                        
                        # Try loading again with the symlink in place (if it exists now)
                        if expected_folder_path.exists():
                            print(f"   🔄 Attempting to load dataset with symlink in place...")
                            print(f"   ⏳ This may take a while as EEGChallengeDataset scans files...")
                            logger.info(f"Attempting to load Release R{release_id} with symlink workaround")
                            for handler in logger.handlers:
                                handler.flush()
                            try:
                                print(f"   📦 Creating EEGChallengeDataset object...")
                                dataset = EEGChallengeDataset(
                                    task=task,
                                    release=f"R{release_id}",
                                    cache_dir=release_dir,
                                    mini=False,
                                    download=False
                                )
                                print(f"   ✅ EEGChallengeDataset object created")
                                logger.info("EEGChallengeDataset object created successfully")
                                for handler in logger.handlers:
                                    handler.flush()
                                print(f"   ✅ Dataset object created, checking recordings...")
                                if len(dataset.datasets) > 0:
                                    print(f"✅ Loaded {len(dataset.datasets)} recordings from Release R{release_id} (using symlink workaround)")
                                    logger.info(f"Successfully loaded Release R{release_id} using symlink workaround: {len(dataset.datasets)} recordings")
                                    return dataset
                                else:
                                    print(f"   ⚠️  Dataset loaded but has no recordings")
                            except Exception as e2:
                                # Clean up symlink if it was created
                                try:
                                    if expected_folder_path.is_symlink():
                                        expected_folder_path.unlink()
                                        print(f"   🧹 Cleaned up symlink")
                                except:
                                    pass
                                print(f"   ❌ Symlink workaround failed: {str(e2)}")
                                import traceback
                                traceback.print_exc()
                                logger.error(f"Symlink workaround failed: {str(e2)}")
                    except Exception as symlink_error:
                        print(f"   ⚠️  Symlink creation error: {str(symlink_error)[:200]}")
                    
                    print(f"   💡 Folder name mismatch - the dataset may not be available for this release")
            else:
                print(f"   ❌ {error_msg[:200]}")
        else:
            print(f"   ❌ {error_msg[:200]}")
        return None
    except Exception as e:
        print(f"⚠️  Failed to load Release {release_id}: {str(e)[:200]}")
        return None


def preprocess_and_window_dataset(dataset, release_id):
    """Preprocess and create windows for a dataset"""
    print(f"\n📊 Preprocessing Release {release_id}...")
    logger.info(f"Starting preprocessing for Release {release_id}")
    
    # Preprocessing
    print("   Step 1/3: Annotating trials with targets...")
    logger.info("Step 1/3: Annotating trials with targets")
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
    # Use n_jobs=1 and ensure we're not triggering downloads during preprocessing
    # The dataset should already be loaded with download=False
    print("   Step 2/3: Running preprocessing (this may take a while)...")
    logger.info("Step 2/3: Running preprocessing")
    preprocess(dataset, transformation_offline, n_jobs=1)
    print("   ✅ Preprocessing complete")
    logger.info("Preprocessing complete")
    
    # Keep only recordings with stimulus anchors
    print("   Step 3/3: Filtering and creating windows...")
    logger.info("Step 3/3: Filtering and creating windows")
    dataset_filtered = keep_only_recordings_with(ANCHOR, dataset)
    print(f"   Filtered to {len(dataset_filtered.datasets)} recordings with stimulus anchors")
    logger.info(f"Filtered to {len(dataset_filtered.datasets)} recordings with stimulus anchors")
    
    # Create windows
    print("   Creating windows (preloading data - this may take a while)...")
    logger.info("Creating windows with preload=True")
    single_windows = create_windows_from_events(
        dataset_filtered,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )
    print("   ✅ Windows created")
    logger.info(f"Windows created: {len(single_windows)} windows")
    
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
print("LOADING TEST DATA ONLY (TESTING MODE)")
print("=" * 70)
print("⚠️  TESTING MODE: Only loading test release. Skipping training/validation data.")
print("=" * 70)

# Skip training and validation - only load test release
train_windows = None
val_windows = None
train_loader = None
val_loader = None

# Load test release
print(f"\n📥 Loading TEST release: {TEST_RELEASE}")
logger.info(f"Starting to load TEST release: {TEST_RELEASE}")
for handler in logger.handlers:
    handler.flush()
test_dataset = load_release(TEST_RELEASE)
if test_dataset is None:
    print(f"\n❌ ERROR: Test release {TEST_RELEASE} could not be loaded!")
    print(f"   This is required for testing models.")
    print(f"   Possible solutions:")
    print(f"   1. Check if the folder name matches what EEGChallengeDataset expects")
    print(f"   2. Verify the data exists in data_new_new/release_{TEST_RELEASE}/")
    print(f"   3. Check if you need to rename the folder to match expected name")
    print(f"\n   Script will exit since test data is required.")
    raise ValueError(f"Test release {TEST_RELEASE} is required but could not be loaded!")

test_windows = preprocess_and_window_dataset(test_dataset, TEST_RELEASE)
print(f"✅ Test windows: {len(test_windows)}")

# Create test dataloader
test_loader = DataLoader(test_windows, batch_size=BATCH_SIZE, shuffle=False)

# Detect input dimensions from test data
sample_X, _, _ = next(iter(test_loader))
_, n_chans, n_times = sample_X.shape
print(f"\n📐 Detected input shape: n_chans={n_chans}, n_times={n_times}")

# ============================================================
# 2. FEATURE EXTRACTION FOR LINEAR/TREE MODELS
# ============================================================
# NOTE: Feature extraction is commented out since linear/tree models are disabled
# Uncomment this section if you want to train linear/tree models

# def extract_features_from_window(window_np, fs=100.0):
#     """
#     Extract features from EEG window (same as submission_5/submission.py)
#     Returns: feature vector of length 1161 (129 channels * 9 features per channel)
#     """
#     import warnings
#     from scipy.signal import welch
#     from scipy.stats import skew, kurtosis
#     
#     def bandpower(data, fs, fmin, fmax):
#         """Compute bandpower using Welch's method"""
#         f, Pxx = welch(data, fs=fs, nperseg=min(256, len(data)), nfft=1024)
#         band = (f >= fmin) & (f <= fmax)
#         # Use trapezoid instead of deprecated trapz
#         return np.trapezoid(Pxx[band], f[band])
#     
#     # Basic stats
#     means = window_np.mean(axis=1)
#     stds = window_np.std(axis=1) + 1e-8
#     
#     # Suppress warnings for skew/kurtosis when data has low variance (expected)
#     with warnings.catch_warnings():
#         warnings.filterwarnings('ignore', category=RuntimeWarning, 
#                                 message='.*Precision loss.*|.*invalid value.*|.*divide by zero.*')
#         skews = skew(window_np, axis=1, bias=False, nan_policy='omit')
#         kurts = kurtosis(window_np, axis=1, fisher=True, bias=False, nan_policy='omit')
#     
#     # Frequency bands
#     bands = {
#         "delta": (1, 4),
#         "theta": (4, 8),
#         "alpha": (8, 13),
#         "beta": (13, 30),
#         "gamma": (30, 50)
#     }
#     
#     # Bandpower for each channel
#     band_feats = []
#     for ch in range(window_np.shape[0]):
#         ch_data = window_np[ch, :]
#         ch_bandpowers = [bandpower(ch_data, fs, fmin, fmax) for fmin, fmax in bands.values()]
#         band_feats.append(ch_bandpowers)
#     band_feats = np.array(band_feats)
#     
#     # Combine: mean, std, skew, kurt, 5 bandpowers = 9 features per channel
#     feats = np.concatenate([
#         means[:, None],
#         stds[:, None],
#         skews[:, None],
#         kurts[:, None],
#         band_feats
#     ], axis=1).reshape(-1)
#     
#     feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
#     return feats
#
#
# def extract_features_from_dataset(windows):
#     """Extract features from all windows in dataset"""
#     import warnings
#     print("🔧 Extracting features from dataset...")
#     features = []
#     targets = []
#     
#     # Suppress deprecation warnings for array-to-scalar conversion
#     warnings.filterwarnings('ignore', category=DeprecationWarning, 
#                           message='.*Conversion of an array.*')
#     
#     for i in tqdm(range(len(windows)), desc="Extracting features"):
#         window_data = windows[i][0]  # (n_chans, n_times)
#         target = windows[i][1]  # scalar or array
#         
#         # Convert to numpy if needed
#         if isinstance(window_data, torch.Tensor):
#             window_np = window_data.numpy()
#         else:
#             window_np = np.array(window_data)
#         
#         feat = extract_features_from_window(window_np, fs=SFREQ)
#         features.append(feat)
#         
#         # Handle target conversion properly (handle both scalar and array cases)
#         if isinstance(target, (np.ndarray, torch.Tensor)):
#             if isinstance(target, torch.Tensor):
#                 target = target.item() if target.numel() == 1 else float(target.flatten()[0])
#             else:
#                 target = target.item() if target.size == 1 else float(target.flatten()[0])
#         else:
#             target = float(target)
#         targets.append(target)
#     
#     return np.array(features), np.array(targets)


# Extract features for linear/tree models
# NOTE: Commented out since linear/tree models are disabled
# print("\n" + "=" * 70)
# print("EXTRACTING FEATURES FOR LINEAR/TREE MODELS")
# print("=" * 70)

# X_train, y_train = extract_features_from_dataset(train_windows)
# X_val, y_val = extract_features_from_dataset(val_windows)
# if test_windows is not None:
#     X_test, y_test = extract_features_from_dataset(test_windows)
# else:
#     X_test, y_test = None, None

# print(f"✅ Training features: {X_train.shape}")
# print(f"✅ Validation features: {X_val.shape}")
# if X_test is not None:
#     print(f"✅ Test features: {X_test.shape}")

# Scale features
# NOTE: Scaler initialization commented out since linear/tree models are disabled
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)
# if X_test is not None:
#     X_test_scaled = scaler.transform(X_test)

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
        
        # Calculate the actual size after convolutions and pooling
        # Use adaptive pooling to avoid size calculation issues
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
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
        # Use adaptive pooling to get fixed size output
        x = self.adaptive_pool(x)  # (batch, 256, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)
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
        
        # Calculate actual output dimension after pooling
        # spatial_pool: kernel=4, default stride=4 -> divides by 4
        # feature_pool: kernel=8, default stride=8 -> divides by 8
        # Total reduction: 4 * 8 = 32
        # After spatial_pool: n_times // 4
        # After feature_pool: (n_times // 4) // 8 = n_times // 32
        spatial_pool_stride = 4  # default stride equals kernel
        feature_pool_stride = 8  # default stride equals kernel
        pooled_time_dim = (n_times // spatial_pool_stride) // feature_pool_stride
        self.classifier = nn.Sequential(
            nn.Linear(32 * pooled_time_dim, 128),
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
        # Calculate actual pooling output size using PyTorch formula:
        # output_size = floor((input_size - kernel_size) / stride) + 1
        # Note: conv_time preserves n_times due to padding, conv_spatial reduces spatial dim to 1
        pool_kernel_size = 75
        pool_stride = 15
        pooled_time_dim = (n_times - pool_kernel_size) // pool_stride + 1
        self.cnn_out_dim = n_filters_spatial * pooled_time_dim
        
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
    
    # Set convergence parameters for Lasso to avoid warnings
    if model_name == 'Lasso':
        # Lasso needs more iterations and tighter tolerance for convergence
        default_kwargs = {'max_iter': 5000, 'tol': 1e-4}
    elif model_name == 'Ridge':
        # Ridge usually converges fine, but set max_iter to be safe
        default_kwargs = {'max_iter': 2000}
    else:
        default_kwargs = {}
    
    if param_grid is None:
        # Default: just train without tuning
        model = model_class(**default_kwargs)
        model.fit(X_train_scaled, y_train)
        val_pred = model.predict(X_val_scaled)
        val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        best_params = None
    else:
        # Grid search on validation set
        # Need to add default_kwargs to each parameter combination
        print(f"🔍 Hyperparameter tuning with {len(list(param_grid.values())[0])} combinations...")
        
        # Create estimator with default kwargs
        base_estimator = model_class(**default_kwargs)
        
        model = GridSearchCV(
            base_estimator, 
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
# SKIPPED IN TESTING MODE - Only testing existing models
print("\n" + "=" * 70)
print("TRAINING SKIPPED (TESTING MODE)")
print("=" * 70)
print("⚠️  Skipping all training. Will only test models from previous runs.")
print("=" * 70)

results = {}  # Empty results dict since we're not training

# ============================================================
# LINEAR MODELS
# ============================================================
# print("\n" + "=" * 70)
# print("LINEAR MODELS")
# print("=" * 70)

# # LinearRegression
# results['LinearRegression'] = train_linear_model(
#     LinearRegression, 
#     'LinearRegression'
# )

# # Ridge with hyperparameter tuning
# ridge_params = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
# results['Ridge'] = train_linear_model(
#     Ridge,
#     'Ridge',
#     param_grid=ridge_params
# )

# # Lasso with hyperparameter tuning
# lasso_params = {'alpha': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]}
# results['Lasso'] = train_linear_model(
#     Lasso,
#     'Lasso',
#     param_grid=lasso_params
# )

# ============================================================
# TREE-BASED MODELS
# ============================================================
# print("\n" + "=" * 70)
# print("TREE-BASED MODELS")
# print("=" * 70)

# # RandomForest with hyperparameter tuning
# rf_params = {
#     'n_estimators': [200, 400, 600],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5, 10]
# }
# results['RandomForest'] = train_linear_model(
#     RandomForestRegressor,
#     'RandomForest',
#     param_grid=rf_params
# )

# # XGBoost with hyperparameter tuning
# if HAS_XGB:
#     xgb_params = {
#         'n_estimators': [300, 500, 700],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'max_depth': [4, 6, 8],
#         'reg_lambda': [0.5, 1.0, 2.0]
#     }
#     results['XGBoost'] = train_linear_model(
#         XGBRegressor,
#         'XGBoost',
#         param_grid=xgb_params
#     )

# ============================================================
# NEURAL NETWORK MODELS
# ============================================================
# print("\n" + "=" * 70)
# print("NEURAL NETWORK MODELS")
# print("=" * 70)

# # Simple CNNs
# print("\n--- Simple CNNs ---")
# cnn1d_model = CNN1D(n_chans=n_chans, n_times=n_times)
# results['CNN1D'] = train_neural_network(
#     cnn1d_model, train_loader, val_loader, 'CNN1D'
# )

# simple_eegnet_model = SimpleEEGNet(n_chans=n_chans, n_times=n_times)
# results['SimpleEEGNet'] = train_neural_network(
#     simple_eegnet_model, train_loader, val_loader, 'SimpleEEGNet'
# )

# eegnet_custom_model = EEGNet(n_chans=n_chans, n_times=n_times)
# results['EEGNet_Custom'] = train_neural_network(
#     eegnet_custom_model, train_loader, val_loader, 'EEGNet_Custom'
# )

# cnn_only_model = CNNOnly(n_chans=n_chans, n_times=n_times)
# results['CNN_only'] = train_neural_network(
#     cnn_only_model, train_loader, val_loader, 'CNN_only'
# )

# # Moderate CNNs
# print("\n--- Moderate CNNs ---")
# eegnex_custom_model = EEGNeX_Custom(n_chans=n_chans, n_times=n_times)
# results['EEGNeX_Custom'] = train_neural_network(
#     eegnex_custom_model, train_loader, val_loader, 'EEGNeX_Custom'
# )

# # Braindecode models
# try:
#     eegnex_braindecode = EEGNeX(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ)
#     results['EEGNeX_Braindecode'] = train_neural_network(
#         eegnex_braindecode, train_loader, val_loader, 'EEGNeX_Braindecode'
#     )
# except Exception as e:
#     print(f"⚠️  Could not train EEGNeX (Braindecode): {e}")

# try:
#     eegminer_model = EEGMiner(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ)
#     results['EEGMiner'] = train_neural_network(
#         eegminer_model, train_loader, val_loader, 'EEGMiner'
#     )
# except Exception as e:
#     print(f"⚠️  Could not train EEGMiner: {e}")

# try:
#     deep4net_model = Deep4Net(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ)
#     results['Deep4Net'] = train_neural_network(
#         deep4net_model, train_loader, val_loader, 'Deep4Net'
#     )
# except Exception as e:
#     print(f"⚠️  Could not train Deep4Net: {e}")

# # Attention-based CNN
# print("\n--- Attention-based CNN ---")
# try:
#     atcnet_model = ATCNet(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ)
#     results['ATCNet'] = train_neural_network(
#         atcnet_model, train_loader, val_loader, 'ATCNet'
#     )
# except Exception as e:
#     print(f"⚠️  Could not train ATCNet: {e}")

# # Transformer Models
# print("\n--- Transformer Models ---")
# simple_conformer_model = SimpleEEGConformer(n_chans=n_chans, n_times=n_times)
# results['SimpleEEGConformer'] = train_neural_network(
#     simple_conformer_model, train_loader, val_loader, 'SimpleEEGConformer'
# )

# conformer_model = EEGConformer(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ)
# results['EEGConformer'] = train_neural_network(
#     conformer_model, train_loader, val_loader, 'EEGConformer'
# )

# Hybrid Models
print("\n--- Hybrid Models ---")
# SKIPPED IN TESTING MODE - All training is skipped
if not TESTING_MODE_ONLY:
    # Check if HybridEEGRegressor already exists
    model_name = 'HybridEEGRegressor'
    found, existing_path, source_dir = find_existing_model(model_name, PREVIOUS_RESULT_DIRS)
    if found:
        print(f"✅ Found existing {model_name} in {source_dir}")
        print(f"   Skipping training. Model will be copied to current results directory.")
        copy_model_to_current_dir(existing_path, model_name, source_dir)
        # Load the model to get validation results if available
        try:
            checkpoint_path = os.path.join(source_dir, f"{model_name}_checkpoint.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                best_rmse = checkpoint.get('best_rmse', None)
                best_epoch = checkpoint.get('best_epoch', None)
                val_history = checkpoint.get('val_history', {})
                if best_rmse is not None:
                    results[model_name] = (best_rmse, best_epoch, checkpoint.get('train_history', {}), val_history)
                    print(f"   Validation RMSE: {best_rmse:.4f} (from epoch {best_epoch})")
                else:
                    results[model_name] = (None, None, {}, {})
            else:
                results[model_name] = (None, None, {}, {})
        except Exception as e:
            print(f"   ⚠️  Could not load checkpoint: {e}")
            results[model_name] = (None, None, {}, {})
    else:
        hybrid_model = HybridEEGRegressor(n_chans=n_chans, n_times=n_times, dropout=0.3)
        results['HybridEEGRegressor'] = train_neural_network(
            hybrid_model, train_loader, val_loader, 'HybridEEGRegressor'
        )
else:
    print("   ⏭️  Training skipped (testing mode)")

# GNN Models (requires graph construction)
if HAS_PYG and not TESTING_MODE_ONLY:
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
elif HAS_PYG and TESTING_MODE_ONLY:
    print("\n--- GNN Models ---")
    print("   ⏭️  Training skipped (testing mode)")
    # Build graph from test data for testing GNN models later
    print("🔧 Building functional connectivity graph from test data for GNN testing...")
    sample_data = []
    for i in range(min(2000, len(test_windows))):
        X = test_windows[i][0]
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
    
    # Check if DualBranchEEGModel already exists
    if not TESTING_MODE_ONLY:
        model_name = 'DualBranchEEGModel'
        found, existing_path, source_dir = find_existing_model(model_name, PREVIOUS_RESULT_DIRS)
        if found:
            print(f"✅ Found existing {model_name} in {source_dir}")
            print(f"   Skipping training. Model will be copied to current results directory.")
            copy_model_to_current_dir(existing_path, model_name, source_dir)
            # Load the model to get validation results if available
            try:
                checkpoint_path = os.path.join(source_dir, f"{model_name}_checkpoint.pt")
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    best_rmse = checkpoint.get('best_rmse', None)
                    best_epoch = checkpoint.get('best_epoch', None)
                    val_history = checkpoint.get('val_history', {})
                    if best_rmse is not None:
                        results[model_name] = (best_rmse, best_epoch, checkpoint.get('train_history', {}), val_history)
                        print(f"   Validation RMSE: {best_rmse:.4f} (from epoch {best_epoch})")
                    else:
                        results[model_name] = (None, None, {}, {})
                else:
                    results[model_name] = (None, None, {}, {})
            except Exception as e:
                print(f"   ⚠️  Could not load checkpoint: {e}")
                results[model_name] = (None, None, {}, {})
        else:
            dualbranch_model = DualBranchEEGModel(
                n_channels=128, n_times=n_times, dropout=0.3, use_gat=False
            )
            results['DualBranchEEGModel'] = train_gnn_model(
                dualbranch_model, train_loader, val_loader, 'DualBranchEEGModel',
                edge_index_t, edge_weights_t
            )
    else:
        print("   ⏭️  GNN training skipped (testing mode)")

# Domain Adaptation Models
print("\n--- Domain Adaptation Models ---")
if not TESTING_MODE_ONLY:
    # Get number of domains (subjects)
    unique_subjects = train_windows.get_metadata()["subject"].unique()
    n_domains = len(unique_subjects)
    print(f"   Number of domains (subjects): {n_domains}")
else:
    # In testing mode, use defaults (will be computed from test data if needed)
    n_domains = 10  # Default, will be updated if needed
    print("   ⏭️  Training skipped (testing mode)")

# Get feature dimension (needed for DANN models)
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

# DANN
if not TESTING_MODE_ONLY:
    model_name = 'DANN'
    found, existing_path, source_dir = find_existing_model(model_name, PREVIOUS_RESULT_DIRS)
    if found:
        print(f"✅ Found existing {model_name} in {source_dir}")
        print(f"   Skipping training. Model will be copied to current results directory.")
        copy_model_to_current_dir(existing_path, model_name, source_dir)
        try:
            checkpoint_path = os.path.join(source_dir, f"{model_name}_checkpoint.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                best_rmse = checkpoint.get('best_rmse', None)
                best_epoch = checkpoint.get('best_epoch', None)
                val_history = checkpoint.get('val_history', {})
                if best_rmse is not None:
                    results[model_name] = (best_rmse, best_epoch, checkpoint.get('train_history', {}), val_history)
                    print(f"   Validation RMSE: {best_rmse:.4f} (from epoch {best_epoch})")
                else:
                    results[model_name] = (None, None, {}, {})
            else:
                results[model_name] = (None, None, {}, {})
        except Exception as e:
            print(f"   ⚠️  Could not load checkpoint: {e}")
            results[model_name] = (None, None, {}, {})
    else:
        dann_model = DANN(n_chans=n_chans, n_times=n_times, n_domains=n_domains, feature_dim=feature_dim)
        results['DANN'] = train_neural_network(
            dann_model, train_loader, val_loader, 'DANN', return_domain=False
        )
else:
    print("   ⏭️  DANN training skipped (testing mode)")

# DANNModel
if not TESTING_MODE_ONLY:
    model_name = 'DANNModel'
    found, existing_path, source_dir = find_existing_model(model_name, PREVIOUS_RESULT_DIRS)
    if found:
        print(f"✅ Found existing {model_name} in {source_dir}")
        print(f"   Skipping training. Model will be copied to current results directory.")
        copy_model_to_current_dir(existing_path, model_name, source_dir)
        try:
            checkpoint_path = os.path.join(source_dir, f"{model_name}_checkpoint.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                best_rmse = checkpoint.get('best_rmse', None)
                best_epoch = checkpoint.get('best_epoch', None)
                val_history = checkpoint.get('val_history', {})
                if best_rmse is not None:
                    results[model_name] = (best_rmse, best_epoch, checkpoint.get('train_history', {}), val_history)
                    print(f"   Validation RMSE: {best_rmse:.4f} (from epoch {best_epoch})")
                else:
                    results[model_name] = (None, None, {}, {})
            else:
                results[model_name] = (None, None, {}, {})
        except Exception as e:
            print(f"   ⚠️  Could not load checkpoint: {e}")
            results[model_name] = (None, None, {}, {})
    else:
        dannmodel_model = DANNModel(n_chans=n_chans, n_times=n_times, n_domains=n_domains)
        results['DANNModel'] = train_neural_network(
            dannmodel_model, train_loader, val_loader, 'DANNModel', return_domain=False
        )
else:
    print("   ⏭️  DANNModel training skipped (testing mode)")

# CNNTransformerDANN
if not TESTING_MODE_ONLY:
    model_name = 'CNNTransformerDANN'
    found, existing_path, source_dir = find_existing_model(model_name, PREVIOUS_RESULT_DIRS)
    if found:
        print(f"✅ Found existing {model_name} in {source_dir}")
        print(f"   Skipping training. Model will be copied to current results directory.")
        copy_model_to_current_dir(existing_path, model_name, source_dir)
        try:
            checkpoint_path = os.path.join(source_dir, f"{model_name}_checkpoint.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                best_rmse = checkpoint.get('best_rmse', None)
                best_epoch = checkpoint.get('best_epoch', None)
                val_history = checkpoint.get('val_history', {})
                if best_rmse is not None:
                    results[model_name] = (best_rmse, best_epoch, checkpoint.get('train_history', {}), val_history)
                    print(f"   Validation RMSE: {best_rmse:.4f} (from epoch {best_epoch})")
                else:
                    results[model_name] = (None, None, {}, {})
            else:
                results[model_name] = (None, None, {}, {})
        except Exception as e:
            print(f"   ⚠️  Could not load checkpoint: {e}")
            results[model_name] = (None, None, {}, {})
    else:
        cnn_trans_dann_model = CNNTransformerDANN(
            n_chans=n_chans, n_times=n_times, n_domains=n_domains, feature_dim=feature_dim
        )
        results['CNNTransformerDANN'] = train_neural_network(
            cnn_trans_dann_model, train_loader, val_loader, 'CNNTransformerDANN', return_domain=False
        )
else:
    print("   ⏭️  CNNTransformerDANN training skipped (testing mode)")

# Labram Model
print("\n--- Labram Model ---")
if not TESTING_MODE_ONLY:
    model_name = 'Labram'
    found, existing_path, source_dir = find_existing_model(model_name, PREVIOUS_RESULT_DIRS)
    if found:
        print(f"✅ Found existing {model_name} in {source_dir}")
        print(f"   Skipping training. Model will be copied to current results directory.")
        copy_model_to_current_dir(existing_path, model_name, source_dir)
        try:
            checkpoint_path = os.path.join(source_dir, f"{model_name}_checkpoint.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                best_rmse = checkpoint.get('best_rmse', None)
                best_epoch = checkpoint.get('best_epoch', None)
                val_history = checkpoint.get('val_history', {})
                if best_rmse is not None:
                    results[model_name] = (best_rmse, best_epoch, checkpoint.get('train_history', {}), val_history)
                    print(f"   Validation RMSE: {best_rmse:.4f} (from epoch {best_epoch})")
                else:
                    results[model_name] = (None, None, {}, {})
            else:
                results[model_name] = (None, None, {}, {})
        except Exception as e:
            print(f"   ⚠️  Could not load checkpoint: {e}")
            results[model_name] = (None, None, {}, {})
    else:
        try:
            labram_model = Labram(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ)
            results['Labram'] = train_neural_network(
                labram_model, train_loader, val_loader, 'Labram'
            )
        except Exception as e:
            print(f"⚠️  Could not train Labram: {e}")
else:
    print("   ⏭️  Labram training skipped (testing mode)")

# ============================================================
# 5.5. HELPER FUNCTION TO CHECK FOR EXISTING MODELS
# ============================================================
def find_existing_model(model_name, result_dirs):
    """
    Check if a model already exists in any of the previous result directories.
    
    Returns:
        tuple: (found, model_path, source_dir) where:
            - found: bool, whether model was found
            - model_path: str, path to the model file if found, None otherwise
            - source_dir: str, directory where model was found, None otherwise
    """
    for result_dir in result_dirs:
        if not os.path.exists(result_dir):
            continue
        model_path = os.path.join(result_dir, f"{model_name}_best.pt")
        if os.path.exists(model_path):
            return True, model_path, result_dir
    return False, None, None


def copy_model_to_current_dir(model_path, model_name, source_dir):
    """Copy an existing model to the current results directory"""
    import shutil
    dest_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pt")
    try:
        shutil.copy2(model_path, dest_path)
        print(f"  📋 Copied model from {source_dir} to current results directory")
        return dest_path
    except Exception as e:
        print(f"  ⚠️  Warning: Could not copy model: {e}")
        return model_path  # Return original path if copy fails

def test_all_models_from_directories(result_dirs, test_loader, n_chans, n_times, n_domains=None, feature_dim=None):
    """Test all models found in the specified result directories"""
    if test_loader is None:
        print("⚠️  No test loader available. Skipping model testing from previous runs.")
        return {}
    
    print("\n" + "=" * 70)
    print("TESTING ALL MODELS FROM PREVIOUS RUNS")
    print("=" * 70)
    
    # Model constructors (same as in main test section)
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
        'DANN': lambda: DANN(n_chans=n_chans, n_times=n_times, n_domains=n_domains, feature_dim=feature_dim) if n_domains and feature_dim else None,
        'DANNModel': lambda: DANNModel(n_chans=n_chans, n_times=n_times, n_domains=n_domains) if n_domains else None,
        'CNNTransformerDANN': lambda: CNNTransformerDANN(n_chans=n_chans, n_times=n_times, n_domains=n_domains, feature_dim=feature_dim) if n_domains and feature_dim else None,
        'Labram': lambda: Labram(n_chans=n_chans, n_times=n_times, n_outputs=1, sfreq=SFREQ),
    }
    
    all_test_results = {}  # {model_name: {dir_name: results}}
    models_found = set()
    
    # Scan all directories
    for result_dir in result_dirs:
        if not os.path.exists(result_dir):
            print(f"⚠️  Directory not found: {result_dir}")
            continue
        
        print(f"\n📁 Scanning {result_dir}...")
        dir_results = {}
        
        # Find all _best.pt files
        pt_files = list(Path(result_dir).glob("*_best.pt"))
        if not pt_files:
            print(f"   No model files found in {result_dir}")
            continue
        
        print(f"   Found {len(pt_files)} model file(s)")
        
        for model_path in pt_files:
            model_name = model_path.stem.replace("_best", "")
            
            if model_name not in neural_network_models:
                # Try GNN model
                if model_name == "DualBranchEEGModel" and HAS_PYG:
                    try:
                        model = DualBranchEEGModel(n_channels=128, n_times=n_times, dropout=0.3, use_gat=False)
                        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                        model = model.to(DEVICE)
                        model.eval()
                        
                        # Need to build graph if not already available
                        if 'edge_index_t' not in globals() and 'edge_index_t' not in locals():
                            print("   🔧 Building functional connectivity graph for GNN...")
                            # Use test_windows if train_windows not available
                            windows_to_use = test_windows if train_windows is None else train_windows
                            sample_data = []
                            for i in range(min(2000, len(windows_to_use))):
                                X = windows_to_use[i][0]
                                if isinstance(X, torch.Tensor):
                                    X = X.numpy()
                                sample_data.append(X)
                            sample_data = np.stack(sample_data, axis=0)
                            edge_index_np, edge_weights_np = build_functional_connectivity_graph(sample_data, threshold=0.3)
                            edge_index_t = torch.from_numpy(edge_index_np).long().to(DEVICE)
                            edge_weights_t = torch.from_numpy(edge_weights_np).float().to(DEVICE) if edge_weights_np is not None else None
                        elif 'edge_index_t' in globals():
                            # Use the global edge_index_t if it exists
                            pass
                        
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
                        
                        dir_results[model_name] = {
                            'rmse': test_rmse,
                            'nrmse': test_nrmse,
                            'source_dir': result_dir
                        }
                        models_found.add(model_name)
                        print(f"   ✅ {model_name}: RMSE={test_rmse:.4f}, NRMSE={test_nrmse:.4f}")
                        logger.info(f"{model_name} ({result_dir}): RMSE={test_rmse:.4f}, NRMSE={test_nrmse:.4f}")
                    except Exception as e:
                        print(f"   ❌ Error testing {model_name}: {e}")
                    continue
                else:
                    print(f"   ⚠️  Unknown model: {model_name}, skipping")
                    continue
            
            # Get model constructor
            model_constructor = neural_network_models[model_name]
            # Check if constructor requires parameters that might not be available
            if model_name in ['DANN', 'DANNModel', 'CNNTransformerDANN']:
                if n_domains is None or (model_name in ['DANN', 'CNNTransformerDANN'] and feature_dim is None):
                    print(f"   ⚠️  Cannot construct {model_name} (missing n_domains/feature_dim), skipping")
                    continue
            
            try:
                print(f"   🧪 Testing {model_name}...")
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
                        else:
                            pred = model(X).view(-1)
                        
                        test_preds.append(pred.cpu().numpy())
                        test_targets.append(y.cpu().numpy())
                
                test_preds = np.concatenate(test_preds)
                test_targets = np.concatenate(test_targets)
                test_rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2))
                test_nrmse = test_rmse / (np.mean(np.abs(test_targets)) + 1e-8)
                
                dir_results[model_name] = {
                    'rmse': test_rmse,
                    'nrmse': test_nrmse,
                    'source_dir': result_dir
                }
                models_found.add(model_name)
                print(f"      ✅ RMSE={test_rmse:.4f}, NRMSE={test_nrmse:.4f}")
                logger.info(f"{model_name} ({result_dir}): RMSE={test_rmse:.4f}, NRMSE={test_nrmse:.4f}")
                
            except Exception as e:
                print(f"      ❌ Error testing {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        if dir_results:
            all_test_results[result_dir] = dir_results
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL TESTED MODELS")
    print("=" * 70)
    
    # Group by model name
    model_summary = defaultdict(list)
    for dir_name, dir_results in all_test_results.items():
        for model_name, results in dir_results.items():
            model_summary[model_name].append((dir_name, results))
    
    for model_name in sorted(model_summary.keys()):
        print(f"\n{model_name}:")
        results_list = model_summary[model_name]
        results_list.sort(key=lambda x: x[1]['rmse'])  # Sort by RMSE
        
        for dir_name, results in results_list:
            print(f"  {dir_name:50s}: RMSE={results['rmse']:.4f} | NRMSE={results['nrmse']:.4f}")
        
        # Best result
        best_dir, best_results = results_list[0]
        print(f"  {'BEST:':50s}: RMSE={best_results['rmse']:.4f} | NRMSE={best_results['nrmse']:.4f} (from {best_dir})")
    
    # Save comprehensive results
    summary_path = os.path.join(RESULTS_DIR, "all_previous_runs_test_results.json")
    with open(summary_path, 'w') as f:
        json.dump(all_test_results, f, indent=2)
    print(f"\n✅ All test results saved to: {summary_path}")
    
    return all_test_results

# Run the function to test all models from previous runs
if test_loader is not None:
    # Ensure n_domains and feature_dim are defined (needed for DANN models)
    if 'n_domains' not in locals() or n_domains is None:
        try:
            # Use test_windows if train_windows not available
            windows_to_use = test_windows if train_windows is None else train_windows
            unique_subjects = windows_to_use.get_metadata()["subject"].unique()
            n_domains = len(unique_subjects)
        except:
            n_domains = 10  # Default fallback
    
    if 'feature_dim' not in locals() or feature_dim is None:
        try:
            temp_model = EEGNetv4(n_chans=n_chans, n_times=n_times, n_outputs=1)
            with torch.no_grad():
                dummy_input = torch.randn(1, n_chans, n_times)
                if hasattr(temp_model, 'classifier'):
                    original_classifier = temp_model.classifier
                    temp_model.classifier = nn.Identity()
                    feat_output = temp_model(dummy_input)
                    feature_dim = feat_output.shape[1]
                    temp_model.classifier = original_classifier
                else:
                    feature_dim = 64
        except:
            feature_dim = 64  # Default fallback
    
    previous_runs_results = test_all_models_from_directories(
        PREVIOUS_RESULT_DIRS, test_loader, n_chans, n_times, n_domains, feature_dim
    )

# ============================================================
# 7. FINAL TESTING ON RELEASE 11 (CURRENT RUN)
# ============================================================
if test_loader is not None:
    print("\n" + "=" * 70)
    print("FINAL TESTING ON RELEASE 11")
    print("=" * 70)
    print("NOTE: Testing all models including those from previous runs")
    print("      (will check current directory first, then previous directories)")
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
        # Check if model exists in current directory first
        model_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pt")
        
        # If not in current directory, check previous directories
        if not os.path.exists(model_path):
            found, existing_path, source_dir = find_existing_model(model_name, PREVIOUS_RESULT_DIRS)
            if found:
                model_path = existing_path
                print(f"\n🧪 Testing {model_name} (from {source_dir})...")
            else:
                # Model doesn't exist anywhere, skip
                continue
        else:
            print(f"\n🧪 Testing {model_name}...")
        
        try:
            # Check if constructor requires parameters that might not be available
            if model_name in ['DANN', 'DANNModel', 'CNNTransformerDANN']:
                if n_domains is None or (model_name in ['DANN', 'CNNTransformerDANN'] and feature_dim is None):
                    print(f"  ⚠️  Cannot construct {model_name} (missing n_domains/feature_dim), skipping")
                    continue
            
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
            
            # Store test results
            test_results[model_name] = {
                'rmse': test_rmse,
                'nrmse': test_nrmse,
                'source': 'current' if os.path.exists(os.path.join(RESULTS_DIR, f"{model_name}_best.pt")) else source_dir
            }
            print(f"  ✅ Test RMSE: {test_rmse:.4f} | Test NRMSE: {test_nrmse:.4f}")
            logger.info(f"{model_name}: RMSE={test_rmse:.4f}, NRMSE={test_nrmse:.4f}")
        except Exception as e:
            print(f"  ❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test GNN models separately if available
    if HAS_PYG:
        model_name = 'DualBranchEEGModel'
        gnn_model_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pt")
        
        # If not in current directory, check previous directories
        if not os.path.exists(gnn_model_path):
            found, existing_path, source_dir = find_existing_model(model_name, PREVIOUS_RESULT_DIRS)
            if found:
                gnn_model_path = existing_path
                print(f"\n🧪 Testing {model_name} (from {source_dir})...")
            else:
                # Model doesn't exist anywhere, skip
                gnn_model_path = None
        
        if gnn_model_path and os.path.exists(gnn_model_path):
            if 'gnn_model_path' in locals() and gnn_model_path != os.path.join(RESULTS_DIR, f"{model_name}_best.pt"):
                print(f"\n🧪 Testing {model_name} (from previous run)...")
            else:
                print(f"\n🧪 Testing {model_name}...")
            try:
                # Ensure graph structure is available
                if 'edge_index_t' not in globals():
                    print("   🔧 Building functional connectivity graph for GNN...")
                    sample_data = []
                    for i in range(min(2000, len(train_windows))):
                        X = train_windows[i][0]
                        if isinstance(X, torch.Tensor):
                            X = X.numpy()
                        sample_data.append(X)
                    sample_data = np.stack(sample_data, axis=0)
                    edge_index_np, edge_weights_np = build_functional_connectivity_graph(sample_data, threshold=0.3)
                    edge_index_t = torch.from_numpy(edge_index_np).long().to(DEVICE)
                    edge_weights_t = torch.from_numpy(edge_weights_np).float().to(DEVICE) if edge_weights_np is not None else None
                
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
                
                # Store test results
                test_results[model_name] = {
                    'rmse': test_rmse,
                    'nrmse': test_nrmse,
                    'source': 'current' if os.path.exists(os.path.join(RESULTS_DIR, f"{model_name}_best.pt")) else source_dir
                }
                print(f"  ✅ Test RMSE: {test_rmse:.4f} | Test NRMSE: {test_nrmse:.4f}")
                logger.info(f"{model_name}: RMSE={test_rmse:.4f}, NRMSE={test_nrmse:.4f}")
            except Exception as e:
                print(f"  ❌ Error testing {model_name}: {e}")
                import traceback
                traceback.print_exc()
    
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
            
            # Store test results
            test_results[model_name] = {
                'rmse': test_rmse,
                'nrmse': test_nrmse
            }
            print(f"  ✅ Test RMSE: {test_rmse:.4f} | Test NRMSE: {test_nrmse:.4f}")
    
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
            
            # Store test results
            test_results['XGBoost'] = {
                'rmse': test_rmse,
                'nrmse': test_nrmse
            }
            print(f"  ✅ Test RMSE: {test_rmse:.4f} | Test NRMSE: {test_nrmse:.4f}")

# ============================================================
# 7. SAVE RESULTS SUMMARY
# ============================================================
summary = {
    'data_split': {
        'train_releases': TRAIN_RELEASES,
        'val_release': VAL_RELEASE,
        'test_release': TEST_RELEASE,
        'train_samples': len(train_windows) if train_windows is not None else 0,
        'val_samples': len(val_windows) if val_windows is not None else 0,
        'test_samples': len(test_windows) if test_windows is not None else 0,
        'testing_mode': TESTING_MODE_ONLY,
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
    logger.info("\n" + "=" * 70)
    logger.info("FINAL TEST RESULTS SUMMARY (Release 11)")
    logger.info("=" * 70)
    for model_name, test_result in summary['test_results'].items():
        if isinstance(test_result, dict):
            print(f"  {model_name:20s}: RMSE = {test_result['rmse']:.4f} | NRMSE = {test_result['nrmse']:.4f}")
            logger.info(f"{model_name:20s}: RMSE = {test_result['rmse']:.4f} | NRMSE = {test_result['nrmse']:.4f}")
        else:
            print(f"  {model_name:20s}: RMSE = {test_result:.4f}")
            logger.info(f"{model_name:20s}: RMSE = {test_result:.4f}")
    logger.info("=" * 70)

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

# ============================================================
# 9. CREATE SUMMARY COMPARISON VISUALIZATIONS
# ============================================================
print("\n" + "=" * 70)
print("CREATING SUMMARY COMPARISON VISUALIZATIONS")
print("=" * 70)

try:
    # 1. Bar chart comparing all models' validation RMSE
    model_names = []
    val_rmses = []
    val_nrmses = []
    test_rmses = []
    test_nrmses = []
    
    for model_name, result in summary['validation_results'].items():
        model_names.append(model_name)
        if 'best_rmse' in result:
            val_rmses.append(result['best_rmse'])
            val_nrmses.append(result.get('best_nrmse', None))
        else:
            val_rmses.append(result['val_rmse'])
            val_nrmses.append(None)
        
        # Get test results if available
        if model_name in summary.get('test_results', {}):
            test_result = summary['test_results'][model_name]
            if isinstance(test_result, dict):
                test_rmses.append(test_result.get('rmse', None))
                test_nrmses.append(test_result.get('nrmse', None))
            else:
                test_rmses.append(test_result)
                test_nrmses.append(None)
        else:
            test_rmses.append(None)
            test_nrmses.append(None)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Validation RMSE comparison
    x_pos = np.arange(len(model_names))
    axes[0, 0].barh(x_pos, val_rmses, alpha=0.7, color='steelblue')
    axes[0, 0].set_yticks(x_pos)
    axes[0, 0].set_yticklabels(model_names, fontsize=8)
    axes[0, 0].set_xlabel('Validation RMSE')
    axes[0, 0].set_title('Validation RMSE Comparison (Release 5)')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    # Add value labels on bars
    for i, v in enumerate(val_rmses):
        axes[0, 0].text(v + max(val_rmses) * 0.01, i, f'{v:.4f}', 
                       va='center', fontsize=7)
    
    # 2. Validation NRMSE comparison (if available)
    val_nrmses_available = [n for n in val_nrmses if n is not None]
    if val_nrmses_available:
        model_names_nrmse = [model_names[i] for i, n in enumerate(val_nrmses) if n is not None]
        x_pos_nrmse = np.arange(len(model_names_nrmse))
        axes[0, 1].barh(x_pos_nrmse, val_nrmses_available, alpha=0.7, color='coral')
        axes[0, 1].set_yticks(x_pos_nrmse)
        axes[0, 1].set_yticklabels(model_names_nrmse, fontsize=8)
        axes[0, 1].set_xlabel('Validation NRMSE')
        axes[0, 1].set_title('Validation NRMSE Comparison (Release 5)')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(val_nrmses_available):
            axes[0, 1].text(v + max(val_nrmses_available) * 0.01, i, f'{v:.4f}', 
                           va='center', fontsize=7)
    else:
        axes[0, 1].text(0.5, 0.5, 'NRMSE data not available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Validation NRMSE Comparison')
    
    # 3. Test RMSE comparison (if available)
    test_rmses_available = [r for r in test_rmses if r is not None]
    if test_rmses_available:
        model_names_test = [model_names[i] for i, r in enumerate(test_rmses) if r is not None]
        x_pos_test = np.arange(len(model_names_test))
        axes[1, 0].barh(x_pos_test, test_rmses_available, alpha=0.7, color='mediumseagreen')
        axes[1, 0].set_yticks(x_pos_test)
        axes[1, 0].set_yticklabels(model_names_test, fontsize=8)
        axes[1, 0].set_xlabel('Test RMSE')
        axes[1, 0].set_title('Test RMSE Comparison (Release 11)')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(test_rmses_available):
            axes[1, 0].text(v + max(test_rmses_available) * 0.01, i, f'{v:.4f}', 
                           va='center', fontsize=7)
    else:
        axes[1, 0].text(0.5, 0.5, 'Test results not available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Test RMSE Comparison')
    
    # 4. Test NRMSE comparison (if available)
    test_nrmses_available = [n for n in test_nrmses if n is not None]
    if test_nrmses_available:
        model_names_test_nrmse = [model_names[i] for i, n in enumerate(test_nrmses) if n is not None]
        x_pos_test_nrmse = np.arange(len(model_names_test_nrmse))
        axes[1, 1].barh(x_pos_test_nrmse, test_nrmses_available, alpha=0.7, color='gold')
        axes[1, 1].set_yticks(x_pos_test_nrmse)
        axes[1, 1].set_yticklabels(model_names_test_nrmse, fontsize=8)
        axes[1, 1].set_xlabel('Test NRMSE')
        axes[1, 1].set_title('Test NRMSE Comparison (Release 11)')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(test_nrmses_available):
            axes[1, 1].text(v + max(test_nrmses_available) * 0.01, i, f'{v:.4f}', 
                           va='center', fontsize=7)
    else:
        axes[1, 1].text(0.5, 0.5, 'Test NRMSE data not available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Test NRMSE Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison_summary.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Model comparison summary saved: model_comparison_summary.png")
    
    # 2. Create a single combined bar chart (Val vs Test RMSE)
    if test_rmses_available:
        # Only plot models that have both val and test results
        combined_models = []
        combined_val = []
        combined_test = []
        for i, name in enumerate(model_names):
            if val_rmses[i] is not None and test_rmses[i] is not None:
                combined_models.append(name)
                combined_val.append(val_rmses[i])
                combined_test.append(test_rmses[i])
        
        if combined_models:
            fig, ax = plt.subplots(figsize=(14, max(8, len(combined_models) * 0.4)))
            x_pos = np.arange(len(combined_models))
            width = 0.35
            
            ax.barh(x_pos - width/2, combined_val, width, label='Validation RMSE (R5)', 
                   alpha=0.8, color='steelblue')
            ax.barh(x_pos + width/2, combined_test, width, label='Test RMSE (R11)', 
                   alpha=0.8, color='mediumseagreen')
            
            ax.set_yticks(x_pos)
            ax.set_yticklabels(combined_models, fontsize=9)
            ax.set_xlabel('RMSE', fontsize=11)
            ax.set_title('Model Performance Comparison: Validation vs Test', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "model_comparison_val_vs_test.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Validation vs Test comparison saved: model_comparison_val_vs_test.png")
    
except Exception as e:
    print(f"⚠️  Warning: Could not create summary visualizations: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)

