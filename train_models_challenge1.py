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
import copy

# Braindecode imports
from braindecode.models import EEGNetv4, EEGConformer
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
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", DEVICE)

EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3

# Data configuration
RELEASE_ID = 5
EPOCH_LEN_S = 2.0
SFREQ = 100
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0
ANCHOR = "stimulus_anchor"

# ============================================================
# 1. DATA LOADING
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

RELEASE_DIR = Path(f"data/release_{RELEASE_ID}")

if not RELEASE_DIR.exists():
    print(f"Available releases: {available_releases}")
    if available_releases:
        RELEASE_ID = available_releases[0]
        RELEASE_DIR = Path(f"data/release_{RELEASE_ID}")
        print(f"🔄 Using Release {RELEASE_ID} instead")
    else:
        raise FileNotFoundError("No release folders found in data/")

print(f"📁 Loading data from: {RELEASE_DIR.resolve()}")

# Load the dataset
dataset_ccd = EEGChallengeDataset(
    task="contrastChangeDetection",
    release=f"R{RELEASE_ID}",
    cache_dir=RELEASE_DIR,
    mini=False
)

print(f"Loaded dataset with {len(dataset_ccd.datasets)} recordings from Release R{RELEASE_ID}")

# ============================================================
# 2. PREPROCESSING AND WINDOWING
# ============================================================
print("\n" + "=" * 70)
print("Preprocessing and Creating Windows")
print("=" * 70)

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
preprocess(dataset_ccd, transformation_offline, n_jobs=1)

# Keep only recordings that actually contain stimulus anchors
dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)

# Create single-interval windows (stim-locked, long enough to include the response)
single_windows = create_windows_from_events(
    dataset,
    mapping={ANCHOR: 0},
    trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),  # +0.5 s
    trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),  # +2.5 s
    window_size_samples=int(EPOCH_LEN_S * SFREQ),
    window_stride_samples=SFREQ,
    preload=True,
)

# Injecting metadata into the extra mne annotation
single_windows = add_extras_columns(
    single_windows,
    dataset,
    desc=ANCHOR,
    keys=("target", "rt_from_stimulus", "rt_from_trialstart",
          "stimulus_onset", "response_onset", "correct", "response_type")
)

# ============================================================
# 3. TRAIN/VAL/TEST SPLIT
# ============================================================
print("\n" + "=" * 70)
print("Splitting Data")
print("=" * 70)

meta_information = single_windows.get_metadata()

valid_frac = 0.1
test_frac = 0.1
seed = 2025

subjects = meta_information["subject"].unique()
rng = check_random_state(seed)

# Split subjects
train_subj, temp_subj = train_test_split(
    subjects, test_size=(valid_frac + test_frac), random_state=rng
)
valid_subj, test_subj = train_test_split(
    temp_subj, test_size=test_frac / (valid_frac + test_frac), random_state=rng
)

# Split windows by subject
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

print("Number of examples in each split:")
print(f"Train:\t{len(train_set)}")
print(f"Valid:\t{len(valid_set)}")
print(f"Test:\t{len(test_set)}")

# Create dataloaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Detect dims from one batch
sample_X, _, _ = next(iter(train_loader))
_, n_chans, n_times = sample_X.shape
print(f"\nDetected input shape: n_chans={n_chans}, n_times={n_times}")

# ============================================================
# 4. TRAINING UTILITIES
# ============================================================
def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2))


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_rmse = 0
    steps = 0
    for X, y, _ in tqdm(loader, desc="Train"):
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
        for X, y, _ in tqdm(loader, desc="Valid"):
            X = X.to(DEVICE).float()
            y = y.to(DEVICE).float().view(-1)
            pred = model(X).view(-1)
            total_rmse += rmse(pred, y).item()
            steps += 1
    return total_rmse / steps


def train_model(name, model, train_loader, valid_loader):
    print(f"\n===== Training {name} =====")
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    best_rmse = float("inf")
    os.makedirs("results", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        tr_rmse = train_one_epoch(model, train_loader, optimizer)
        va_rmse = eval_epoch(model, valid_loader)

        print(f"Train RMSE={tr_rmse:.4f} | Valid RMSE={va_rmse:.4f}")

        # save best
        if va_rmse < best_rmse:
            best_rmse = va_rmse
            torch.save(model.state_dict(), f"results/{name}_best.pt")
            print(f"Saved best {name} (RMSE={best_rmse:.4f})")

    return best_rmse


# ============================================================
# 5. MODELS
# ============================================================

# ---------------- MODEL 1: EEGNet baseline (Braindecode EEGNetv4) ----------------
cnn_model = EEGNetv4(
    n_chans=n_chans,
    n_outputs=1,
    n_times=n_times
)

# ---------------- MODEL 2: EEGConformer baseline ----------------
conformer_model = EEGConformer(
    n_chans=n_chans,
    n_times=n_times,
    n_outputs=1,
    sfreq=100  # set to your true sampling rate
)

# ---------------- MODEL 3: DANN (EEGNet frontend + GRL + domain head) ----------------
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
    def __init__(self, n_chans, n_times, n_domains=10, lambd=0.5, feature_dim=10):
        super().__init__()
        # Use EEGNetv4 as the feature extractor
        self.feature = EEGNetv4(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=feature_dim    # temp, will be replaced
        )

        # Replace EEGNetv4 classifier with nothing; keep only features
        # EEGNetv4 has attribute self.classifier, so:
        self.feature.classifier = nn.Identity()

        # Regressor
        self.reg_head = nn.Sequential(
            nn.Linear(feature_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

        # Domain head
        self.grl = GRL(lambd)
        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim, 10),
            nn.ReLU(),
            nn.Linear(10, n_domains)
        )

    def forward(self, x, return_domain=False):
        feat = self.feature(x)      # (B, feature_dim)
        y_hat = self.reg_head(feat).squeeze(-1)

        if not return_domain:
            return y_hat

        dom = self.domain_head(self.grl(feat))
        return y_hat, dom


# Get number of unique subjects for DANN domains
unique_subjects = meta_information["subject"].unique()
DANN_DOMAINS = len(unique_subjects)
print(f"\nNumber of unique subjects (for DANN): {DANN_DOMAINS}")

# Determine feature dimension for DANN by testing EEGNetv4
print("Determining feature dimension for DANN...")
temp_model = EEGNetv4(n_chans=n_chans, n_times=n_times, n_outputs=1)
with torch.no_grad():
    dummy_input = torch.randn(1, n_chans, n_times)
    try:
        # Check if EEGNetv4 has a classifier attribute
        if hasattr(temp_model, 'classifier'):
            # Temporarily replace classifier to get feature dim
            original_classifier = temp_model.classifier
            temp_model.classifier = nn.Identity()
            feat_output = temp_model(dummy_input)
            feature_dim = feat_output.shape[1]
            temp_model.classifier = original_classifier
            print(f"Detected feature dimension: {feature_dim}")
        else:
            # Fallback: use a reasonable default
            feature_dim = 10  # As in the provided code
            print(f"Could not determine feature dim automatically, using default: {feature_dim}")
    except Exception as e:
        print(f"Could not determine feature dim automatically: {e}")
        feature_dim = 10  # As in the provided code
        print(f"Using default feature dimension: {feature_dim}")

dann_model = DANN(n_chans=n_chans, n_times=n_times, n_domains=DANN_DOMAINS, lambd=0.5, feature_dim=feature_dim)

# ============================================================
# 6. TRAIN ALL THREE MODELS
# ============================================================

cnn_rmse = train_model("EEGNet", cnn_model, train_loader, valid_loader)
conformer_rmse = train_model("EEGConformer", conformer_model, train_loader, valid_loader)
# For DANN, use regression-only first (domain labels require your subject IDs).
dann_rmse = train_model("DANN", dann_model, train_loader, valid_loader)

# ============================================================
# 7. PRINT FINAL RESULTS
# ============================================================
print("\n================== FINAL RESULTS ==================")
print("EEGNet RMSE:       ", cnn_rmse)
print("EEGConformer RMSE: ", conformer_rmse)
print("DANN RMSE:         ", dann_rmse)
print("===================================================")

