# ##########################################################################
# # Example of submission files
# # ---------------------------
# The zip file needs to be single level depth!
# NO FOLDER
# my_submission.zip
# ├─ submission.py
# ├─ RandomForest_200_20251101_170036.pt
# └─ weights_challenge_2.pt

import torch
import numpy as np
import scipy.signal as sig
from scipy.stats import skew, kurtosis
from pathlib import Path


def resolve_path(name="model_file_name"):
    if Path(f"/app/input/res/{name}").exists():
        return f"/app/input/res/{name}"
    elif Path(f"/app/input/{name}").exists():
        return f"/app/input/{name}"
    elif Path(f"{name}").exists():
        return f"{name}"
    elif Path(__file__).parent.joinpath(f"{name}").exists():
        return str(Path(__file__).parent.joinpath(f"{name}"))
    else:
        raise FileNotFoundError(
            f"Could not find {name} in /app/input/res/ or /app/input/ or current directory"
        )

def bandpower(data, fs, fmin, fmax):
    """Compute bandpower using Welch's method"""
    f, Pxx = sig.welch(data, fs=fs, nperseg=min(256, len(data)), nfft=1024)
    band = (f >= fmin) & (f <= fmax)
    return np.trapz(Pxx[band], f[band])

def extract_features_from_window(window_np: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    Extract statistical + spectral features from EEG window.
    window_np shape: (n_chans, n_times)
    Returns: feature vector of length 1161 (129 channels * 9 features per channel)
    """
    # Basic stats
    means = window_np.mean(axis=1)
    stds = window_np.std(axis=1) + 1e-8
    skews = skew(window_np, axis=1, bias=False, nan_policy='omit')
    kurts = kurtosis(window_np, axis=1, fisher=True, bias=False, nan_policy='omit')

    # Define canonical EEG bands
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }

    # Compute bandpower for each channel
    band_feats = []
    for ch in range(window_np.shape[0]):
        ch_data = window_np[ch, :]
        ch_bandpowers = [bandpower(ch_data, fs, fmin, fmax) for fmin, fmax in bands.values()]
        band_feats.append(ch_bandpowers)
    band_feats = np.array(band_feats)  # shape: (n_chans, n_bands)

    # Combine all features
    feats = np.concatenate([
        means[:, None],
        stds[:, None],
        skews[:, None],
        kurts[:, None],
        band_feats
    ], axis=1).reshape(-1)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


class RandomForestWrapper(torch.nn.Module):
    """
    Wrapper class that mimics PyTorch model interface but uses Random Forest under the hood.
    This allows the Random Forest model to work with the existing submission infrastructure.
    """
    def __init__(self, pipeline, sfreq, device):
        super().__init__()
        self.pipeline = pipeline  # sklearn Pipeline (StandardScaler + RandomForestRegressor)
        self.sfreq = sfreq
        self.device = device
        self.eval_mode = True
        
    def forward(self, X):
        """
        Forward pass: Extract features from EEG windows and predict using Random Forest.
        
        Args:
            X: Tensor of shape (batch_size, n_chans, n_times) or (batch_size, 1, n_chans, n_times)
        
        Returns:
            Tensor of shape (batch_size, 1) with predictions
        """
        # Convert to numpy and handle different input shapes
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.array(X)
        
        # Handle different input shapes: (batch, chans, times) or (batch, 1, chans, times)
        while X_np.ndim > 3:
            X_np = X_np.squeeze(1)
        
        batch_size = X_np.shape[0]
        
        # Extract features for each window in the batch
        features_list = []
        for i in range(batch_size):
            window = X_np[i]  # Shape: (n_chans, n_times)
            # Ensure shape is (n_chans, n_times)
            if window.ndim != 2:
                window = window.reshape(129, -1)
            features = extract_features_from_window(window, fs=self.sfreq)
            features_list.append(features)
        
        # Stack features: (batch_size, n_features)
        features_array = np.array(features_list)
        
        # Predict using the pipeline (includes scaling + Random Forest)
        predictions = self.pipeline.predict(features_array)  # Shape: (batch_size,)
        
        # Reshape to (batch_size, 1) to match expected output format
        predictions = predictions.reshape(-1, 1)
        
        # Convert back to torch tensor
        predictions_tensor = torch.from_numpy(predictions).float().to(self.device)
        
        return predictions_tensor
    
    def eval(self):
        """Set model to evaluation mode (no-op for Random Forest, but maintains compatibility)"""
        self.eval_mode = True
        return self
    
    def train(self, mode=True):
        """Set model to training mode (no-op for Random Forest, but maintains compatibility)"""
        self.eval_mode = not mode
        return self
    
    def to(self, device):
        """Move model to device (stores device for tensor conversion)"""
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        return self


class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        """
        Load Random Forest model for Challenge 1.
        Tries multiple filenames with .pt extension.
        """
        # Try multiple potential filenames, ordered by preference
        model_filenames = [
            "RandomForest_200_20251101_170036.pt",
        ]
        
        pipeline = None
        loaded_path = None
        
        # Try each filename until we find one that works
        for filename in model_filenames:
            try:
                model_path = resolve_path(filename)
                if Path(model_path).exists():
                    pipeline = torch.load(model_path, map_location='cpu')
                    loaded_path = model_path
                    print(f"✅ Loaded Random Forest model from: {model_path}")
                    break
            except (FileNotFoundError, Exception):
                continue
        
        if pipeline is None:
            tried_files = '\n'.join(f"  - {f}" for f in model_filenames)
            raise FileNotFoundError(
                f"Could not load Random Forest model for Challenge 1.\n"
                f"Tried the following files:\n{tried_files}\n"
                f"Please ensure the model file exists in the submission directory."
            )
        
        model = RandomForestWrapper(pipeline, self.sfreq, self.device)
        return model

    def get_model_challenge_2(self):
        """
        Load model for Challenge 2 (keep original neural network approach).
        If you want to use Random Forest here too, adapt this method similarly.
        """
        from braindecode.models import EEGNeX
        model_challenge2 = EEGNeX(
            n_chans=129, n_outputs=1, n_times=int(2 * self.sfreq)
        ).to(self.device)
        model_challenge2.load_state_dict(torch.load(resolve_path("weights_challenge_2_10_30.pt"), map_location=self.device))
        return model_challenge2


# ##########################################################################
# # How Submission class will be used
# # ---------------------------------
# from submission import Submission

# SFREQ = 100
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sub = Submission(SFREQ, DEVICE)
# model_1 = sub.get_model_challenge_1()
# model_1.eval()

# warmup_loader_challenge_1 = DataLoader(HBN_R5_dataset1, batch_size=BATCH_SIZE)
# final_loader_challenge_1 = DataLoader(secret_dataset1, batch_size=BATCH_SIZE)

# with torch.inference_mode():
#     for batch in warmup_loader_challenge_1:  # and final_loader later
#         X, y, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X.shape is (BATCH_SIZE, 129, 200)

#         # Forward pass
#         y_pred = model_1.forward(X)
#         # save prediction for computing evaluation score
#         ...
# score1 = compute_score_challenge_1(y_true, y_preds)
# del model_1
# gc.collect()

# model_2 = sub.get_model_challenge_2()
# model_2.eval()

# warmup_loader_challenge_2 = DataLoader(HBN_R5_dataset2, batch_size=BATCH_SIZE)
# final_loader_challenge_2 = DataLoader(secret_dataset2, batch_size=BATCH_SIZE)

# with torch.inference_mode():
#     for batch in warmup_loader_challenge_2:  # and final_loader later
#         X, y, crop_inds, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X shape is (BATCH_SIZE, 129, 200)

#         # Forward pass
#         y_pred = model_2.forward(X)
#         # save prediction for computing evaluation score
#         ...
# score2 = compute_score_challenge_2(y_true, y_preds)
# overall_score = compute_leaderboard_score(score1, score2)
