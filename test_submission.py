from braindecode.models import EEGNeX
import torch
from pathlib import Path

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        """Load model for Challenge 1: Response Time Prediction"""
        model_challenge1 = EEGNeX(
            n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)
        ).to(self.device)
        
        # Try to load the model weights
        weights_path = Path("submission_1/model_weights_challenge_1.pt")
        if weights_path.exists():
            print(f"✅ Loading Challenge 1 weights from: {weights_path}")
            model_challenge1.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            print(f"⚠️ Weights not found at {weights_path}, using random weights")
            
        return model_challenge1

    def get_model_challenge_2(self):
        """Load model for Challenge 2: Externalizing Score Prediction"""
        model_challenge2 = EEGNeX(
            n_chans=129, n_outputs=1, n_times=int(2 * self.sfreq)
        ).to(self.device)
        
        # Try to load the model weights
        weights_path = Path("submission_1/model_weights_challenge_2.pt")
        if weights_path.exists():
            print(f"✅ Loading Challenge 2 weights from: {weights_path}")
            model_challenge2.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            print(f"⚠️ Weights not found at {weights_path}, using random weights")
            
        return model_challenge2
