from braindecode.models import EEGNeX
import torch
import os

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        """Load model for Challenge 1: Response Time Prediction"""
        model_challenge1 = EEGNeX(
            n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)
        ).to(self.device)
        
        # Try multiple possible paths for the weights
        possible_paths = [
            "weights_challenge_1.pt",
            "./weights_challenge_1.pt",
            os.path.join(os.getcwd(), "weights_challenge_1.pt"),
            os.path.join(os.path.dirname(__file__), "weights_challenge_1.pt")
        ]
        
        weights_loaded = False
        for path in possible_paths:
            try:
                print(f"🔍 Trying to load weights from: {path}")
                model_challenge1.load_state_dict(torch.load(path, map_location=self.device))
                print(f"✅ Successfully loaded weights from: {path}")
                weights_loaded = True
                break
            except FileNotFoundError:
                print(f"❌ Not found: {path}")
                continue
        
        if not weights_loaded:
            raise FileNotFoundError("Could not find weights_challenge_1.pt in any expected location")
        
        model_challenge1.eval()  # Set to evaluation mode
        return model_challenge1

    def get_model_challenge_2(self):
        """Load model for Challenge 2: Externalizing Score Prediction"""
        model_challenge2 = EEGNeX(
            n_chans=129, n_outputs=1, n_times=int(2 * self.sfreq)
        ).to(self.device)
        
        # Try multiple possible paths for the weights
        possible_paths = [
            "weights_challenge_2.pt",
            "./weights_challenge_2.pt",
            os.path.join(os.getcwd(), "weights_challenge_2.pt"),
            os.path.join(os.path.dirname(__file__), "weights_challenge_2.pt")
        ]
        
        weights_loaded = False
        for path in possible_paths:
            try:
                print(f"🔍 Trying to load weights from: {path}")
                model_challenge2.load_state_dict(torch.load(path, map_location=self.device))
                print(f"✅ Successfully loaded weights from: {path}")
                weights_loaded = True
                break
            except FileNotFoundError:
                print(f"❌ Not found: {path}")
                continue
        
        if not weights_loaded:
            raise FileNotFoundError("Could not find weights_challenge_2.pt in any expected location")
        
        model_challenge2.eval()  # Set to evaluation mode
        return model_challenge2

