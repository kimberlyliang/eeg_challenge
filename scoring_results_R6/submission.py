from braindecode.models import EEGNeX
import torch

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        model_challenge1 = EEGNeX(
            n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)
        ).to(self.device)
        # load from the current directory (where scoring system extracts them)
        model_challenge1.load_state_dict(torch.load("weights_challenge_1.pt", map_location=self.device))
        model_challenge1.eval()  # Set to evaluation mode
        return model_challenge1

    def get_model_challenge_2(self):
        model_challenge2 = EEGNeX(
            n_chans=129, n_outputs=1, n_times=int(2 * self.sfreq)
        ).to(self.device)
        model_challenge2.load_state_dict(torch.load("weights_challenge_2.pt", map_location=self.device))
        model_challenge2.eval()  # Set to evaluation mode
        return model_challenge2