from braindecode.datasets.base import BaseDataset
import random, torch

class DatasetWrapper(BaseDataset):
    def __init__(self, dataset, crop_size_samples, target_name="externalizing", seed=None):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.target_name = target_name
        self.rng = random.Random(seed)

    def __len__(self): return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]
        target = float(self.dataset.description[self.target_name])
        i_window_in_trial, i_start, i_stop = crop_inds
        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        X = X[:, start_offset:start_offset + self.crop_size_samples]
        i_start = i_start + start_offset
        i_stop = i_start + self.crop_size_samples
        return X, target, (i_window_in_trial, i_start, i_stop), {
            "subject": self.dataset.description["subject"],
            "sex": self.dataset.description["sex"],
            "age": float(self.dataset.description["age"]),
            "task": self.dataset.description["task"],
            "session": self.dataset.description.get("session", "") or "",
            "run": self.dataset.description.get("run", "") or "",
              
              
        }