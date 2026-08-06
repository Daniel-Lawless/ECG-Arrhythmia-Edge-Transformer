from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_arrhythmia.data.label_mapping import LABEL_TO_INDEX

# PyTorch’s nn.CrossEntropyLoss() expects the
# target labels to be integer class indices,
# as a LongTensor, which is what LABEL_TO_INDEX provides.


class ECGDataset(Dataset):
    # Loads data.
    def __init__(self, split_path: Path):
        # Load the data from which ever split is chosen
        self.X = np.load(split_path / "X.npy")
        self.y = np.load(split_path / "y.npy")
        self.rr_features = np.load(split_path / "rr_features.npy")

        if (
            self.X.shape[0] != self.y.shape[0]
            or self.y.shape[0] != self.rr_features.shape[0]
        ):
            raise ValueError(
                f"Shape mismatch: X={self.X.shape[0]}, y={self.y.shape[0]}, "
                f"rr_features={self.rr_features.shape[0]}"
            )

        # Convert labels to its corresponding integer value
        self.y_indices = np.array([LABEL_TO_INDEX[label] for label in self.y])

    # Tells Pytorch how many samples exist.
    def __len__(self):
        return len(self.y_indices)

    # tell PyTorch how to fetch one sample
    # This is meant to return one sample only without the batch dimension
    # DataLoader adds the batch dimension later.
    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_tensor shape would be (240,), rr_tensor would be (2,),
        # y_tensor would be a scalar tensor: shape ()
        x_tensor = torch.tensor(self.X[index], dtype=torch.float32)
        rr_tensor = torch.tensor(self.rr_features[index], dtype=torch.float32)
        y_tensor = torch.tensor(self.y_indices[index], dtype=torch.long)

        # conv1d expects x_tensor shape (channels, sequence length).
        # So we have to add another dimension at the start.
        # Shape is now (1, 240)
        x_tensor = x_tensor.unsqueeze(0)

        return x_tensor, rr_tensor, y_tensor


# Similar to PyTorch models, this is the standard pattern for creating PyTorch datasets.
# __init__(): load and store the data.
# __len__(): tell PyTorch how many samples are in the dataset.
# __getitem__(): return one ECG window, its RR features, and its label.
