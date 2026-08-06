from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_arrhythmia.data.label_mapping import LABEL_TO_INDEX


class ECGSequenceDataset(Dataset):
    """
    PyTorch dataset for transformer-style beat sequences.

    Returns:
    - X sequence with shape (sequence_length, 1, window_size)
    - RR sequence with shape (sequence_length, rr_feature_dim)
    - target label index for the final beat in the sequence
    """

    def __init__(self, split_path: Path):
        # Load our sequence data
        self.X = np.load(split_path / "X.npy")
        self.y = np.load(split_path / "y.npy")
        self.rr_features = np.load(split_path / "rr_features.npy")

        if self.X.ndim != 3:
            raise ValueError(
                "X must have shape (num_sequences, K, window_size). "
                f"Found {self.X.shape}"
            )

        if self.rr_features.ndim != 3:
            raise ValueError(
                "rr_features must have shape (num_sequences, K, rr_feature_dim). "
                f"Found {self.rr_features.shape}"
            )

        if self.X.shape[:2] != self.rr_features.shape[:2]:
            raise ValueError(
                "X and rr_features must agree on num_sequences and sequence_length. "
                f"Found X={self.X.shape}, rr_features={self.rr_features.shape}"
            )

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"Shape mismatch: X={self.X.shape[0]}, y={self.y.shape[0]}"
            )

        self.y_indices = np.array([LABEL_TO_INDEX[label] for label in self.y])

    def __len__(self) -> int:
        # Tell PyTorch how many samples are in the dataset.
        # The DataLoader uses this to work out how many batches make up one epoch.
        return len(self.y_indices)

    # Define what one sample contains and how to load it by index.
    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_tensor = torch.tensor(self.X[index], dtype=torch.float32).unsqueeze(1)
        rr_tensor = torch.tensor(self.rr_features[index], dtype=torch.float32)
        # CrossEntropyLoss expects class labels as integer/long tensors.
        y_tensor = torch.tensor(self.y_indices[index], dtype=torch.long)

        return x_tensor, rr_tensor, y_tensor
