import os
from collections import deque

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class EMGDataset(Dataset):
    """
    A Dataset class that reads (N, C, T) shaped physiological signals from an HDF5 file,
    optionally including labels. The following features are supported:

        - squeeze (bool): Whether to add a dimension at the start (e.g., (C, T) -> (1, C, T))
        - transform (callable): A function for data transformation (e.g., min-max normalization to [-1, 1])
        - zero_pad_chans (int): Zero-pad along the channel dimension
        - zero_pad_toks (int): Zero-pad along the time dimension
        - finetune (bool): If True and labels exist in the file, returns (x, y); otherwise returns x only.
    """

    def __init__(
        self,
        file_path: str,
        transform=None,
        squeeze: bool = False,
        finetune: bool = True,
        zero_pad_chans: int = None,
        zero_pad_toks: int = None,
        regression: bool = False,
    ):
        super().__init__()
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        self.file_path = file_path
        self.transform = transform
        self.squeeze = squeeze
        self.finetune = finetune
        self.zero_pad_chans = zero_pad_chans
        self.zero_pad_toks = zero_pad_toks
        self.regression = regression

        # Read HDF5 file
        with h5py.File(file_path, "r") as h5f:
            # Data shape: (N, C, T)
            self._data = h5f["data"][:]  # => shape (N, C, T)
            # Read labels if they exist
            if "label" in h5f.keys():
                self._labels = h5f["label"][:]  # => shape (N,)
            else:
                self._labels = None

        # Compute the number of samples
        self._num_samples = self._data.shape[0]

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx: int):
        # Extract a single sample: shape (C, T)
        x_np = self._data[idx]
        x = torch.tensor(x_np, dtype=torch.float32)

        # Optionally squeeze the data => (1, C, T) if it was (C, T)
        if self.squeeze:
            x = x.unsqueeze(0)

        # Optional transform (e.g., min-max normalization to [-1, 1])
        if self.transform:
            max_val = x.amax(dim=-1, keepdim=True)
            min_val = x.amin(dim=-1, keepdim=True)
            # Avoid division by zero
            x = (x - min_val) / (max_val - min_val + 1e-10)
            x = (x - 0.5) * 2  # Rescales to [-1, 1]

        # Zero-pad along the channel dimension if requested
        if self.zero_pad_chans is not None:
            # Current shape: (C, T)
            # Transpose to (T, C), pad, then transpose back to (C, T)
            x = x.transpose(0, 1)  # => (T, C)
            x = F.pad(x, (0, self.zero_pad_chans), value=0.0)  # => (T, C + zero_pad_chans)
            x = x.transpose(0, 1)  # => (C + zero_pad_chans, T)

        # Zero-pad along the time dimension if requested
        if self.zero_pad_toks is not None:
            # Current shape: (C, T)
            # Pad on the right side of time dimension
            x = F.pad(x, (0, self.zero_pad_toks), value=0.0)

        # If we are finetuning and labels exist, return (x, y), otherwise return x
        if self.finetune:
            y = self._labels[idx]
            y = torch.tensor(y, dtype=torch.float32 if self.regression else torch.long)

        return (x, y) if self.finetune else x