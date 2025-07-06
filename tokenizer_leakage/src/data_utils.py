import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from pathlib import Path

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def load_data(train_path: str, valid_path: str) -> tuple[npt.NDArray, npt.NDArray]:
    """Loads memory-mapped training and validation data."""
    training_data = np.load(train_path, mmap_mode="r")
    validation_data = np.load(valid_path, mmap_mode="r")
    print(f"Loaded training data with {len(training_data):,} tokens.")
    print(f"Loaded validation data with {len(validation_data):,} tokens.")
    return training_data, validation_data

def load_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples a random batch of data."""

    n = len(dataset)
    indices = np.random.randint(0, n - context_length, size=batch_size)
    x = torch.from_numpy(dataset[indices[:, None] + np.arange(context_length)].astype(np.int64))
    y = torch.from_numpy(dataset[indices[:, None] + np.arange(context_length) + 1].astype(np.int64))

    return x.to(device), y.to(device)


# Switched to PyTorch Dataset for distributed sampling instead of manually sharding.

class MemmapDataset(Dataset):
    def __init__(self,
                 file_path: str | Path,
                 context_length: int,
                 stride: int):
        super().__init__()
        self.context_length = context_length
        self.data = np.load(file_path, mmap_mode="r")
        self.stride = stride if stride is not None else context_length
        self.max_idx = (len(self.data) - context_length - 1) // self.stride

    def __len__(self) -> int:
        return self.max_idx

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:

        if idx >= self.max_idx:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.max_idx}")

        start_idx = idx * self.stride
        chunk = self.data[start_idx: start_idx + self.context_length + 1]

        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def create_loader(data_path, context_length, batch_size, stride=None,  shuffle=False):
    dataset = MemmapDataset(data_path, context_length, stride)
    sampler = DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=shuffle
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False

    )

    return pl.MpDeviceLoader(loader, xm.xla_device())