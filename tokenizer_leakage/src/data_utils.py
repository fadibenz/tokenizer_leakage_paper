import numpy as np
import numpy.typing as npt
import torch

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