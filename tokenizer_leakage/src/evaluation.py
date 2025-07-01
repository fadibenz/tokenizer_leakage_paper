import torch
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

@torch.no_grad()
def evaluate_perplexity(
        model: torch.nn.Module,
        data: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: torch.device
) -> (float, float):
    """Deterministically calculates the perplexity of a model on a given dataset.

       I'm not yet sure if this is the standard way to validate or if a sliding window
       would be a better approach.
    """

    model.eval()
    losses = []

    num_batches = (len(data) - 1) // (context_length * batch_size)

    for i in tqdm(range(num_batches), desc="Evaluating Perplexity"):
        start_idx = i * batch_size * context_length
        indices = np.arange(start_idx, start_idx + batch_size * context_length, context_length)

        inputs_np = np.array([data[j: j + context_length] for j in indices])
        targets_np = np.array([data[j + 1: j + context_length + 1] for j in indices])

        inputs = torch.from_numpy(inputs_np.astype(np.int64)).to(device)
        targets = torch.from_numpy(targets_np.astype(np.int64)).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = model(inputs).logits
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        losses.append(loss.item())

    if not losses:
        return float('inf')

    avg_loss = np.mean(losses)
    return avg_loss, np.exp(avg_loss)