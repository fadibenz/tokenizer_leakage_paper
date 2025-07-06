import time

import torch
import numpy as np
from torch_xla.amp import autocast
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import gc


@torch.no_grad()
def evaluate_perplexity(
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        parent_pbar
) -> (float, float):
    """Deterministically calculates the perplexity of a model on a given dataset."""
    model.eval()
    total_val_loss = 0
    num_val_batches = 0

    if parent_pbar is not None and xm.is_master_ordinal():
        parent_pbar.write("Starting evaluation...")

    start_time = time.time()
    total_batches = len(data_loader)
    for batch_idx,  (x, y )in enumerate(data_loader):
        with autocast(device):
            logits = model(x).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            del logits
            total_val_loss += loss.item()
            del loss

        num_val_batches += 1

        xm.mark_step()

        if parent_pbar is not None and xm.is_master_ordinal():
            if batch_idx % max(1, total_batches // 10) == 0:
                progress_pct = (batch_idx + 1) / total_batches * 100
                current_loss = total_val_loss / num_val_batches
                parent_pbar.write(f"  Eval progress: {progress_pct:.0f}% | Loss: {current_loss:.4f}")


        if num_val_batches % 10 == 0:
            gc.collect()


    total_loss_tensor = torch.tensor([total_val_loss], dtype=torch.float32, device=device)
    num_batches_tensor = torch.tensor([num_val_batches], dtype=torch.float32).to(device)

    xm.all_reduce("sum", total_loss_tensor)
    xm.all_reduce('sum', num_batches_tensor)

    avg_loss = total_loss_tensor.item() / num_batches_tensor.item() if num_batches_tensor.item() > 0 else 0.0
    perplexity = np.exp(avg_loss)

    total_time = time.time() - start_time

    if parent_pbar is not None and xm.is_master_ordinal():
        parent_pbar.write(f"  Eval complete: Loss={avg_loss:.4f}, PPL={perplexity:.2f} ({total_time:.1f}s)")
    return avg_loss, perplexity