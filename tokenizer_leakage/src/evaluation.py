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
        parent_pbar = None
) -> (float, float):
    """Deterministically calculates the perplexity of a model on a given dataset."""
    model.eval()
    total_val_loss = 0
    num_val_batches = 0
    total_batches = len(data_loader)

    if parent_pbar is not None and xm.is_master_ordinal():
        parent_pbar.write("Starting evaluation...")

    start_time = time.time()
    for batch_idx,  (x, y )in enumerate(data_loader):

        with autocast(device):
            logits = model(x).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            del logits
            total_val_loss += loss.detach()
            del loss

        del x, y
        num_val_batches += 1

        xm.mark_step()

        if num_val_batches % 10 == 0:
            if parent_pbar is not None and xm.is_master_ordinal():
                    progress_pct = (batch_idx + 1) / total_batches * 100
                    parent_pbar.write(f"  Eval progress: {progress_pct:.0f}%")

    xm.rendezvous("eval_sync")

    total_loss_reduced = xm.mesh_reduce('loss_sum', total_val_loss, np.sum)
    total_batches_reduced = xm.mesh_reduce('batch_sum', num_val_batches, np.sum)

    avg_loss = total_loss_reduced / total_batches_reduced
    perplexity = np.exp(avg_loss)



    total_time = time.time() - start_time
    if parent_pbar is not None and xm.is_master_ordinal():
        parent_pbar.write(f"  Eval complete: Loss={avg_loss:.4f}, PPL={perplexity:.2f} ({total_time:.1f}s)")

    return avg_loss, perplexity