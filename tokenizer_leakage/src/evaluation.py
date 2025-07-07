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
    try:
        for batch_idx,  (x, y )in enumerate(data_loader):
            with autocast(device):
                logits = model(x).logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                del logits
                total_val_loss += loss.detach().to(device)
                del loss

            del x, y
            num_val_batches += 1

            if num_val_batches % 10 == 0:
                if parent_pbar is not None and xm.is_master_ordinal():
                        progress_pct = (batch_idx + 1) / total_batches * 100
                        current_loss = total_val_loss / num_val_batches
                        parent_pbar.write(f"  Eval progress: {progress_pct:.0f}% | Loss: {current_loss:.4f}")

            if num_val_batches % 100 == 0:
                xm.mark_step()

    except Exception as e:
        print(f"evaluation failed for {e}")

    xm.mark_step()

    total_loss_tensor = total_val_loss
    num_batches_tensor = torch.tensor([num_val_batches], dtype=torch.float32, device=device)

    try:
        xm.all_reduce("sum", total_loss_tensor)
        xm.all_reduce('sum', num_batches_tensor)

        avg_loss = total_loss_tensor.detach().cpu().item() / num_batches_tensor.detach().cpu().item()
        perplexity = np.exp(avg_loss)
    except Exception as e:
        print(f"Reducing validation scores failed for {e}")


    total_time = time.time() - start_time
    if parent_pbar is not None and xm.is_master_ordinal():
        parent_pbar.write(f"  Eval complete: Loss={avg_loss:.4f}, PPL={perplexity:.2f} ({total_time:.1f}s)")

    return avg_loss, perplexity