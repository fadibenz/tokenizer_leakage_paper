import time

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import gc
from torch_xla.amp import autocast


@torch.no_grad()
def evaluate_perplexity(
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        parent_pbar: tqdm = None
) -> (float, float):
    """Deterministically calculates the perplexity of a model on a given dataset."""
    model.eval()
    total_val_loss = 0
    num_val_batches = 0

    val_pbar = tqdm(
        total=len(data_loader),
        desc="Evaluating Perplexity",
        disable=not xm.is_master_ordinal(),
        ncols=120,
        leave=False,
        parent=parent_pbar
    )

    for x, y in data_loader:
        start = time.time()
        with autocast(enablre=True):
            logits = model(x).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            del logits
            total_val_loss += loss.item()
            del loss

        num_val_batches += 1

        xm.mark_step()
        duration = time.time() - start

        if xm.is_master_ordinal():
            val_pbar.update(1)
            val_pbar.set_postfix({"duration": duration})


        if num_val_batches % 10 == 0:
            gc.collect()


    total_loss_tensor = torch.tensor([total_val_loss], dtype=torch.float32, device=device)
    num_batches_tensor = torch.tensor([num_val_batches], dtype=torch.float32).to(device)

    xm.all_reduce("sum", total_loss_tensor)
    xm.all_reduce('sum', num_batches_tensor)

    avg_loss = total_loss_tensor.item() / num_batches_tensor.item() if num_batches_tensor.item() > 0 else 0.0
    perplexity = np.exp(avg_loss)

    val_pbar.close()
    return avg_loss, perplexity