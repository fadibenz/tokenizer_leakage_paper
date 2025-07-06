import time

import torch
import wandb
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from tqdm.auto import tqdm
import torch_xla.core.xla_model as xm
from tokenizer_leakage.src.evaluation import evaluate_perplexity
import itertools
from torch_xla.amp import autocast


def train_model(model, optimizer, scheduler, training_loader, validation_loader, config, device, run_name):
    """Main training loop."""
    output_dir = Path(config['results_dir']) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    num_training_steps = config['num_training_steps']

    pbar = tqdm(total=num_training_steps,
                desc="Training Progress",
                disable=not xm.is_master_ordinal(),
                ncols=120)

    train_iterator = itertools.cycle(training_loader)

    while global_step < num_training_steps:
        start = time.time()
        model.train()
        try:
            x, y = next(train_iterator)
            # Forward and backward pass
            with autocast(device):
                logits = model(x).logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            loss_value = loss.item()
            loss.backward()

            del x, y, logits

            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_l2_norm'])
            xm.optimizer_step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            xm.mark_step()

            global_step += 1
            duration = time.time() - start

            if xm.is_master_ordinal():
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}",
                                  "duration": duration})

                if global_step % config['logging_freq'] == 0 and xm.is_master_ordinal():

                    wandb.log({
                        "train/loss": loss_value,
                        "train/perplexity": np.exp(loss_value),
                        "train/lr": scheduler.get_last_lr()[0],
                        "global_step": global_step,
                    })

                if global_step % config['checkpoint_freq'] == 0:
                    # Save checkpoint
                    checkpoint_path = output_dir / f"checkpoint_{global_step}.pt"
                    xm.save(model.state_dict(), checkpoint_path)
                    xm.mark_step()

            if global_step % config['validation_freq'] == 0 :
                if xm.is_master_ordinal():
                    pbar.write(f"\nStep {global_step}: Running validation...")

                val_loss, val_perplexity = evaluate_perplexity(model, validation_loader, device, pbar)

                if xm.is_master_ordinal():
                    pbar.write(f"Step {global_step}: Validation Perplexity: {val_perplexity:.4f}")
                    wandb.log({"eval/loss": val_loss, "eval/perplexity": val_perplexity, "global_step": global_step})

        except Exception as e:
            if xm.is_master_ordinal():
                print(f"Error at step {global_step}: {str(e)}")
            raise e

    xm.rendezvous("training_end")
    if xm.is_master_ordinal():
        pbar.close()
        print("\n--- Training Complete ---")
        final_checkpoint_path = output_dir / "final_model.pt"
        xm.save({
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'config': config
            }, final_checkpoint_path
        })

    return model