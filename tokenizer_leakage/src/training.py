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

def train_model(model, optimizer, scheduler, training_loader, training_sampler, validation_loader, config, device, run_name):
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
    steps_per_epoch = len(training_loader)
    while global_step < num_training_steps:
        if global_step % steps_per_epoch == 0:
            epoch = global_step // steps_per_epoch
            training_sampler.set_epoch(epoch)
            train_iterator = iter(training_loader)

        start = time.time()
        model.train()
        x, y = next(train_iterator)
        # Forward and backward pass
        with torch.autocast("xla", dtype=torch.bfloat16):
            logits = model(x).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_l2_norm'])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        xm.mark_step()

        global_step += 1
        duration = time.time() - start

        if xm.is_master_ordinal():
            pbar.update(1)
            loss_value = loss.item()
            pbar.set_postfix({"loss": f"{loss_value:.4f}", "duration": f"{duration:.3f}s"})

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

            xm.rendezvous("before_final_save")
            if xm.is_master_ordinal():
                pbar.write(f"Step {global_step}: Validation Perplexity: {val_perplexity:.4f}")
                wandb.log({"eval/loss": val_loss, "eval/perplexity": val_perplexity, "global_step": global_step})

    xm.rendezvous("training_end")
    if xm.is_master_ordinal():
        pbar.close()
        final_checkpoint_path = output_dir / "final_model.pt"
        xm.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'config': config
        }, final_checkpoint_path)
        print("\n--- Training Complete ---")
    return model