import torch
import wandb
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from tqdm.auto import tqdm # Import tqdm

from tokenizer_leakage.src.data_utils import load_batch
from tokenizer_leakage.src.evaluation import evaluate_perplexity

def train_model(model, optimizer, scheduler, training_data, validation_data, config, device, run_name):
    """Main training loop."""
    output_dir = Path(config['results_dir']) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    global_step = 0

    num_training_steps = config['num_training_steps']

    pbar = tqdm(total=num_training_steps, desc="Training Progress")

    while global_step < num_training_steps:
        model.train()

        # Prepare batch
        x, y = load_batch(training_data, config['batch_size'], config['context_length'], device)

        # Forward and backward pass
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = model(x).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_l2_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        global_step += 1
        pbar.update(1)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})

        if global_step % config['logging_freq'] == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/perplexity": np.exp(loss.item()),
                "train/lr": scheduler.get_last_lr()[0],
                "global_step": global_step,
            })

        if global_step % config['validation_freq'] == 0:
            pbar.write(f"\nStep {global_step}: Running validation...")
            val_loss, val_perplexity = evaluate_perplexity(model, validation_data, config['eval_batch_size'],
                                                 config['context_length'], device)
            pbar.write(f"Step {global_step}: Validation Perplexity: {val_perplexity:.4f}")
            wandb.log({"eval/loss": val_loss, "eval/perplexity": val_perplexity, "global_step": global_step})

        if global_step % config['checkpoint_freq'] == 0:
            # Save checkpoint
            checkpoint_path = output_dir / f"checkpoint_{global_step}.pt"
            torch.save(model.state_dict(), checkpoint_path)

    pbar.close()
    return model