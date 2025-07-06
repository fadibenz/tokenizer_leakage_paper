import argparse
import wandb
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizer_leakage.src.utils import set_seed_everything, load_config, get_lr_scheduler
from tokenizer_leakage.src.data_utils import  create_loader
from tokenizer_leakage.src.model import create_model
from tokenizer_leakage.src.training import train_model
from tokenizer_leakage.src.evaluation import evaluate_perplexity
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.amp import syncfree
import time

import torch_xla.runtime as xr



def _mp_fn(index, args):
    """Runs one full training and evaluation instance."""
    config = load_config(args.config)

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    set_seed_everything(args.seed + index)
    device = xm.xla_device()

    if xm.is_master_ordinal():
        run_name = f"{config['dataset_name']}-{config['model_size']}-{args.tokenizer_type}-seed{args.seed}"
        # Initialize wandb

        wandb.init(
            project=config['project_name'],
            name=run_name,
            config={**config, "tokenizer_type": args.tokenizer_type, "seed": args.seed}
        )
    else:
        run_name = ""
    try:
        # Create data_loaders
        train_path = config[f'{args.tokenizer_type}_train_path'].format(data_dir=config[f'{args.tokenizer_type}_data_dir'])
        valid_path = config[f'{args.tokenizer_type}_valid_path'].format(data_dir=config[f'{args.tokenizer_type}_data_dir'])
        test_path = config[f'{args.tokenizer_type}_test_path'].format(data_dir=config[f'{args.tokenizer_type}_data_dir'])
        train_loader = create_loader(train_path, config["context_length"], config["batch_size"], shuffle=True, stride=1)

        val_loader = create_loader(valid_path, config["context_length"], config["eval_batch_size"], stride=config["context_length"])
        # used common stride context_length - 128
        test_loader = create_loader(test_path, config["context_length"] , config['eval_batch_size'], stride=config["context_length"] - 128)
    except Exception as e:
        print(f"there was an error loading data:{e}")

    # Create model and optimizer
    model = create_model(config).to(device=device)


    optimizer = syncfree.AdamW(model.parameters(), lr=config['max_lr'], betas=(config['beta_1'], config['beta_2']),
                                  weight_decay=config['weight_decay'])
    scheduler = get_lr_scheduler(optimizer, config['warmup_steps'], config['annealing_steps'], config['max_lr'],
                                 config['min_lr'])

    # Train model

    if xm.is_master_ordinal():
        print(f"\n --- Training {run_name} ---")

    final_model = train_model(model, optimizer, scheduler, train_loader, val_loader, config, device, run_name)

    # Final evaluation, granular
    if xm.is_master_ordinal():
        print("--- Running Final Evaluation ---")

    start_time = time.time()
    # use an overlapping window for final evaluation, common stride context_length - 128
    val_loader = create_loader(valid_path, config["context_length"], config["eval_batch_size"], stride=config["context_length"] - 128)
    val_loss, val_ppl = evaluate_perplexity(final_model, val_loader, device)
    duration = time.time() - start_time

    if xm.is_master_ordinal():
        print("--- Finished Final Evaluation ---")
        print(f"Final validation took: {duration:.2f}s")

    if xm.is_master_ordinal():
        print("--- Running Evaluation On Final Test Dataset ---")

    test_loss, test_ppl = evaluate_perplexity(final_model, test_loader, device)

    if xm.is_master_ordinal():
        print(f"Final Results for {run_name}: Val PPL: {val_ppl:.4f}, Test PPL: {test_ppl:.4f}")
        wandb.log({"final/val_perplexity": val_ppl, "final/test_perplexity": test_ppl})

    xm.rendezvous("experiment_end")
    if xm.is_master_ordinal():
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Run tokenization leakage experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment config file.")
    parser.add_argument("--seed", type=int, default=2024, help="Seed to run.")
    parser.add_argument("--tokenizer_type", type=str, choices=['clean', 'leaky'], required=True, help="Tokenizer type to use.")

    args = parser.parse_args()
    try:
        xmp.spawn(_mp_fn, args=(args,), start_method='fork')
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise e

if __name__ == "__main__":
    main()