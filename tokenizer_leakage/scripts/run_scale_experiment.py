import argparse
import torch
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


def _mp_fn(index, args):
    """Runs one full training and evaluation instance."""
    config = load_config(args.config)


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


    # Create data_loaders
    train_path = config[f'{args.tokenizer_type}_train_path'].format(data_dir=config['data_dir'])
    valid_path = config[f'{args.tokenizer_type}_valid_path'].format(data_dir=config['data_dir'])

    train_loader = create_loader(train_path, config["context_length"], config["batch_size"], shuffle=True)
    val_loader = create_loader(valid_path, config["context_length"], config["batch_size"])

    # Create model and optimizer
    model = create_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['max_lr'], betas=(config['beta_1'], config['beta_2']),
                                  weight_decay=config['weight_decay'])
    scheduler = get_lr_scheduler(optimizer, config['warmup_steps'], config['annealing_steps'], config['max_lr'],
                                 config['min_lr'])

    # Train model
    print(f"\n [{xm.get_ordinal()}]--- Training {run_name} ---")
    final_model = train_model(model, optimizer, scheduler, train_loader, val_loader, config, device, run_name)

    if xm.is_master_ordinal():
        # Final evaluation
        print("--- Running Final Evaluation ---")
        test_loader = create_loader(config[f'{args.tokenizer_type}_test_path'].format(data_dir=config['data_dir']), config["context_length"] , config['eval_batch_size'], shuffle=False )

        val_loss, val_ppl = evaluate_perplexity(final_model, val_loader,device)
        test_loss, test_ppl = evaluate_perplexity(final_model, test_loader, device)

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
    xmp.spawn(_mp_fn, args=(args,), start_method="fork")

if __name__ == "__main__":
    main()