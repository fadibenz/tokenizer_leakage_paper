import argparse
import pandas as pd
import torch
import wandb
import os
from pathlib import Path
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizer_leakage.src.utils import set_seed_everything, load_config, get_lr_scheduler
from tokenizer_leakage.src.data_utils import load_data
from tokenizer_leakage.src.model import create_model
from tokenizer_leakage.src.training import train_model
from tokenizer_leakage.src.evaluation import evaluate_perplexity

def run_single_setting(config, tokenizer_type, seed, device):
    """Runs one full training and evaluation instance."""
    set_seed_everything(seed)

    run_name = f"{config['dataset_name']}-{config['model_size']}-{tokenizer_type}-seed{seed}"

    # Initialize wandb
    wandb.init(
        project=config['project_name'],
        name=run_name,
        config={**config, "tokenizer_type": tokenizer_type, "seed": seed}
    )

    # Load data
    train_path = config[f'{tokenizer_type}_train_path'].format(data_dir=config['data_dir'])
    valid_path = config[f'{tokenizer_type}_valid_path'].format(data_dir=config['data_dir'])
    training_data, validation_data = load_data(train_path, valid_path)

    # Create model and optimizer
    model = create_model(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['max_lr'], betas=(config['beta_1'], config['beta_2']),
                                  weight_decay=config['weight_decay'])

    scheduler = get_lr_scheduler(optimizer, config['warmup_steps'], config['annealing_steps'], config['max_lr'],
                                 config['min_lr'])

    # Train model
    print(f"\n--- Training {run_name} ---")
    final_model = train_model(model, optimizer, scheduler, training_data, validation_data, config, device, run_name)

    # Final evaluation
    val_loss, val_ppl = evaluate_perplexity(final_model, validation_data, config['eval_batch_size'], config['context_length'],
                                  device)

    test_data = np.load(config['test_path'].format(data_dir=config['data_dir']), mmap_mode="r")
    test_loss, test_ppl = evaluate_perplexity(final_model, test_data, config['eval_batch_size'], config['context_length'], device)

    print(f"Final Results for {run_name}: Val PPL: {val_ppl:.4f}, Test PPL: {test_ppl:.4f}")
    wandb.log({"final/val_perplexity": val_ppl, "final/test_perplexity": test_ppl})
    wandb.finish()

    return {"val_ppl": val_ppl, "test_ppl": test_ppl}


def main():
    parser = argparse.ArgumentParser(description="Run tokenization leakage experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment config file.")
    parser.add_argument("--seeds", type=int, nargs='+', default=[2023, 2024, 2025], help="List of seeds to run.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []
    for seed in args.seeds:
        print(f"\n{'=' * 50}\nRunning experiment for seed {seed}\n{'=' * 50}")

        clean_results = run_single_setting(config, "clean", seed, device)
        leaky_results = run_single_setting(config, "leaky", seed, device)

        results.append({
            "seed": seed,
            "clean_val_ppl": clean_results["val_ppl"],
            "clean_test_ppl": clean_results["test_ppl"],
            "leaky_val_ppl": leaky_results["val_ppl"],
            "leaky_test_ppl": leaky_results["test_ppl"],
            "val_advantage": clean_results["val_ppl"] - leaky_results["val_ppl"],
            "test_disadvantage": leaky_results["test_ppl"] - clean_results["test_ppl"]
        })

    # Save and print summary
    df = pd.DataFrame(results)
    output_dir = Path("results") / config['dataset_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{config['model_size']}_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\n--- Experiment Summary ---")
    print(df.describe())
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()