import torch
import numpy as np
import random
import yaml
import math
import torch.optim as optim


def set_seed_everything(seed: int = 2025):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_lr_scheduler(optimizer, warmup_steps, annealing_steps, max_lr, min_lr):
    """Cosine annealing with warmup learning rate scheduler."""

    def lr_lambda(current_step):
        ratio = min_lr / max_lr
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < annealing_steps:
            progress = float(current_step - warmup_steps) / float(annealing_steps - warmup_steps)
            return ratio + 0.5 * (1.0 + math.cos(math.pi * progress)) * (1 - ratio)
        else:
            return ratio

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

