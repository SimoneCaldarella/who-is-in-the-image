import torch

def load_optimizer(optimizer):
    if optimizer == "adamw":
        return torch.optim.AdamW
    else:
        raise ValueError(f"Optimizer {optimizer} not found")
