"""Optimizer utilities.

By: Florian Wiesner
Date: 2025-09-11
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP


def get_optimizer(model: nn.Module | DDP, config: dict) -> torch.optim.Optimizer:
    """Create an optimizer.

    Parameters
    ----------
    model : nn.Module
        The model to optimize
    config : dict
        Configuration dictionary for the optimizer

    Returns
    -------
    torch.optim.Optimizer
        Optimizer
    """
    lr = config["learning_rate"]
    name = config["name"]

    if name == "AdamW":
        weight_decay = config["weight_decay"]
        betas = config["betas"]

        if isinstance(model, DDP):
            optimizer = ZeroRedundancyOptimizer(
                model.parameters(),
                optimizer_class=torch.optim.AdamW,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )

    elif name == "SGD":
        momentum = config["momentum"]
        weight_decay = config["weight_decay"]

        if isinstance(model, DDP):
            optimizer = ZeroRedundancyOptimizer(
                model.parameters(),
                optimizer_class=torch.optim.SGD,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
    else:
        raise ValueError(f"Optimizer {name} not supported")

    return optimizer
