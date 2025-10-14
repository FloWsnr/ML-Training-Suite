from pathlib import Path
from typing import Optional

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.amp.grad_scaler import GradScaler
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def sanitize_model_dict(model: torch.nn.Module) -> dict:
    """Sanitize a model's state dict for saving.

    This function removes any prefixes added by distributed training or compiling
    from the state dict keys.

    Parameters
    ----------
    model : torch.nn.Module
        The model to sanitize.

    Returns
    -------
    dict
        The sanitized state dict.
    """
    state_dict = model.state_dict()
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    consume_prefix_in_state_dict_if_present(state_dict, "_orig_mod.")
    return state_dict


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    """Load a checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the checkpoint
    device : torch.device
        Device to load the checkpoint to

    Returns
    -------
    dict
        Checkpoint
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    return checkpoint


def save_checkpoint(
    global_rank: int,
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    samples_trained: int,
    batches_trained: int,
    epoch: int,
    grad_scaler: GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
) -> None:
    """Save a checkpoint."""
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        # sync the optimizer state dict across ranks, must occur on all ranks
        optimizer.consolidate_state_dict()

    # save only on rank 0
    if global_rank == 0:
        checkpoint = {
            "samples_trained": samples_trained,
            "batches_trained": batches_trained,
            "epoch": epoch,
            "model_state_dict": sanitize_model_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "grad_scaler_state_dict": grad_scaler.state_dict(),
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(checkpoint, checkpoint_path)
