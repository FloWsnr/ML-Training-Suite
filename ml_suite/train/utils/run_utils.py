"""Utility functions for training and evaluation.

By: Florian Wiesner
Date: 2025-09-15
"""

from pathlib import Path, PurePath
from typing import Optional

import torch
import torch.distributed as dist


def human_format(num: int | float) -> str:
    """Format a number with SI prefixes (K, M, B).

    Parameters
    ----------
    num : int or float
        The number to format.

    Returns
    -------
    str
        Formatted string with SI prefix.
    """
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000:
            return f"{num:.2f}{unit}"
        num /= 1000
    return f"{num:.2f}P"


def reduce_all_losses(losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reduce the losses across all GPUs."""
    for loss_name, loss in losses.items():
        losses[loss_name] = _reduce_loss(loss)
    return losses


def _reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    """Reduce the loss across all GPUs."""
    if dist.is_initialized():
        loss_tensor = loss.clone().detach()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        return loss_tensor
    else:
        return loss


@torch.inference_mode()
def compute_metrics(
    x: torch.Tensor,
    target: torch.Tensor,
    metrics: dict[str, torch.nn.Module],
) -> dict[str, torch.Tensor]:
    """Compute the metrics.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    target : torch.Tensor
        The target tensor.
    metrics : dict[str, torch.nn.Module]
        a dictionary of metric names and metric functions

    Returns
    -------
    dict[str, torch.Tensor]
        a dictionary of metric names and metric values
    """
    metrics_values = {}
    for metric_name, metric in metrics.items():
        metric_value = metric(x, target)
        metrics_values[metric_name] = metric_value
    return metrics_values
