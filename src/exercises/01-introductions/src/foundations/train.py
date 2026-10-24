"""One training loop for every classifier here, and the device it runs on.

Every experiment trains full-batch with Adam. The datasets are a few hundred points, so a
mini-batch would only add a second source of randomness to results that are meant to differ in one
thing at a time.
"""

import torch
from torch import nn
from torch.nn import functional


def pick_device(requested: str | None = None) -> torch.device:
    """`requested` if given, else CUDA, then Apple's MPS, then CPU.

    These networks are tiny, so a CPU is often as fast as an accelerator: kernel launches dominate.
    The choice changes the floating-point rounding, never the claims.
    """
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_classifier(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
    device: torch.device | str = "cpu",
) -> list[float]:
    """Train a one-logit binary classifier with Adam on binary cross-entropy, full batch.

    Args:
        model: Maps `[n, 2]` points to `[n, 1]` logits. Moved to `device` in place.
        x: Training points.
        y: Float labels, 0 or 1.
        steps: Optimiser steps.
        lr: Adam learning rate.
        device: Where to train.

    Returns:
        The training loss after every step.
    """
    model.to(device)
    x, y = x.to(device), y.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        optimiser.zero_grad()
        loss = functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
        loss.backward()
        optimiser.step()
        losses.append(loss.detach())
    # One transfer at the end: `.item()` every step would stall an accelerator on each one.
    return torch.stack(losses).cpu().tolist() if losses else []


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of points whose logit has the sign of their label."""
    device = next(model.parameters()).device
    predicted = (model(x.to(device)).squeeze(-1) > 0).float().cpu()
    return (predicted == y).float().mean().item()
