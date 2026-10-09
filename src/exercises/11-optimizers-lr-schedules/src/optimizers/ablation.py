"""AdamW with bias correction switchable off — the one knob PyTorch's optimiser does not expose.

`torch.optim.AdamW` always divides by `1 − βᵗ`. To measure what that division does on a real model,
this optimiser implements the same update with a `bias_correction` flag. With the flag on it must
match `torch.optim.AdamW` to floating-point rounding; the tests hold it to that, so the ablation
changes exactly one thing.

Update, per parameter, matching PyTorch's AdamW:

    w ← w · (1 − η·λ)                       decoupled weight decay
    m ← β₁·m + (1 − β₁)·g
    v ← β₂·v + (1 − β₂)·g²
    w ← w − η · m̂ / (√v̂ + ε)              with m̂, v̂ bias-corrected, or m, v when the flag is off
"""

import math

import torch


class SwitchableAdamW(torch.optim.Optimizer):
    """AdamW whose bias correction can be disabled.

    Args:
        params: Parameters or parameter groups.
        lr: η.
        betas: (β₁, β₂).
        eps: ε.
        weight_decay: λ, decoupled.
        bias_correction: Divide the moments by `1 − βᵗ`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        bias_correction: bool = True,
    ) -> None:
        """Store the hyperparameters as PyTorch's optimisers do: as group defaults."""
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "bias_correction": bias_correction,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102 - torch.optim.Optimizer's contract
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                p.mul_(1.0 - group["lr"] * group["weight_decay"])
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(beta1).add_(p.grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)
                if group["bias_correction"]:
                    c1, c2 = 1.0 - beta1**t, 1.0 - beta2**t
                else:
                    c1 = c2 = 1.0
                denom = (v.sqrt() / math.sqrt(c2)).add_(group["eps"])
                p.addcdiv_(m, denom, value=-group["lr"] / c1)
        return loss
