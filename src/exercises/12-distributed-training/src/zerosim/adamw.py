"""AdamW on a flat fp32 slice — the same arithmetic whether it holds the whole model or 1/N of it.

**Every operation is elementwise**, and that is the whole reason ZeRO-1 works. Element `i`'s update
reads only element `i` of the weight, the gradient, `m` and `v`. So a rank holding only a shard of
`m` and `v` can update exactly that shard of the weights, with no information about the rest — and
the result is the same, element for element, as updating the whole buffer on one device.

The update, with bias correction on (`m` and `v` start at zero, so for the first steps they
underestimate the true averages; dividing by `1 − βᵗ` undoes that):

    w ← w · (1 − lr · wd)                              decoupled weight decay
    m ← β₁ · m + (1 − β₁) · g                           first moment
    v ← β₂ · v + (1 − β₂) · g²                          second moment
    w ← w − (lr / (1 − β₁ᵗ)) · m / (√v / √(1 − β₂ᵗ) + ε)

**Written with single-rounding operations only — and the reason is a finding, not a style.** The
first version used torch's own fused kernels (`lerp_`, `addcmul_`, `addcdiv_`), exactly as
`torch.optim.AdamW` does. ZeRO-1, -2 and -3 then agreed with each other bit for bit and **disagreed
with stage 0 by up to 1e-6** — in an update that is elementwise and therefore cannot depend on
how the buffer is sliced. It did anyway. Probed directly (arm64, torch 2.13): cut at multiples of
64 elements, the fused kernels matched the whole buffer bit for bit; cut at other offsets,
`addcmul_` did not — consistent with a vectorised loop and its scalar tail rounding differently,
so a shard boundary changes which elements take which path. That probe is platform-specific and
is not a test here. Multiply, add, subtract, divide and square root are each correctly rounded on
every path, so built from those alone the update is identical however the buffer is cut. The cost
is that this is no longer bit-identical to `torch.optim.AdamW`; `tests/test_zerosim_adamw.py`
compares the two to a stated tolerance and checks the slicing invariance directly.
"""

from dataclasses import dataclass

import torch

ADAMW_FLOPS_PER_ELEMENT = 15
"""Arithmetic operations per element in `adamw_step`, counted line by line below.

decay 1 · first moment 3 (scale m, scale g, add) · second moment 4 (scale v, square g, scale it,
add) · `sqrt` 1 · divide by the bias-correction root 1 · add `eps` 1 · `m / denom` 1 · scale by the
step size 1 · subtract 1. Fifteen. Used only to express the optimiser's work in FLOPs; the
measurement that matters is **elements updated per rank**, which is exact.
"""


@dataclass(frozen=True)
class Hyper:
    """AdamW's hyper-parameters."""

    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float


def adamw_step(
    weight: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    step: int,
    hyper: Hyper,
) -> None:
    """Update `weight`, `m` and `v` in place for optimiser step `step` (1-based).

    Args:
        weight: fp32 weights — the master copy in bf16-mixed, the weights themselves in fp32.
        grad: The averaged gradient for the same elements. Upcast to fp32 here if it is bf16.
        m: First moment, fp32, same shape.
        v: Second moment, fp32, same shape.
        step: 1 on the first update. Bias correction depends on it.
        hyper: Learning rate, betas, eps, weight decay.
    """
    if step < 1:
        raise ValueError("AdamW steps are counted from 1; bias correction divides by 1 − β^step")
    grad = grad.to(torch.float32)
    weight.mul_(1 - hyper.lr * hyper.weight_decay)
    m.mul_(hyper.beta1).add_(grad * (1 - hyper.beta1))
    v.mul_(hyper.beta2).add_((grad * grad) * (1 - hyper.beta2))
    bias_correction1 = 1 - hyper.beta1**step
    bias_correction2 = 1 - hyper.beta2**step
    step_size = hyper.lr / bias_correction1
    denom = (v.sqrt() / bias_correction2**0.5).add_(hyper.eps)
    weight.sub_((m / denom).mul_(step_size))
