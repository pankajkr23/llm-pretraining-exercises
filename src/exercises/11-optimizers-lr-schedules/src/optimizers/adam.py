"""Adam, written out one line per quantity, and the same steps read back out of PyTorch.

For one weight `w` and its gradients `g_1 … g_T`, Adam keeps two running averages and turns them
into a step:

    m_t = β₁·m_{t-1} + (1 − β₁)·g_t              first moment: the direction
    v_t = β₂·v_{t-1} + (1 − β₂)·g_t²             second moment: the scale
    m̂_t = m_t / (1 − β₁ᵗ)                       bias correction: both averages start at zero,
    v̂_t = v_t / (1 − β₂ᵗ)                       so early on they are too small by exactly this
    w_t = w_{t-1} − η · m̂_t / (√v̂_t + ε)

`adam_by_hand` computes exactly that in Python floats. `torch_adam` runs `torch.optim.Adam` on a
single float64 weight, injects the same gradients, and reads `m` and `v` back out of the optimizer's
own state, so the two can be compared quantity by quantity rather than only on the final weight.

**Why bias correction matters, as a number.** With a constant gradient `g`, `m_t = g(1 − β₁ᵗ)` and
`v_t = g²(1 − β₂ᵗ)` exactly, so correction recovers `g` and `g²` exactly and the corrected step is
`η` every time. Without it the step is `η · (1 − β₁ᵗ)/√(1 − β₂ᵗ)` (`uncorrected_over_corrected`).
At the default betas that is **3.16η at step 1, and it grows before it shrinks**: β₁ = 0.9 forgets
in about 10 steps while β₂ = 0.999 takes about a thousand, so `m` recovers long before `v` does.
`steps_until_within` reports when the two finally agree to a tolerance.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AdamStep:
    """Everything Adam computes at one step, for one weight.

    Attributes:
        t: 1-based step number (the exponent in the bias correction).
        g: The gradient fed in.
        m: First moment after the update.
        v: Second moment after the update.
        m_hat: `m` after bias correction (equal to `m` when correction is off).
        v_hat: `v` after bias correction (equal to `v` when correction is off).
        update: The change applied to the weight, `w_t − w_{t-1}`.
        w: The weight after the step.
    """

    t: int
    g: float
    m: float
    v: float
    m_hat: float
    v_hat: float
    update: float
    w: float


def adam_by_hand(
    w0: float,
    grads: list[float],
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    bias_correction: bool = True,
) -> list[AdamStep]:
    """Run Adam on one weight with the given gradients, every quantity written out.

    Args:
        w0: The starting weight.
        grads: The gradient at each step, in order.
        lr: η.
        beta1: β₁.
        beta2: β₂.
        eps: ε, added to `√v̂` so a zero second moment cannot divide by zero.
        bias_correction: Divide `m` and `v` by `1 − βᵗ`. Turning it off is the ablation.

    Returns:
        One `AdamStep` per gradient.
    """
    m = v = 0.0
    w = w0
    steps: list[AdamStep] = []
    for t, g in enumerate(grads, start=1):
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        m_hat = m / (1.0 - beta1**t) if bias_correction else m
        v_hat = v / (1.0 - beta2**t) if bias_correction else v
        update = -lr * m_hat / (math.sqrt(v_hat) + eps)
        w = w + update
        steps.append(AdamStep(t, g, m, v, m_hat, v_hat, update, w))
    return steps


def torch_adam(
    w0: float,
    grads: list[float],
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> list[AdamStep]:
    """The same steps, taken by `torch.optim.Adam` on a float64 weight and read back from its state.

    PyTorch stores `m` as `exp_avg` and `v` as `exp_avg_sq`; `m̂` and `v̂` are not stored, so they are
    derived from those with the same `1 − βᵗ`. The update is measured as the change in the weight.

    Args:
        w0: The starting weight.
        grads: The gradients.
        lr: η.
        beta1: β₁.
        beta2: β₂.
        eps: ε.

    Returns:
        One `AdamStep` per gradient.
    """
    import torch

    weight = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float64))
    optimizer = torch.optim.Adam([weight], lr=lr, betas=(beta1, beta2), eps=eps)
    steps: list[AdamStep] = []
    for t, g in enumerate(grads, start=1):
        before = weight.item()
        weight.grad = torch.tensor(g, dtype=torch.float64)
        optimizer.step()
        state = optimizer.state[weight]
        m = state["exp_avg"].item()
        v = state["exp_avg_sq"].item()
        after = weight.item()
        steps.append(
            AdamStep(t, g, m, v, m / (1 - beta1**t), v / (1 - beta2**t), after - before, after)
        )
    return steps


def largest_difference(a: list[AdamStep], b: list[AdamStep]) -> dict[str, float]:
    """The largest absolute difference in each quantity across two runs of the same steps."""
    if len(a) != len(b):
        raise ValueError(f"runs differ in length: {len(a)} and {len(b)}")
    fields = ("m", "v", "m_hat", "v_hat", "update", "w")
    return {
        f: max(abs(getattr(x, f) - getattr(y, f)) for x, y in zip(a, b, strict=True))
        for f in fields
    }


def uncorrected_over_corrected(t: int, beta1: float = 0.9, beta2: float = 0.999) -> float:
    """How much bigger Adam's step is without bias correction, at step `t`, for a constant gradient.

    Exact for a constant gradient with ε → 0: `(1 − β₁ᵗ) / √(1 − β₂ᵗ)`. At step 1 with the default
    betas this is `0.1 / √0.001 ≈ 3.16`.

    Args:
        t: 1-based step.
        beta1: β₁.
        beta2: β₂.

    Returns:
        The ratio of the uncorrected step to the corrected one.
    """
    if t < 1:
        raise ValueError(f"t is 1-based, got {t}")
    return (1.0 - beta1**t) / math.sqrt(1.0 - beta2**t)


def steps_until_within(tolerance: float, beta1: float = 0.9, beta2: float = 0.999) -> int:
    """The first step from which the uncorrected step stays within `tolerance` of the corrected one.

    The ratio rises above 1, peaks, and then falls back towards 1 for good, so the answer is the
    first step after the peak at which `ratio − 1 ≤ tolerance`; past it the ratio only falls.

    Args:
        tolerance: Relative difference treated as "no longer mattering", e.g. 0.01 for 1%.
        beta1: β₁.
        beta2: β₂.

    Returns:
        A 1-based step number.
    """
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    t, previous = 1, uncorrected_over_corrected(1, beta1, beta2)
    while True:
        t += 1
        ratio = uncorrected_over_corrected(t, beta1, beta2)
        if ratio < previous and ratio - 1.0 <= tolerance:
            return t
        previous = ratio
