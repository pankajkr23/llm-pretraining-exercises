"""A reversible residual stack: activations rebuilt during the backward pass instead of stored.

A standard transformer advances its residual stream one block at a time,

    p⁽ℓ⁺¹⁾ = p⁽ℓ⁾ + f_ℓ(p⁽ℓ⁾)                                   (forward Euler)

and to backpropagate it must keep every `p⁽ℓ⁾` and every intermediate inside every block, so
activation memory grows with depth. That update cannot be run backwards: recovering `p⁽ℓ⁾` from
`p⁽ℓ⁺¹⁾` needs `f_ℓ(p⁽ℓ⁾)`, which needs the state being recovered.

Gal et al., *Reversing Large Language Models for Efficient Training and Fine-Tuning*
(arXiv:2512.02056v1), replace it with two-step rules that can. Every rule implemented here has one
shape,

    p⁽ℓ⁺¹⁾ = A·p⁽ℓ⁻¹⁾ + B·p⁽ℓ⁾ + C·f_ℓ(p⁽ℓ⁾)            with A ≠ 0,

so it inverts exactly as

    p⁽ℓ⁻¹⁾ = (p⁽ℓ⁺¹⁾ − B·p⁽ℓ⁾ − C·f_ℓ(p⁽ℓ⁾)) / A,

and `f_ℓ` is evaluated at `p⁽ℓ⁾` — the state the backward pass already holds. The three rules, with
their equation numbers in that paper:

| rule | equation | A | B | C |
| --- | --- | --- | --- | --- |
| midpoint | (4) `p⁽ℓ⁺¹⁾ = p⁽ℓ⁻¹⁾ + 2h·f(p⁽ℓ⁾)` | 1 | 0 | 2h |
| blend ("midpoint (a)") | (15) `p_{j+1} = a·p_{j−1} + (1−a)·p_j + h·f(p_j)` | a | 1 − a | h |
| leapfrog | (6) `p⁽ℓ⁺¹⁾ = 2p⁽ℓ⁾ − p⁽ℓ⁻¹⁾ + h²·f(p⁽ℓ⁾)` | −1 | 2 | h² |

The paper notes the blend behaves in expectation like forward Euler (its Eq 18); at `a = 0` it *is*
forward Euler, and stops being invertible, which is why `a` is refused at 0.

**What the forward pass keeps.** Only the two states entering the stack and the two leaving it. The
backward pass walks down the layers: at each it re-runs `f_ℓ` on the state it holds — once, with
autograd on — uses that one evaluation both to rebuild the state below and to backpropagate through
`f_ℓ`, then frees it. So activation memory is one block's worth, whatever the depth, and the price
is one extra forward evaluation per block. A plain-autograd reference (`run_stored`) computes the
same recurrence and keeps everything; the tests require the two to agree.

**Seeding the recurrence.** A two-step rule needs two starting states. The paper's text does not
state how its first step is seeded; here `p⁽⁻¹⁾ = p⁽⁰⁾`, the embedding output, so the first layer
takes an ordinary step. That is our choice (DECISIONS D2).

**The forward pass must be deterministic.** The rebuild re-runs `f_ℓ`; a dropout mask drawn twice
would differ, and the rebuilt state would be wrong. These blocks have no dropout.
"""

from dataclasses import dataclass

import torch
from torch import nn

RULES = ("midpoint", "blend", "leapfrog")


@dataclass(frozen=True)
class Rule:
    """One reversible update, as its three coefficients.

    Attributes:
        name: `midpoint`, `blend` or `leapfrog`.
        A: Weight on the state two layers back. Never zero.
        B: Weight on the current state.
        C: Weight on the block's update.
    """

    name: str
    A: float
    B: float
    C: float

    @classmethod
    def make(cls, name: str, h: float, a: float = 0.5) -> "Rule":
        """The coefficients for one rule at step size `h` (and blend `a`).

        Args:
            name: One of `RULES`.
            h: Step size, > 0.
            a: The blend's weight on the state two layers back; only used by `blend`.

        Returns:
            The rule.
        """
        if h <= 0:
            raise ValueError(f"h must be positive, got {h}")
        if name == "midpoint":
            return cls(name, 1.0, 0.0, 2.0 * h)
        if name == "blend":
            if a == 0:
                raise ValueError("a = 0 is forward Euler, which cannot be inverted")
            return cls(name, a, 1.0 - a, h)
        if name == "leapfrog":
            return cls(name, -1.0, 2.0, h * h)
        raise ValueError(f"rule must be one of {RULES}, got {name!r}")

    def step(
        self, previous: torch.Tensor, current: torch.Tensor, update: torch.Tensor
    ) -> torch.Tensor:
        """`A·previous + B·current + C·update`."""
        return self.A * previous + self.B * current + self.C * update

    def invert(
        self, nxt: torch.Tensor, current: torch.Tensor, update: torch.Tensor
    ) -> torch.Tensor:
        """The state two layers back, from the two above it and the block's update between."""
        return (nxt - self.B * current - self.C * update) / self.A


def run_stored(blocks: nn.ModuleList, rule: Rule, p0: torch.Tensor) -> torch.Tensor:
    """The recurrence with ordinary autograd, keeping every state: the reference."""
    previous, current = p0, p0
    for block in blocks:
        previous, current = current, rule.step(previous, current, block.delta(current))
    return current


class _ReversibleStack(torch.autograd.Function):
    """Forward without a graph; backward by rebuilding each layer's input from the layer above."""

    @staticmethod
    def forward(ctx, p0: torch.Tensor, blocks: nn.ModuleList, rule: Rule, *params: torch.Tensor):
        with torch.no_grad():
            previous, current = p0, p0
            for block in blocks:
                previous, current = current, rule.step(previous, current, block.delta(current))
        ctx.blocks, ctx.rule = blocks, rule
        # The boundary states only: the two at the top. Nothing per layer.
        ctx.save_for_backward(previous, current)
        return current

    @staticmethod
    def backward(ctx, grad_top: torch.Tensor):
        blocks, rule = ctx.blocks, ctx.rule
        below_top, top = ctx.saved_tensors
        # State pair (lower, upper) = (p⁽ℓ⁾, p⁽ℓ⁺¹⁾) and the gradient arriving at each.
        lower, upper = below_top, top
        grad_lower, grad_upper = torch.zeros_like(top), grad_top
        param_grads: list[torch.Tensor] = []
        for block in reversed(blocks):
            # This block made `upper` from (p⁽ℓ⁻¹⁾, lower): rebuild p⁽ℓ⁻¹⁾ and backprop through f.
            with torch.enable_grad():
                x = lower.detach().requires_grad_(True)
                update = block.delta(x)
            rebuilt = rule.invert(upper, lower, update.detach())
            weights = [p for p in block.parameters() if p.requires_grad]
            grads = torch.autograd.grad(update, (x, *weights), grad_outputs=rule.C * grad_upper)
            # ∂upper/∂p⁽ℓ⁻¹⁾ = A·I ;  ∂upper/∂lower = B·I + C·∂f/∂lower
            grad_below = rule.A * grad_upper
            grad_lower = grad_lower + rule.B * grad_upper + grads[0]
            param_grads = list(grads[1:]) + param_grads
            lower, upper = rebuilt, lower
            grad_lower, grad_upper = grad_below, grad_lower
            del update, x
        # The stack was seeded with p⁽⁻¹⁾ = p⁽⁰⁾ = p0: both gradients belong to p0.
        grad_p0 = grad_lower + grad_upper
        return (grad_p0, None, None, *param_grads)


def run_reversible(blocks: nn.ModuleList, rule: Rule, p0: torch.Tensor) -> torch.Tensor:
    """The recurrence with activations rebuilt in backward; output and gradients as `run_stored`."""
    params = [p for block in blocks for p in block.parameters() if p.requires_grad]
    return _ReversibleStack.apply(p0, blocks, rule, *params)


def reconstruction_error(blocks: nn.ModuleList, rule: Rule, p0: torch.Tensor) -> float:
    """Largest relative error when every state is rebuilt from the top two, layer by layer.

    Runs the recurrence forwards keeping every state, then inverts it from the top and compares each
    rebuilt state with the stored one. In exact arithmetic it is zero; in floating point it measures
    how much the inversion amplifies rounding: the practical form of the stability analysis.
    """
    with torch.no_grad():
        states = [p0, p0]
        for block in blocks:
            states.append(rule.step(states[-2], states[-1], block.delta(states[-1])))
        lower, upper = states[-2], states[-1]
        worst = 0.0
        for index, block in zip(range(len(states) - 3, -1, -1), reversed(blocks), strict=True):
            rebuilt = rule.invert(upper, lower, block.delta(lower))
            truth = states[index]
            worst = max(worst, float((rebuilt - truth).norm() / truth.norm().clamp_min(1e-30)))
            lower, upper = rebuilt, lower
        return worst
