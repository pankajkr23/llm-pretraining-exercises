"""Exercise 11's decoder, with its residual stack run by a chosen rule.

The embeddings, blocks, final norm and head are exercise 11's `optimizers.model.GPT`, unchanged;
only how the blocks are chained differs:

- `standard` — the ordinary residual stack, `p⁽ℓ⁺¹⁾ = p⁽ℓ⁾ + f_ℓ(p⁽ℓ⁾)` (forward Euler), trained
  with ordinary autograd. This is the baseline.
- `midpoint`, `blend`, `leapfrog` — the reversible rules in `reversible.stack`, trained either with
  activations rebuilt in backward (`memory="reversible"`, the point of the exercise) or with
  ordinary autograd through the same recurrence (`memory="stored"`, the reference the tests compare
  against).

Reusing exercise 11's model means a reversible run and the baseline share every weight shape and
every initial value for a given seed: the only difference between them is the chaining.
"""

import copy

import torch
from optimizers.model import GPT, ModelConfig

from .memory import chunked_loss
from .stack import Rule, reconstruction_error, run_reversible, run_stored

VARIANTS = ("standard", "midpoint", "blend", "leapfrog")


class ChainedGPT(GPT):
    """`optimizers.model.GPT` whose blocks are chained by `variant`.

    Args:
        config: Exercise 11's model shape.
        variant: `standard` or a reversible rule's name.
        h: Step size for a reversible rule.
        a: The blend's weight on the state two layers back.
        memory: `reversible` (rebuild in backward) or `stored` (ordinary autograd).
        loss_chunk: Rows per chunk of the recomputed loss (`reversible.memory.chunked_loss`).
        seed: Initialisation seed — the same seed gives the same weights for every variant.
    """

    def __init__(
        self,
        config: ModelConfig,
        variant: str = "standard",
        h: float = 0.5,
        a: float = 0.5,
        memory: str = "reversible",
        loss_chunk: int = 2048,
        seed: int = 0,
    ) -> None:
        """Build exercise 11's model and record how its blocks are chained."""
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if memory not in ("reversible", "stored"):
            raise ValueError(f"memory must be 'reversible' or 'stored', got {memory!r}")
        super().__init__(config, seed=seed)
        self.variant, self.memory, self.loss_chunk = variant, memory, loss_chunk
        self.rule = None if variant == "standard" else Rule.make(variant, h, a)

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """Embed, run the stack by the chosen rule, normalise."""
        positions = torch.arange(ids.shape[1], device=ids.device)
        p0 = self.tokens(ids) + self.positions(positions)
        if self.rule is None:
            x = p0
            for block in self.blocks:
                x = block(x)
        elif self.memory == "reversible" and torch.is_grad_enabled():
            x = run_reversible(self.blocks, self.rule, p0)
        else:
            x = run_stored(self.blocks, self.rule, p0)
        return self.norm(x)

    def loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Mean next-token cross-entropy, through the chunked loss every variant shares."""
        hidden = self.hidden(batch[:, :-1])
        targets = batch[:, 1:]
        return chunked_loss(
            hidden.reshape(-1, hidden.shape[-1]),
            self.head.weight,
            targets.reshape(-1),
            self.loss_chunk,
        )


def rebuild_agreement(model: ChainedGPT, batch: torch.Tensor) -> dict[str, float]:
    """How closely rebuilding in backward matches storing, at this model's weights, dtype, device.

    The tests hold the two to each other in float64 on a CPU. A real run is float32 on a GPU, deeper
    and with trained weights, and the inversion amplifies rounding — so this measures the same two
    quantities on the run's own model: the worst relative error of a state rebuilt from the top, and
    how far the gradients of `memory="reversible"` are from those of `memory="stored"`. Both copies
    are taken from `model`, which is left untouched.

    Args:
        model: A model chained by a reversible rule.
        batch: `(batch, seq_len + 1)` token ids, on the model's device.

    Returns:
        `rebuild_error` (worst relative error over the rebuilt states), `gradient_error`
        (‖g_rev − g_stored‖ / ‖g_stored‖ over all parameters together) and
        `gradient_error_worst_tensor` (the same ratio for the worst single parameter tensor).
    """
    if model.rule is None:
        raise ValueError("the standard variant rebuilds nothing")
    grads = []
    for memory in ("reversible", "stored"):
        twin = copy.deepcopy(model)
        twin.memory = memory
        twin.zero_grad(set_to_none=True)
        twin.loss(batch).backward()
        grads.append([p.grad.double() for p in twin.parameters()])
        del twin
    rebuilt, stored = grads
    diff = [(a - b).norm() for a, b in zip(rebuilt, stored, strict=True)]
    norm = [b.norm() for b in stored]
    total = float(torch.stack(diff).norm() / torch.stack(norm).norm().clamp_min(1e-300))
    worst = max(float(d / n.clamp_min(1e-300)) for d, n in zip(diff, norm, strict=True))
    with torch.no_grad():
        ids = batch[:, :-1]
        positions = torch.arange(ids.shape[1], device=ids.device)
        p0 = model.tokens(ids) + model.positions(positions)
        rebuild = reconstruction_error(model.blocks, model.rule, p0)
    return {
        "rebuild_error": rebuild,
        "gradient_error": total,
        "gradient_error_worst_tensor": worst,
    }
