"""Cross-entropy, the number it becomes when you can read it, and three ways to change it.

**One objective, several implementations.** `cross_entropy` and `chunked_cross_entropy` return the
same number to floating point; what differs is how much has to exist in memory at once. Neither
changes what the model learns, and saying so is not a detail — it is the whole reason chunking is
allowed to be a default rather than a compromise.

**`perplexity` is the same loss in a form you can reason about.** Read `exp(mean loss)` as a
count: the size of the uniform menu the model is behaving as though it were picking from. That
framing makes one number worth memorising: an untrained model over a vocabulary of `V` sits
at `ln(V)`. If a run does not start near there the target alignment is wrong, and no amount of
training fixes it. It is the cheapest correctness check in this exercise. It is also **not
comparable across tokenizers** —
a tokenizer that splits text more finely is being asked an easier question at each step, so its
perplexity looks better while the model is not.

**Two knobs on the loss itself, each on a different axis.** Label smoothing changes the *target*;
z-loss changes what is penalised *besides* the prediction. Both reduce to plain cross-entropy at
zero, and `tests/` asserts each reduction rather than describing it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    import torch

from .config import Config


def cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, config: Config | None = None
) -> torch.Tensor:
    """Mean cross-entropy over the positions that count. The base every function here modifies.

    Positions whose target is `Config.ignore_index` are dropped **and** removed from the
    denominator, which is the third of the four quiet bugs `masks.py` describes: dividing by
    `batch × seq_len` instead of by what contributed scales the loss by whatever fraction of the
    batch happened to be real.

    Args:
        logits: `(n, vocab)` unnormalised scores.
        targets: `(n,)` integer class indices, or `Config.ignore_index` to drop the position.
        config: Supplies `ignore_index`. Defaults to `Config()`.

    Returns:
        A scalar tensor.
    """
    import torch.nn.functional as functional

    config = config or Config()
    return functional.cross_entropy(logits, targets, ignore_index=config.ignore_index)


def contributing(targets: torch.Tensor, config: Config | None = None) -> int:
    """How many positions the loss above actually averaged over.

    The denominator, printed. A loss that moved is evidence of nothing until you know whether the
    set of positions under it moved too.
    """
    config = config or Config()
    return int((targets != config.ignore_index).sum().item())


def perplexity(loss: torch.Tensor | float) -> float:
    """`exp(loss)` — the loss as an effective number of equally likely choices per token."""
    return math.exp(float(loss))


def untrained_perplexity(vocab_size: int) -> float:
    """What an untrained model should score: uniform over the vocabulary, so `V`.

    Returned as a float rather than as `vocab_size` directly so a caller sees it is being
    *computed* from the uniform-distribution argument, not restated.
    """
    return math.exp(math.log(vocab_size))


def label_smoothed_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 0.0,
    config: Config | None = None,
) -> torch.Tensor:
    """Cross-entropy with `epsilon` of the target's mass spread across the whole vocabulary.

    Written out rather than delegated to the framework's own `label_smoothing=` argument, because
    the point is to *see* what the smoothing does to the objective: the target keeps `1 - epsilon`,
    and `epsilon` is divided evenly over every class including the target itself, which is the
    convention the framework uses and the one the equivalence below depends on.

    At `epsilon = 0` this is exactly `cross_entropy`, and a test asserts it — an implementation that
    diverges from the base at its own no-op setting is one whose other settings cannot be trusted.

    Args:
        logits: `(n, vocab)` unnormalised scores.
        targets: `(n,)` integer class indices, or `Config.ignore_index` to drop the position.
        epsilon: Mass moved off the target, in `[0, 1)`.
        config: Supplies `ignore_index`. Defaults to `Config()`.

    Returns:
        A scalar tensor.
    """
    import torch.nn.functional as functional

    config = config or Config()
    keep = targets != config.ignore_index
    logits, targets = logits[keep], targets[keep]

    log_probs = functional.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.mean(dim=-1)
    return ((1.0 - epsilon) * nll + epsilon * smooth).mean()


def z_loss(logits: torch.Tensor) -> torch.Tensor:
    """The auxiliary penalty on `logsumexp(logits)`, squared and averaged.

    It does not change what the model predicts — softmax is shift-invariant, so adding a constant to
    every logit leaves the distribution untouched — which is precisely why the raw logit scale is
    free to drift, and why it does. The cross-entropy gradient sums to zero, so nothing in the
    primary loss can push the logits up or down as a group; that direction is simply unconstrained.
    `log Z` then walks, bf16 loses precision, and a healthy-looking run produces a NaN thousands of
    steps later. **This penalty constrains something the primary loss cannot see.**

    It is not interchangeable with the other two fixes for the same drift. Logit soft-capping bounds
    the logits and therefore bounds `log Z` with them. Output-embedding centering pins the *mean*
    logit and says nothing about the spread, so it can leave `log Z` higher than a plain run. Three
    mechanisms, not three settings of one.
    """
    import torch

    return torch.logsumexp(logits, dim=-1).pow(2).mean()


def cross_entropy_with_z_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: float = 0.0,
    config: Config | None = None,
) -> torch.Tensor:
    """`cross_entropy` plus `weight ×` the z-loss. At `weight = 0` it is exactly the base."""
    loss = cross_entropy(logits, targets, config)
    if weight == 0.0:
        return loss
    return loss + weight * z_loss(logits)


def chunked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 128,
    config: Config | None = None,
) -> torch.Tensor:
    """Cross-entropy computed in row blocks, to cap the peak size of the softmax intermediate.

    Process `chunk_size` rows, get their summed loss, throw those logits away, take the next block.
    Peak logit memory becomes `chunk_size × vocab × bytes` instead of `rows × vocab × bytes`, and
    the loss is the same number. Backward recomputes a block's logits at the moment it needs
    them, so arithmetic is spent to avoid holding memory — and that exchange is the technique, not
    a detail of it.

    **Two denominators can go wrong here and only one is obvious.** Averaging the chunk *means*
    rather than summing and dividing once is wrong whenever the last chunk is short. Dividing by the
    row count rather than by the *contributing* count is wrong whenever anything is masked — and
    that one survives every test written on unmasked input, which is how it ships.

    Args:
        logits: `(n, vocab)` unnormalised scores.
        targets: `(n,)` integer class indices, or `Config.ignore_index` to drop the position.
        chunk_size: Rows per block. Must be positive.
        config: Supplies `ignore_index`. Defaults to `Config()`.

    Returns:
        A scalar tensor equal to `cross_entropy(logits, targets)` within floating-point tolerance.

    Raises:
        ValueError: When `chunk_size` is not positive.
    """
    import torch
    import torch.nn.functional as functional

    config = config or Config()
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    total = torch.zeros((), dtype=logits.dtype, device=logits.device)
    counted = 0
    for start in range(0, logits.shape[0], chunk_size):
        stop = start + chunk_size
        block_targets = targets[start:stop]
        total = total + functional.cross_entropy(
            logits[start:stop],
            block_targets,
            ignore_index=config.ignore_index,
            reduction="sum",
        )
        counted += int((block_targets != config.ignore_index).sum().item())

    if counted == 0:
        return total * 0.0
    return total / counted
