"""Which positions are allowed to contribute, and the count that proves it.

Two positions must never reach the loss, for different reasons:

**Padding.** Batched sequences have different lengths, so short ones are padded. A padding token is
not a prediction. Train on it and the loss *improves*, because padding is trivially predictable —
the number gets better while the model gets worse.

**A packed document boundary.** Exercise 06 packs many documents into one fixed-length sequence to
avoid wasting compute on padding, and that packing creates a join: the last token of one document
sits beside the first token of the next, with no relationship whatsoever. Score that pair and
the gradient asserts a continuation between two texts that have nothing to do with one another.

**The count is the evidence, and it is why these functions return one.** Both bugs are silent —
they move the loss without raising, and a loss that moved could have moved for any reason. The
number of positions that actually contributed says *which* lie was told, which is exactly what the
requirements ask to be shown.

**And the denominator is a third bug hiding behind the same code.** The mean must divide by the
positions that counted, not by `batch × seq_len`. Divide by the wrong one and every batch is scaled
by whatever fraction of it happened to be real — a number that changes batch to batch, for reasons
that have nothing to do with learning. `masked_targets` sets dropped positions to
`Config.ignore_index`, and the framework's cross-entropy then divides by the right count on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    import torch

from .config import Config


@dataclass(frozen=True)
class MaskReport:
    """What a masking step did, in the form the requirements ask for it to be printed.

    Attributes:
        total: Positions the shift produced, before any masking.
        contributing: Positions that will reach the loss.
        dropped: `total - contributing`. Named rather than derived at the call site, because this
            is the number a reader checks against what they expected to be masked.
        reason: What was masked, in words.
    """

    total: int
    contributing: int
    dropped: int
    reason: str

    def __str__(self) -> str:
        """The one line this report exists to be printed as."""
        share = self.dropped / self.total if self.total else 0.0
        return (
            f"{self.reason}: {self.contributing:,} of {self.total:,} positions contribute "
            f"({self.dropped:,} dropped, {share:.1%})"
        )


def pad_sequences(
    sequences: list[list[int]], seq_len: int, config: Config | None = None
) -> torch.Tensor:
    """Right-pad ragged sequences to `[len(sequences), seq_len]` with `Config.pad_id`.

    Args:
        sequences: Token ids, any lengths. Longer than `seq_len` is truncated.
        seq_len: The width every sequence is brought to.
        config: Supplies `pad_id`. Defaults to `Config()`.

    Returns:
        A `[batch, seq_len]` long tensor.
    """
    import torch

    config = config or Config()
    rows = [
        (sequence[:seq_len] + [config.pad_id] * max(0, seq_len - len(sequence)))
        for sequence in sequences
    ]
    return torch.tensor(rows, dtype=torch.long)


def masked_targets(
    targets: torch.Tensor, keep: torch.Tensor, config: Config | None = None
) -> torch.Tensor:
    """Targets with every dropped position set to `Config.ignore_index`.

    Args:
        targets: `[batch, positions]` target ids.
        keep: `[batch, positions]` boolean, `True` where the position may contribute.
        config: Supplies `ignore_index`. Defaults to `Config()`.

    Returns:
        A tensor of the same shape, safe to hand straight to cross-entropy.
    """
    import torch

    config = config or Config()
    return torch.where(keep, targets, torch.full_like(targets, config.ignore_index))


def keep_non_padding(
    inputs: torch.Tensor, targets: torch.Tensor, config: Config | None = None
) -> tuple[torch.Tensor, MaskReport]:
    """Drop any position whose input **or** target is padding.

    Both halves matter and only one is obvious. A padded *target* is a prediction of padding, which
    is the case everyone remembers. A padded *input* is a position predicting from nothing, which
    is just as meaningless and is the one that gets left in.

    Args:
        inputs: `[batch, positions]` input ids.
        targets: `[batch, positions]` target ids.
        config: Supplies `pad_id`. Defaults to `Config()`.

    Returns:
        `(keep, report)` — a boolean mask and the count that evidences it.
    """
    config = config or Config()
    keep = (inputs != config.pad_id) & (targets != config.pad_id)
    total = int(keep.numel())
    contributing = int(keep.sum().item())
    return keep, MaskReport(total, contributing, total - contributing, "padding masked")


NO_DOCUMENT = -1
"""The document id padding carries. It is not a document, and it is not a boundary either."""


def keep_within_document(
    document_ids: torch.Tensor, horizon: int = 1
) -> tuple[torch.Tensor, MaskReport]:
    """Drop any position whose target is in a **different** document, or in no document at all.

    **The second half of that sentence was missing, and the bug it caused is worth recording.** The
    first version was `keep = source == destination`, which reads correctly and is not: padding
    carries `NO_DOCUMENT`, and `-1 == -1` is `True`, so every pad-to-pad pair was **kept**. On the
    packed example this exercise publishes, 68 of the 125 "contributing" positions were padding
    predicting padding — in the exercise whose item 3 exists to say that must never happen.

    Nothing caught it. The guard asserted `report.dropped` equalled a count of *transitions*, which
    is the same expression the buggy implementation used, so it held for any input at all.

    Args:
        document_ids: `[batch, seq_len]`, one id per position saying which document it came from.
            `NO_DOCUMENT` marks a position that belongs to none — padding.
        horizon: Positions ahead the target sits. A `t+2` head crosses a boundary two positions
            early, so a mask built for `t+1` would leave one crossing pair per join — which is the
            kind of near-miss that survives review.

    Returns:
        `(keep, report)`, with `keep` shaped `[batch, seq_len - horizon]` to match the shift.
    """
    source = document_ids[:, :-horizon]
    destination = document_ids[:, horizon:]
    keep = (source == destination) & (source != NO_DOCUMENT) & (destination != NO_DOCUMENT)
    total = int(keep.numel())
    contributing = int(keep.sum().item())
    return (
        keep,
        MaskReport(total, contributing, total - contributing, "document boundaries masked"),
    )


def pack_documents(
    documents: list[list[int]], seq_len: int, config: Config | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate documents into one sequence, recording which document each position came from.

    This is the arrangement exercise 06 uses and the one item 4 of the requirements asks for: two
    documents in one sequence, with a join in the middle that must not be predicted across.

    Args:
        documents: Token ids per document, concatenated in order.
        seq_len: Width of the resulting sequence; padded or truncated to fit.
        config: Supplies `pad_id`. Defaults to `Config()`.

    Returns:
        `(tokens, document_ids)`, both `[1, seq_len]`. Padding carries `NO_DOCUMENT`, which is not
        a document — `keep_within_document` drops those positions explicitly rather than by
        comparing them to each other, since two padding positions do compare equal.
    """
    import torch

    config = config or Config()
    tokens: list[int] = []
    owners: list[int] = []
    for index, document in enumerate(documents):
        tokens.extend(document)
        owners.extend([index] * len(document))

    tokens = tokens[:seq_len]
    owners = owners[:seq_len]
    padding = seq_len - len(tokens)
    tokens.extend([config.pad_id] * padding)
    owners.extend([NO_DOCUMENT] * padding)

    return (
        torch.tensor([tokens], dtype=torch.long),
        torch.tensor([owners], dtype=torch.long),
    )
