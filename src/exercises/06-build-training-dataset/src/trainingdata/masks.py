"""Attention masks, position ids and loss masks for packed sequences.

**The problem.** Documents vary in length; the window is always the same size. Pad the remainder and
you burn compute on nothing — *"padding is literally being on the chair for 8 hours"* — and worse,
the model learns to predict padding, which it does effortlessly, so the loss looks good and means
nothing. Pack several documents into one window instead and a different failure appears: the model
may learn that unrelated text is a natural continuation of whatever preceded it.

**The strategy.** Pack, then **wall the documents off from each other**:

- the attention mask is **block-diagonal**, so document B cannot see document A at all;
- position ids **restart per document**, so document B does not appear to begin at position 400;
- the loss mask decides which tokens are graded, so context and padding earn nothing.

Everything here is numpy. That is deliberate: these are the invariants most worth testing, and
keeping them out of torch is what lets CI — which installs no torch — test them.

**Position ids restart per document.** The alternative, running them continuously across the window,
would tell the model that a document beginning at offset 400 is 400 tokens into something. It is
not; it is the start of its own text, and it will be at position 0 at inference. Restarting is what
makes packing invisible to the model, which is the entire goal.
"""

import numpy as np

from . import spec

#: Additive mask value for a disallowed position. `-inf` is avoided on purpose: a row that is
#: entirely `-inf` produces `nan` after softmax rather than a uniform row, and a single `nan`
#: poisons every gradient it touches. A large finite negative underflows to zero weight instead.
NEG = -1e9


def segment_ids(lengths: list[int], window: int) -> np.ndarray:
    """Which document each position belongs to.

    Padding is segment `-1` — a real segment id would make padding attend to itself and, worse,
    look like a document to every check downstream.

    Args:
        lengths: Token count of each packed document, in order.
        window: Total positions.

    Returns:
        `(window,)` of `int32`.

    Raises:
        ValueError: If the documents do not fit.
    """
    total = sum(lengths)
    if total > window:
        raise ValueError(f"{total} tokens do not fit a window of {window}")
    out = np.full(window, -1, dtype=np.int32)
    at = 0
    for i, length in enumerate(lengths):
        out[at : at + length] = i
        at += length
    return out


def position_ids(segments: np.ndarray, *, offsets: list[int] | None = None) -> np.ndarray:
    """Position of each token **within its own document**.

    `offsets` is what makes this correct at a window edge. Concat-and-chop cuts every
    `sequence_length` tokens without regard for documents, so a window usually opens part-way
    through one. Numbering that leading fragment from 0 tells the model it is the start of a
    document when it is not — the same error restarting positions exists to prevent, reintroduced
    at the seam. Passing the fragment's true offset continues the numbering instead.

    Args:
        segments: As returned by `segment_ids`.
        offsets: How far into its own document each segment's first token sits. Defaults to zero
            for every segment, which is right only when each segment is a whole document.

    Returns:
        `(window,)` of `int32`. Padding positions are `0`, which is arbitrary and unused — nothing
        attends to them and nothing is graded on them.

    Raises:
        ValueError: If `offsets` does not have one entry per segment. Silently zero-filling a short
            list would restart exactly the fragments the argument exists to fix.
    """
    out = np.zeros_like(segments, dtype=np.int32)
    present = [int(s) for s in np.unique(segments) if s >= 0]
    if offsets is not None and len(offsets) != len(present):
        raise ValueError(
            f"{len(offsets)} offsets for {len(present)} segments — one per segment, in order"
        )
    for i, seg in enumerate(present):
        where = np.flatnonzero(segments == seg)
        base = 0 if offsets is None else offsets[i]
        out[where] = base + np.arange(where.size, dtype=np.int32)
    return out


def attention_mask(segments: np.ndarray) -> np.ndarray:
    """Block-diagonal causal mask.

    A position may attend to earlier positions **of its own document**, and to itself. Not to a
    later position, and never to another document.

    Args:
        segments: As returned by `segment_ids`.

    Returns:
        `(window, window)` boolean. `True` means *allowed*.
    """
    same = segments[:, None] == segments[None, :]
    causal = np.tril(np.ones((segments.size, segments.size), dtype=bool))
    real = segments[:, None] >= 0
    return same & causal & real


def additive_mask(segments: np.ndarray) -> np.ndarray:
    """The same mask as an additive bias, for attention implementations that want one.

    Args:
        segments: As returned by `segment_ids`.

    Returns:
        `(window, window)` of `float32`: `0.0` where allowed, `NEG` where not.
    """
    return np.where(attention_mask(segments), 0.0, NEG).astype(np.float32)


def loss_mask(
    segments: np.ndarray, tokens: np.ndarray, *, context_spans: list[tuple[int, int]] | None = None
) -> np.ndarray:
    """Which positions contribute to the gradient.

    Excluded: padding, and any span declared context-only (an SFT prompt, a tool observation).
    Included by default: everything else, which is what plain pretraining wants.

    **The last token of each document is excluded**, because next-token prediction has no target for
    it — its "next token" is the first token of an unrelated document, and grading that would teach
    exactly the cross-document continuation the block-diagonal mask exists to prevent.

    Args:
        segments: As returned by `segment_ids`.
        tokens: The packed token ids, same length.
        context_spans: Half-open `[start, end)` ranges that provide context but earn no loss.

    Returns:
        `(window,)` boolean. `True` means graded.
    """
    keep = segments >= 0
    keep &= tokens != spec.PAD

    for seg in np.unique(segments):
        if seg < 0:
            continue
        where = np.flatnonzero(segments == seg)
        keep[where[-1]] = False  # no target for the final token of a document

    for start, end in context_spans or []:
        keep[start:end] = False

    return keep


def utilization(segments: np.ndarray) -> float:
    """Share of the window holding real tokens.

    Args:
        segments: As returned by `segment_ids`.

    Returns:
        Between 0 and 1.
    """
    return float(np.count_nonzero(segments >= 0) / segments.size)


def loss_utilization(mask: np.ndarray) -> float:
    """Share of the window that actually earns gradient.

    The number that matters, and the one a raw token-throughput figure hides: a loader can report a
    high tokens-per-second while most of those tokens are padding or context.

    Args:
        mask: As returned by `loss_mask`.

    Returns:
        Between 0 and 1.
    """
    return float(np.count_nonzero(mask) / mask.size)
