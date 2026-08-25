"""Turning a span of shard tokens into a window the model can eat, and the ledger can name.

**The problem.** `plan.py` hands out spans: *shard A, tokens 4096–4608*. That is a slice of a
concatenated token stream, and it knows nothing about where documents begin or end. Feed it to the
model as one undifferentiated block and every boundary guarantee `masks.py` provides is unused — the
model sees document B as a continuation of document A, which is the failure with no symptom.

**The strategy.** Shards are written as documents separated by `EOS`, so the boundaries are already
in the data; no side file to fall out of sync. Locate them, and hand `masks.py` the lengths it
needs.

**The part that is easy to get wrong.** Concat-and-chop cuts every `sequence_length` tokens without
regard for documents, so **a window usually begins in the middle of one**. Numbering that leading
fragment from position 0 tells the model it is the start of a document when it is not — the same
error that restarting positions per document exists to prevent, reintroduced at the window edge. So
the fragment carries its **true offset**, recovered by looking back in the shard for the previous
`EOS`.

Doing that lookup naively is a linear scan of everything before the span, per window; at 5M tokens
a shard and thousands of windows that is billions of operations. `DocIndex` computes each shard's
`EOS` positions **once** and answers by binary search instead.

Numpy only. These are the invariants most worth testing, and CI installs no torch.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from . import masks, spec


def hash_array(array: np.ndarray) -> str:
    """A stable content hash of an array, for the ledger.

    The dtype and shape are hashed alongside the bytes: the same numbers as `int32` and as `int64`
    are not the same tensor, and a hash that collapsed them would call two different inputs equal.

    `order="C"` is what makes a sliced view hash like its contents rather than like its strides.
    An `ascontiguousarray` call here was removed after a mutation proved it dead — `tobytes` already
    copies in the requested order — and keeping both would have read as two defences where there is
    one.

    Args:
        array: Any array.

    Returns:
        `"b2:<32 hex>"`.
    """
    header = f"{array.dtype.str}|{array.shape}|".encode()
    return "b2:" + hashlib.blake2b(header + array.tobytes(order="C"), digest_size=16).hexdigest()


@dataclass(frozen=True, slots=True)
class Fragment:
    """One document's contribution to a window.

    Attributes:
        doc_index: Which document of the shard this is.
        shard_start: Where the fragment begins in the shard.
        shard_end: Where it ends, exclusive.
        offset: How far into its own document the fragment's first token sits. Non-zero means the
            window opened part-way through a document, and positions must continue from here rather
            than restart at 0.
        complete: Whether the fragment ends where its document ends.
    """

    doc_index: int
    shard_start: int
    shard_end: int
    offset: int
    complete: bool

    @property
    def length(self) -> int:
        """Tokens in the fragment.

        Returns:
            `shard_end - shard_start`.
        """
        return self.shard_end - self.shard_start


class DocIndex:
    """Where each document begins and ends inside one shard.

    Built once per shard from the `EOS` positions. Every later question is a binary search, which is
    what keeps packing O(log n) per window instead of O(shard length).
    """

    def __init__(self, tokens: np.ndarray, *, shard_id: str = "", lane: str = "") -> None:
        """Index a shard.

        A shard whose final document is unterminated is indexed as ending at the shard's end. That
        is a real case — `split()` cuts shards at a fixed token count, not at a document boundary —
        and treating it as an error would refuse most shards.

        Args:
            tokens: The shard's token stream.
            shard_id: Recorded so fragments can name their source.
            lane: The shard's data lane. One lane per shard, so every document in it shares one.
        """
        self.shard_id = shard_id
        self.lane = lane
        self.length = int(tokens.size)
        eos = np.flatnonzero(np.asarray(tokens) == spec.EOS)
        # A document ends one past its EOS. The tail after the last EOS is a document too.
        ends = (eos + 1).tolist()
        if not ends or ends[-1] != self.length:
            ends.append(self.length)
        self._ends = np.asarray(ends, dtype=np.int64)
        self._starts = np.concatenate([[0], self._ends[:-1]])

    @property
    def count(self) -> int:
        """How many documents the shard holds.

        Returns:
            The document count.
        """
        return int(self._ends.size)

    def document_at(self, position: int) -> int:
        """Which document a token position falls in.

        Args:
            position: A token position in the shard.

        Returns:
            The document index.

        Raises:
            ValueError: If the position is outside the shard.
        """
        if not 0 <= position < self.length:
            raise ValueError(f"position {position} is outside a shard of {self.length} tokens")
        return int(np.searchsorted(self._ends, position, side="right"))

    def document_start(self, doc: int) -> int:
        """Where a document begins in the shard.

        Replay needs this to recover a fragment's offset within its own document, which is what
        keeps positions continuous across a window edge on the way back in as well as on the way
        out.

        Args:
            doc: The document index.

        Returns:
            The first token position of that document.

        Raises:
            IndexError: If there is no such document.
        """
        if not 0 <= doc < self.count:
            raise IndexError(f"document {doc} does not exist in a shard of {self.count}")
        return int(self._starts[doc])

    def fragments(self, start: int, end: int) -> list[Fragment]:
        """Every document fragment covering `[start, end)`.

        Args:
            start: First token position.
            end: One past the last.

        Returns:
            Fragments in order, together covering the range exactly once.

        Raises:
            ValueError: If the range is empty or reaches past the shard.
        """
        if end <= start:
            raise ValueError(f"empty span [{start}, {end})")
        if end > self.length:
            raise ValueError(f"span [{start}, {end}) reaches past a shard of {self.length} tokens")

        out: list[Fragment] = []
        first, last = self.document_at(start), self.document_at(end - 1)
        for doc in range(first, last + 1):
            doc_start, doc_end = int(self._starts[doc]), int(self._ends[doc])
            piece_start, piece_end = max(doc_start, start), min(doc_end, end)
            out.append(
                Fragment(
                    doc_index=doc,
                    shard_start=piece_start,
                    shard_end=piece_end,
                    offset=piece_start - doc_start,
                    complete=piece_end == doc_end,
                )
            )
        return out


@dataclass(frozen=True, slots=True)
class Window:
    """One packed sequence, with everything the model and the ledger both need.

    Attributes:
        tokens: `(sequence_length,)` of `int64` — what the model reads.
        segments: Which fragment each position belongs to; `-1` for padding.
        positions: Position within the token's own **document**, continuing across a window edge.
        loss: Which positions are graded.
        fragments: What was packed, in order.
    """

    tokens: np.ndarray
    segments: np.ndarray
    positions: np.ndarray
    loss: np.ndarray
    fragments: tuple[Fragment, ...]

    @property
    def pack_utilization(self) -> float:
        """Share of the window holding real tokens.

        Returns:
            Between 0 and 1.
        """
        return masks.utilization(self.segments)

    @property
    def loss_tokens(self) -> int:
        """Positions that actually earn gradient.

        Returns:
            The count.
        """
        return int(np.count_nonzero(self.loss))

    @property
    def pad_tokens(self) -> int:
        """Positions holding nothing.

        Returns:
            The count.
        """
        return int(np.count_nonzero(self.segments < 0))

    def hashes(self) -> dict[str, str]:
        """Content hashes of everything replay must reproduce.

        Recorded per array rather than as one digest of the lot: when replay disagrees, this says
        *which* of the four went wrong, and a mask bug and a token bug need different fixes.

        Returns:
            Keys `microbatch_hash`, `loss_mask_hash`, `position_ids_hash`, `segment_ids_hash`.
        """
        return {
            "microbatch_hash": hash_array(self.tokens),
            "loss_mask_hash": hash_array(self.loss),
            "position_ids_hash": hash_array(self.positions),
            "segment_ids_hash": hash_array(self.segments),
        }


def build_window(
    index: DocIndex,
    tokens: np.ndarray,
    start: int,
    end: int,
    *,
    window: int | None = None,
) -> Window:
    """Pack one span into a window, walled off document by document.

    Args:
        index: The shard's document index.
        tokens: The shard's token stream.
        start: First token position of the span.
        end: One past the last.
        window: Window size. Defaults to the span length, which is the concat-and-chop case.

    Returns:
        The packed window.

    Raises:
        ValueError: If the span does not fit the window.
    """
    fragments = index.fragments(start, end)
    size = end - start if window is None else window
    if end - start > size:
        raise ValueError(f"a span of {end - start} tokens does not fit a window of {size}")

    lengths = [f.length for f in fragments]
    segments = masks.segment_ids(lengths, size)
    positions = masks.position_ids(segments, offsets=[f.offset for f in fragments])

    packed = np.full(size, spec.PAD, dtype=np.int64)
    packed[: end - start] = np.asarray(tokens[start:end], dtype=np.int64)

    return Window(
        tokens=packed,
        segments=segments,
        positions=positions,
        loss=masks.loss_mask(segments, packed),
        fragments=tuple(fragments),
    )
