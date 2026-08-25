"""From a coordinate to a microbatch — the last torch-free step before the model.

**The problem.** `plan.py` answers *which span* belongs at `(step, rank, accum, seq)`.
`pack.py` turns one span into one window. Something has to hold the shards open, run those two
together for every sequence in a microbatch, and produce both the arrays the model eats **and** the
record the ledger writes. Those two must be built from the same objects or they can disagree, and a
ledger that disagrees with what was fed is worse than no ledger.

**The strategy.** One object per shard, opened once (`ShardHandle`), and one function that assembles
a microbatch and its ledger fields **together**, from the same windows. There is no code path that
produces one without the other.

**Every shard is verified when it is opened, not just when it was written.** `0444` and a read-only
memmap protect a handle; neither survives a shell, a rebuild, or a restore from a stale backup.
Re-hashing on open is what catches those, and it is the reason a ledger entry naming
`(shard_id, start, end)` means something months later.

Numpy only. The microbatch crosses into torch in `train.py` and nowhere else.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import masks, pack, shards
from . import plan as plan_module


@dataclass(frozen=True, slots=True)
class ShardHandle:
    """One shard, opened and indexed.

    Attributes:
        shard_id: Content-addressed id.
        lane: Which data lane the shard came from. One lane per shard, so every document in it
            shares one and the ledger can attribute tokens to lanes without a per-document label.
        tokens: The token stream, read-only.
        index: Document boundaries within it.
        content_hash: Re-derived at open time, not copied from the manifest.
    """

    shard_id: str
    lane: str
    tokens: np.ndarray
    index: pack.DocIndex
    content_hash: str


def open_shard(shard_id: str, path, lane: str, *, expected_hash: str | None = None) -> ShardHandle:
    """Open a shard, verify it, and index its documents.

    Args:
        shard_id: The shard's id.
        path: Where its tokens live.
        lane: Which data lane it came from.
        expected_hash: What the manifest recorded. Verified when given.

    Returns:
        The handle.

    Raises:
        ValueError: If the bytes on disk no longer hash to what was recorded — which means every
            span pointing into this shard is now naming different tokens than the ledger says.
    """
    tokens = shards.read(path)
    content_hash = shards.content_hash(np.asarray(tokens))
    if expected_hash is not None and content_hash != expected_hash:
        raise ValueError(
            f"shard {shard_id} hashes to {content_hash} but its manifest records {expected_hash} — "
            f"the bytes changed after they were sealed, and every ledger entry naming this shard "
            f"now points at different tokens"
        )
    return ShardHandle(
        shard_id=shard_id,
        lane=lane,
        tokens=tokens,
        index=pack.DocIndex(np.asarray(tokens), shard_id=shard_id, lane=lane),
        content_hash=content_hash,
    )


@dataclass(frozen=True, slots=True)
class Microbatch:
    """What one rank feeds the model in one accumulation slot, plus what the ledger records.

    Attributes:
        tokens: `(microbatch, sequence_length)` of `int64`.
        segments: Which fragment each position belongs to; `-1` for padding.
        positions: Position within each token's own document.
        loss: Which positions are graded.
        additive: `(microbatch, 1, sequence_length, sequence_length)` attention bias.
        windows: The packed windows, in order.
        samples: One entry per document fragment, for the ledger.
    """

    tokens: np.ndarray
    segments: np.ndarray
    positions: np.ndarray
    loss: np.ndarray
    additive: np.ndarray
    windows: tuple[pack.Window, ...]
    samples: tuple[dict, ...]

    @property
    def token_count(self) -> int:
        """Token positions in the microbatch, padding included.

        Returns:
            The count.
        """
        return int(self.tokens.size)

    @property
    def loss_token_count(self) -> int:
        """Positions that earn gradient.

        Returns:
            The count.
        """
        return int(np.count_nonzero(self.loss))

    @property
    def pad_token_count(self) -> int:
        """Positions holding nothing.

        Returns:
            The count.
        """
        return int(np.count_nonzero(self.segments < 0))

    @property
    def pack_utilization(self) -> float:
        """Share of the microbatch holding real tokens.

        Returns:
            Between 0 and 1.
        """
        return float(np.count_nonzero(self.segments >= 0) / self.segments.size)

    @property
    def lane_mix(self) -> dict[str, int]:
        """Real tokens contributed by each lane.

        The number a mixture claim is checked against. Recorded per microbatch rather than
        aggregated, because "the mix over the run" is a sum an auditor should be able to recompute
        rather than a number it has to trust.

        Returns:
            Lane name to token count.
        """
        counts: Counter[str] = Counter()
        for sample in self.samples:
            counts[sample["lane"]] += sample["end"] - sample["start"]
        return dict(sorted(counts.items()))

    def hashes(self) -> dict[str, str]:
        """Content hashes of everything replay must reproduce.

        Returns:
            Keys `microbatch_hash`, `loss_mask_hash`, `position_ids_hash`, `segment_ids_hash`.
        """
        return {
            "microbatch_hash": pack.hash_array(self.tokens),
            "loss_mask_hash": pack.hash_array(self.loss),
            "position_ids_hash": pack.hash_array(self.positions),
            "segment_ids_hash": pack.hash_array(self.segments),
        }


def build_microbatch(
    schedule: plan_module.Plan,
    handles: dict[str, ShardHandle],
    step: int,
    rank: int,
    accum: int,
) -> Microbatch:
    """Assemble one microbatch, and the ledger record for it, from the same windows.

    Args:
        schedule: The run's plan.
        handles: Every open shard, by id.
        step: Optimizer step.
        rank: Worker process.
        accum: Accumulation slot within the step.

    Returns:
        The microbatch.

    Raises:
        KeyError: If the plan names a shard that was not opened. Refused rather than skipped: a
            missing shard would silently shrink the batch and every count downstream with it.
    """
    config = schedule.config
    windows: list[pack.Window] = []
    samples: list[dict] = []

    for seq in range(config.microbatch):
        coord = plan_module.Coordinate(step=step, rank=rank, accum=accum, seq=seq)
        span = schedule.span_for(coord)
        if span.shard_id not in handles:
            raise KeyError(
                f"the plan places shard {span.shard_id} at step {step} rank {rank} accum {accum} "
                f"seq {seq}, but that shard was never opened"
            )
        handle = handles[span.shard_id]
        window = pack.build_window(handle.index, handle.tokens, span.start, span.end)
        windows.append(window)

        pass_no = schedule.pass_number(coord)
        at = 0
        for fragment in window.fragments:
            graded = int(np.count_nonzero(window.loss[at : at + fragment.length]))
            samples.append(
                {
                    "shard_id": handle.shard_id,
                    "start": fragment.shard_start,
                    "end": fragment.shard_end,
                    "lane": handle.lane,
                    "loss_tokens": graded,
                    "pass_no": pass_no,
                    "window": seq,
                }
            )
            at += fragment.length

    segments = np.stack([w.segments for w in windows])
    return Microbatch(
        tokens=np.stack([w.tokens for w in windows]),
        segments=segments,
        positions=np.stack([w.positions for w in windows]),
        loss=np.stack([w.loss for w in windows]),
        additive=np.stack([masks.additive_mask(w.segments) for w in windows])[:, None, :, :],
        windows=tuple(windows),
        samples=tuple(samples),
    )
