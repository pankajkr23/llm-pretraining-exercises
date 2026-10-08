"""Ring reduce-scatter, all-gather and all-reduce — run for real across the in-process ranks.

**Each is the actual ring algorithm, not `torch.sum` with a byte count attached.** The buffer on
every rank is cut into `N` equal chunks. Rank `r` only ever sends to rank `r + 1`, one chunk per
step, and the collective takes `N − 1` steps:

- **reduce-scatter.** At step `s`, rank `r` forwards its running partial sum of chunk
  `r − s − 1`; the receiver adds its own copy of that chunk and forwards the result next step.
  After `N − 1` steps each chunk has visited every rank once, and **rank `r` holds the complete sum
  of chunk `r`** — and nothing else.
- **all-gather.** The reverse: rank `r` starts with chunk `r`, and at each step forwards the chunk
  it most recently received. After `N − 1` steps every rank holds every chunk.
- **all-reduce = reduce-scatter, then all-gather.** Literally — `all_reduce` calls the other two.
  That identity is why ZeRO-1 and ZeRO-2 cost no more communication than plain data parallelism:
  they run the same two phases and keep the intermediate result instead of discarding it.

**Bytes, exactly.** Every rank sends one chunk per step, so per rank per collective:

    reduce-scatter:  (N − 1) × (numel / N) × itemsize   =  P × (N − 1) / N
    all-gather:      the same
    all-reduce:      2 × P × (N − 1) / N

where `P = numel × itemsize` is the size of the whole buffer. `CommLog` counts every send, and the
tests assert the counts equal these formulas to the byte.

**`numel` must divide by `N`, and these functions refuse otherwise** rather than pad silently.
Padding is the caller's decision and is made once, in `flat.py`, where it is recorded.

**What the counters cannot tell you is time.** Nothing here moves a byte across a wire. Time comes
from `timing.py`, which applies *assumed* link bandwidths to these measured byte counts.
"""

from collections import defaultdict

import torch

from .world import CallRecord, World


def _chunks(buffer: torch.Tensor, n: int) -> torch.Tensor:
    if buffer.dim() != 1:
        raise ValueError("collectives operate on flat (1-D) buffers")
    if buffer.numel() % n:
        raise ValueError(
            f"a buffer of {buffer.numel()} elements cannot be cut into {n} equal chunks. Pad it "
            "first (flat.padded_numel) — padding is recorded there, never done silently here."
        )
    return buffer.view(n, -1)


def _record(world: World, op: str, label: str, numel: int, itemsize: int, per_link: dict) -> None:
    worst: dict[str, int] = {}
    for (src, dst), nbytes in per_link.items():
        kind = world.link_kind(src, dst)
        worst[kind] = max(worst.get(kind, 0), nbytes)
    world.comm.calls.append(CallRecord(op, label, numel, itemsize, worst))


def reduce_scatter(
    world: World, buffers: list[torch.Tensor], label: str = ""
) -> list[torch.Tensor]:
    """Sum `buffers` across ranks; rank `r` receives only chunk `r` of the sum.

    The inputs are not modified: like a real reduce-scatter, each rank reads its send buffer and
    writes a separate output. The summation order of every chunk is fixed by the ring, so the result
    is deterministic and identical however the caller uses it — which is what lets the stages agree
    with each other bit for bit.

    Args:
        world: Supplies the topology and the counters.
        buffers: One flat buffer per rank, same length and dtype, length divisible by `N`.
        label: Recorded with the call.

    Returns:
        One new tensor per rank: its fully reduced chunk, `numel / N` elements.
    """
    n = world.size
    if len(buffers) != n:
        raise ValueError(f"expected {n} buffers, one per rank, got {len(buffers)}")
    chunks = [_chunks(b, n) for b in buffers]
    itemsize = buffers[0].element_size()
    per_link: dict[tuple[int, int], int] = defaultdict(int)

    # outgoing[r] is the partial sum rank r forwards at the current step.
    outgoing = [chunks[r][(r - 1) % n].clone() for r in range(n)]
    for step in range(n - 1):
        incoming: list[torch.Tensor | None] = [None] * n
        for src in range(n):
            dst = (src + 1) % n
            nbytes = outgoing[src].numel() * itemsize
            world.comm.send("reduce_scatter", src, dst, nbytes)
            per_link[(src, dst)] += nbytes
            incoming[dst] = outgoing[src]
        for rank in range(n):
            chunk = (rank - step - 2) % n
            outgoing[rank] = incoming[rank] + chunks[rank][chunk]
    _record(world, "reduce_scatter", label, buffers[0].numel(), itemsize, per_link)
    # After N − 1 steps, rank r's running sum is chunk (r − N) mod N = r, complete.
    return outgoing


def all_gather(
    world: World,
    shards: list[torch.Tensor],
    outputs: list[torch.Tensor] | None = None,
    label: str = "",
) -> list[torch.Tensor]:
    """Every rank ends with every rank's shard, concatenated in rank order.

    Args:
        world: Supplies the topology and the counters.
        shards: Rank `r`'s chunk `r`, all the same length and dtype.
        outputs: Optional full-length buffers to write into, one per rank. Omitted, new ones are
            allocated — the caller decides whether that allocation is charged to a ledger.
        label: Recorded with the call.

    Returns:
        The full buffer on each rank.
    """
    n = world.size
    if len(shards) != n:
        raise ValueError(f"expected {n} shards, one per rank, got {len(shards)}")
    size = shards[0].numel()
    itemsize = shards[0].element_size()
    if outputs is None:
        outputs = [torch.empty(size * n, dtype=shards[0].dtype) for _ in range(n)]
    views = [_chunks(out, n) for out in outputs]
    for rank in range(n):
        views[rank][rank].copy_(shards[rank])
    per_link: dict[tuple[int, int], int] = defaultdict(int)

    outgoing = [views[r][r] for r in range(n)]
    for step in range(n - 1):
        incoming: list[torch.Tensor | None] = [None] * n
        for src in range(n):
            dst = (src + 1) % n
            nbytes = outgoing[src].numel() * itemsize
            world.comm.send("all_gather", src, dst, nbytes)
            per_link[(src, dst)] += nbytes
            incoming[dst] = outgoing[src].clone()  # what arrives is a copy, as over a wire
        for rank in range(n):
            chunk = (rank - step - 1) % n
            views[rank][chunk].copy_(incoming[rank])
            outgoing[rank] = views[rank][chunk]
    _record(world, "all_gather", label, size * n, itemsize, per_link)
    return outputs


def all_reduce(world: World, buffers: list[torch.Tensor], label: str = "") -> list[torch.Tensor]:
    """Sum `buffers` across ranks, in place, every rank ending with the whole sum.

    Implemented as `reduce_scatter` followed by `all_gather` — not as a separate algorithm — so the
    sum it produces is bit-identical to the one ZeRO-2 keeps, and its byte count is exactly the two
    phases added together.
    """
    reduced = reduce_scatter(world, buffers, label=label)
    return all_gather(world, reduced, outputs=buffers, label=label)
