"""The two bills attention sends, as arithmetic rather than as assertion.

Session 8 is organised around one idea: attention charges you twice, and every mechanism after the
original is somebody paying down one of the two bills.

    compute   grows with T^2   -- every token scores against every other token
    KV cache  grows with T     -- every token's key and value are kept for the next one

Both are closed forms. Nothing here is estimated, sampled or trained, which is why this module has
no torch and no randomness: given the configuration, the answer is the answer.

**Read the formula, not the headline.** The session quotes 6.44 GB for one user at 32K and 51.54 GB
for eight. `kv_cache_bytes` reproduces both exactly, and a test pins them -- so if someone edits the
yardstick, the documents that quote those numbers break rather than drift.
"""

from dataclasses import dataclass

from attention.config import Yardstick


def kv_cache_bytes(
    yardstick: Yardstick,
    context: int,
    batch: int = 1,
    kv_heads: int | None = None,
) -> int:
    """Bytes of key/value cache held for a whole model at a given context and batch.

    The factor of two is K and V; there is one of each per layer per KV head per position.

        2 x layers x kv_heads x head_dim x context x batch x bytes_per_number

    Args:
        yardstick: The model being costed.
        context: Tokens held in the cache.
        batch: Concurrent sequences (the session calls these "users").
        kv_heads: Override the yardstick's KV-head count, to price MHA against GQA against MQA
            without inventing a second model. Defaults to the yardstick's own.

    Returns:
        Exact byte count.
    """
    heads = yardstick.kv_heads if kv_heads is None else kv_heads
    return (
        2
        * yardstick.layers
        * heads
        * yardstick.head_dim
        * context
        * batch
        * yardstick.bytes_per_number
    )


def attention_scores(context: int, heads: int = 1, layers: int = 1) -> int:
    """How many query-key scores a full forward pass computes.

    The `T^2` bill. Quadratic in context, which is the cost every sparse, windowed and linear
    variant exists to avoid.
    """
    return context * context * heads * layers


@dataclass(frozen=True)
class Sharing:
    """One point on the MHA -> GQA -> MQA line.

    Attributes:
        name: What the arrangement is called.
        kv_heads: Key/value heads it keeps.
        note: What it costs, in one clause.
    """

    name: str
    kv_heads: int
    note: str


def sharing_ladder(yardstick: Yardstick) -> tuple[Sharing, ...]:
    """The three arrangements Session 8 compares, at that session's own head count.

    MQA is one KV head by definition. GQA is a free parameter; the session uses two, which is where
    its "one quarter of the cache" figure comes from.
    """
    return (
        Sharing("MHA", yardstick.query_heads, "one KV head per query head; the largest cache"),
        Sharing("GQA", 2, "query heads share KV heads in groups; the practical default"),
        Sharing("MQA", 1, "every query head reads one KV head; the smallest cache, most sharing"),
    )


def compressed_positions(context: int, block: int) -> int:
    """Positions still stored after compressing every `block` tokens into one summary.

    The other lever. GQA divides the cache by *heads*; sequence compression divides it by
    *positions*, so the two multiply rather than compete.
    """
    if block < 1:
        raise ValueError("block size must be at least 1")
    return -(-context // block)  # ceil, so a partial trailing block still costs a position
