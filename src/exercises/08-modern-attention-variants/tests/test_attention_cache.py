"""The session's own arithmetic, reproduced rather than quoted.

`s8.md` states its cache figures as results: 6.44 GB for one user at 32,768 tokens, 51.54 GB for
eight, and a quarter of the cache for GQA at two KV heads. Those are the numbers this exercise
builds an argument on, so it recomputes them from the formula instead of copying them into prose.

The difference matters. A quoted number goes stale silently when someone edits the yardstick; a
recomputed one takes this file red with it.
"""

from attention.cache import (
    attention_scores,
    compressed_positions,
    kv_cache_bytes,
    sharing_ladder,
)
from attention.config import Yardstick

GB = 1e9
SESSION = Yardstick()


def test_the_yardstick_is_the_one_the_session_used() -> None:
    """Pinned because every figure below is only "the session's number" for this configuration."""
    assert (SESSION.layers, SESSION.kv_heads, SESSION.head_dim, SESSION.dtype) == (
        48,
        8,
        128,
        "bf16",
    )
    assert SESSION.bytes_per_number == 2


def test_one_user_at_32k_costs_what_the_session_says() -> None:
    """`s8.md`: one user at a 32,768-token context is 6.44 GB of KV cache."""
    got = kv_cache_bytes(SESSION, context=32_768, batch=1)
    assert got == 6_442_450_944
    assert round(got / GB, 2) == 6.44


def test_eight_users_at_32k_costs_what_the_session_says() -> None:
    """And eight of them is 51.54 GB — the figure that makes the point about serving."""
    got = kv_cache_bytes(SESSION, context=32_768, batch=8)
    assert round(got / GB, 2) == 51.54
    assert got == 8 * kv_cache_bytes(SESSION, context=32_768, batch=1)


def test_the_cache_grows_linearly_in_context_and_batch() -> None:
    """The claim the whole session rests on: linear in T, so it never stops growing."""
    base = kv_cache_bytes(SESSION, context=8_192)
    assert kv_cache_bytes(SESSION, context=16_384) == 2 * base
    assert kv_cache_bytes(SESSION, context=8_192, batch=4) == 4 * base


def test_gqa_at_two_heads_is_exactly_a_quarter_of_mha() -> None:
    """`s8.md`'s headline sharing figure, and it is exact rather than approximate."""
    mha = kv_cache_bytes(SESSION, context=32_768, kv_heads=8)
    gqa = kv_cache_bytes(SESSION, context=32_768, kv_heads=2)
    mqa = kv_cache_bytes(SESSION, context=32_768, kv_heads=1)
    assert mha == 4 * gqa
    assert mha == 8 * mqa


def test_the_sharing_ladder_only_moves_kv_heads() -> None:
    """MQA is one KV head *by definition*; the ladder must not invent a second variable."""
    ladder = {s.name: s for s in sharing_ladder(SESSION)}
    assert ladder["MQA"].kv_heads == 1
    assert ladder["MHA"].kv_heads == SESSION.query_heads
    assert all(s.note for s in ladder.values()), "a rung with no stated cost is half the story"


def test_scores_grow_quadratically() -> None:
    """The other bill. The session's own numbers: 6x6=36, 600x600=360,000."""
    assert attention_scores(6) == 36
    assert attention_scores(600) == 360_000
    assert attention_scores(10_000) == 100_000_000
    assert attention_scores(2_000) == 4 * attention_scores(1_000)


def test_compression_divides_positions_and_rounds_up() -> None:
    """A partial trailing block still costs a stored position; flooring would undercount."""
    assert compressed_positions(1_024, block=4) == 256
    assert compressed_positions(10, block=4) == 3
    assert compressed_positions(1_000, block=1) == 1_000


def test_sharing_and_compression_multiply_rather_than_compete() -> None:
    """Why the two levers are worth having together: they divide different factors.

    GQA divides by *heads*; compression divides by *positions*. Applying both gives the product of
    the two savings, which is the argument for DeepSeek-style designs stacking them.
    """
    full = kv_cache_bytes(SESSION, context=32_768, kv_heads=8)
    both = kv_cache_bytes(SESSION, context=compressed_positions(32_768, block=4), kv_heads=2)
    assert full == 16 * both  # 4x from heads, 4x from positions
