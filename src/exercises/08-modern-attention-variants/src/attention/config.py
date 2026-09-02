"""Every dimension this exercise measures against, in one place.

The numbers are the topic's own yardstick, not ours. The reference notes work the cache
arithmetic against a 48-layer model with 8 KV heads and a head dimension of 128 in bf16, and
report 6.44 GB for one user at a 32,768-token context. Recording the configuration here rather
than inlining it means the claim
"we reproduce the reference number" is checkable: change one field and the test that reproduces
6.44 GB fails, which is the point.
"""

from dataclasses import dataclass

#: Bytes per stored number, by dtype name. The cache stores activations, not weights, so this is the
#: only place precision enters the arithmetic.
BYTES_PER_NUMBER: dict[str, int] = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1}


@dataclass(frozen=True)
class Yardstick:
    """The model Exercise 08 costs everything against.

    Attributes:
        layers: Transformer blocks. Each one holds its own K and V cache.
        kv_heads: Key/value heads. The number that separates MHA, GQA and MQA.
        query_heads: Query heads. Does not affect cache size, which is why GQA works.
        head_dim: Width of one head's key or value vector.
        dtype: Precision the cache is stored in.
        source: Where these values come from, so a reader can check them.
    """

    layers: int = 48
    kv_heads: int = 8
    query_heads: int = 8
    head_dim: int = 128
    dtype: str = "bf16"
    source: str = "the reference notes — the configuration used throughout the cache section"

    @property
    def bytes_per_number(self) -> int:
        """Storage width of one cached number."""
        return BYTES_PER_NUMBER[self.dtype]


#: The context lengths the notes and their transcript quote figures at.
YARDSTICK_CONTEXTS: tuple[int, ...] = (8_192, 32_768, 262_144, 1_000_000)
