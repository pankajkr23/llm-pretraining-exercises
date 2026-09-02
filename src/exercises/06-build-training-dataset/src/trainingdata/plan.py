"""Which rank trains on which tokens — decided without the ranks talking to each other.

**The problem.** Four worker processes must each know which data is theirs, with no coordination,
and without ever overlapping or skipping a span. Coordination would be a synchronisation point on
every step; overlap would train twice on the same tokens while claiming otherwise; a gap would drop
data the ledger says was consumed.

**The strategy — an odometer.** Give every sequence in the entire run one number, and make that
number decodable. This is the "sample index" idea the source material's references point at
(Megatron Core's
document/sample/shuffle indices, Mosaic's deterministic ordering): *look up* which sample belongs at
a position, never shuffle a list you then have to remember.

A car odometer reads `miles = 1000·d₃ + 100·d₂ + 10·d₁ + d₀`. Same idea, different digit sizes::

    R = ranks · A = accumulation steps · M = microbatch size
    B = R × A × M                       sequences in one global batch

    flat = step·B + rank·(A·M) + accum·M + seq

Each digit is smaller than its place value — an odometer never shows "13" in the tens column — so
the mapping is a **bijection**: every `flat` decodes to exactly one coordinate and back. That round
trip is what lets rank 2 compute its own work by asking "which flats are mine?" and getting an
answer without talking to anyone.

**Why a permutation, and why a keyed one.** Reading spans in file order would train on one shard at
a time, so the mixture would be whatever shard happened to be next. The order is therefore permuted
— deterministically, from a key that records *what produced it*. A plan you cannot re-derive is a
plan you cannot audit, and a plan that changes when unrelated code changes is one you cannot
compare runs across.

This module is pure arithmetic: no torch, no I/O, no model.
"""

import hashlib
from dataclasses import dataclass

from .config import Config


@dataclass(frozen=True, slots=True)
class Coordinate:
    """Where a sequence sits in the run.

    Attributes:
        step: Optimizer step.
        rank: Worker process.
        accum: Gradient-accumulation slot within the step.
        seq: Sequence within the microbatch.
    """

    step: int
    rank: int
    accum: int
    seq: int


@dataclass(frozen=True, slots=True)
class Span:
    """A contiguous run of tokens inside one shard.

    Half-open: `[start, end)`, so adjacent spans share an endpoint and `end - start` is the length
    without an off-by-one.
    """

    shard_id: str
    start: int
    end: int

    @property
    def length(self) -> int:
        """Tokens in the span.

        Returns:
            `end - start`.
        """
        return self.end - self.start

    def overlaps(self, other: "Span") -> bool:
        """Whether two spans share any token.

        Args:
            other: The span to compare against.

        Returns:
            True when they are in the same shard and their ranges intersect.
        """
        if self.shard_id != other.shard_id:
            return False
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True, slots=True)
class PlanKey:
    """Everything that determines the plan, recorded so it can be re-derived.

    If any of these differs, the plan differs — and two runs whose plans differ are not comparable,
    however similar their loss curves look. Recording the key is what turns that from a hidden
    confound into a checkable fact.
    """

    config_fingerprint: str
    #: A digest of the shard set, in order. Adding or removing a shard changes the plan.
    shard_set_hash: str
    seed: int
    #: Bumped by hand when the planning *algorithm* changes. Without it, a code change would
    #: silently produce a different plan under an unchanged key — and the ledger would be the only
    #: evidence anything moved.
    planner_version: int = 1

    def digest(self) -> str:
        """A short stable digest of the whole key.

        Returns:
            Sixteen hex characters.
        """
        payload = (
            f"{self.planner_version}|{self.config_fingerprint}|{self.shard_set_hash}|{self.seed}"
        )
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def shard_set_hash(shard_ids: list[str]) -> str:
    """Digest a shard set.

    Order-sensitive on purpose: the span table is built in this order, so two runs over the same
    shards in a different order have genuinely different plans and must not claim the same key.

    Args:
        shard_ids: The shards, in the order the span table will use.

    Returns:
        Sixteen hex characters.
    """
    joined = "|".join(shard_ids).encode("utf-8")
    return hashlib.blake2b(joined, digest_size=8).hexdigest()


def flat(coord: Coordinate, config: Config) -> int:
    """The coordinate's position in the whole run.

    Args:
        coord: Where the sequence sits.
        config: Supplies the digit sizes.

    Returns:
        The flat index.

    Raises:
        ValueError: If any digit is out of range — which would alias two different coordinates onto
            one index and silently train twice on the same span.
    """
    _check(coord, config)
    per_rank = config.accumulation * config.microbatch
    return (
        coord.step * config.sequences_per_step
        + coord.rank * per_rank
        + coord.accum * config.microbatch
        + coord.seq
    )


def decode(index: int, config: Config) -> Coordinate:
    """The inverse of `flat`.

    Args:
        index: A flat index.
        config: Supplies the digit sizes.

    Returns:
        The coordinate it came from.

    Raises:
        ValueError: If the index is negative.
    """
    if index < 0:
        raise ValueError(f"flat index must be non-negative, got {index}")
    per_rank = config.accumulation * config.microbatch
    step, rest = divmod(index, config.sequences_per_step)
    rank, rest = divmod(rest, per_rank)
    accum, seq = divmod(rest, config.microbatch)
    return Coordinate(step=step, rank=rank, accum=accum, seq=seq)


def _check(coord: Coordinate, config: Config) -> None:
    """Refuse a coordinate whose digits do not fit their places.

    Args:
        coord: The coordinate.
        config: Supplies the bounds.

    Raises:
        ValueError: If any component is out of range.
    """
    for name, value, limit in (
        ("rank", coord.rank, config.ranks),
        ("accum", coord.accum, config.accumulation),
        ("seq", coord.seq, config.microbatch),
    ):
        if not 0 <= value < limit:
            raise ValueError(
                f"{name}={value} is outside [0, {limit}) — the odometer would carry into the next "
                f"digit and two different coordinates would share a flat index"
            )
    if coord.step < 0:
        raise ValueError(f"step={coord.step} is negative")


def build_span_table(shards: list[tuple[str, int]], sequence_length: int) -> list[Span]:
    """Cut every shard into non-overlapping spans of exactly `sequence_length` tokens.

    A shard's tail is **dropped** when it is shorter than a full sequence. Padding it instead would
    put tokens into the run that nothing put there, and every count downstream would inherit them;
    carrying the remainder into the next shard would make a span cross a shard boundary, so it could
    no longer be named by `(shard_id, start, end)`.

    Args:
        shards: `(shard_id, token_count)` in the order the plan will use.
        sequence_length: Tokens per span.

    Returns:
        Every span, shard by shard, in order.

    Raises:
        ValueError: If `sequence_length` is not positive.
    """
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    table: list[Span] = []
    for shard_id, count in shards:
        whole = count // sequence_length
        for i in range(whole):
            table.append(Span(shard_id, i * sequence_length, (i + 1) * sequence_length))
    return table


def permute(n: int, key: PlanKey) -> list[int]:
    """A deterministic permutation of `range(n)`, derived from the key alone.

    Keyed rather than seeded from a global RNG, and derived per index rather than by shuffling in
    place: the result depends on nothing but `(key, n)`, so it survives a process restart, a
    different Python version's shuffle implementation, and a different machine.

    Args:
        n: How many indices.
        key: What the permutation is derived from.

    Returns:
        Every index in `range(n)`, exactly once, in permuted order.
    """
    if n <= 0:
        return []
    digest = key.digest().encode("utf-8")
    # Sort by a keyed hash of each index. O(n log n), no RNG state to carry, and reproducible from
    # the key alone -- which is the property that matters more than speed here.
    return sorted(
        range(n),
        key=lambda i: hashlib.blake2b(digest + i.to_bytes(8, "big"), digest_size=16).digest(),
    )


@dataclass(frozen=True, slots=True)
class Plan:
    """The whole run's data order, addressable by coordinate.

    Holds a span table and a permutation of it. Nothing is materialised per step: `span_for` is a
    lookup, which is what keeps this O(1) at any corpus size.
    """

    key: PlanKey
    spans: tuple[Span, ...]
    order: tuple[int, ...]
    config: Config

    @property
    def total_spans(self) -> int:
        """How many sequences the plan can serve.

        Returns:
            The span count.
        """
        return len(self.spans)

    def span_for(self, coord: Coordinate) -> Span:
        """The span a coordinate should train on.

        Wraps when the run is longer than the corpus, which is how a second epoch happens. The
        wrap is deliberate and visible rather than an error, but `pass_number` exists so a consumer
        can tell a re-read from a first exposure.

        Args:
            coord: Where the sequence sits.

        Returns:
            The span.

        Raises:
            ValueError: If the plan has no spans at all.
        """
        if not self.spans:
            raise ValueError("the plan has no spans: no shard was long enough for one sequence")
        return self.spans[self.order[flat(coord, self.config) % self.total_spans]]

    def pass_number(self, coord: Coordinate) -> int:
        """How many times this span has been seen before, plus one.

        Args:
            coord: Where the sequence sits.

        Returns:
            1 on first exposure, 2 on the first re-read, and so on.
        """
        return flat(coord, self.config) // self.total_spans + 1


def build(shards: list[tuple[str, int]], config: Config, *, seed: int | None = None) -> Plan:
    """Compile a plan from a shard set.

    Args:
        shards: `(shard_id, token_count)` in the order to use.
        config: Run shape.
        seed: Overrides `config.seed` when given.

    Returns:
        The plan.
    """
    spans = build_span_table(shards, config.sequence_length)
    key = PlanKey(
        config_fingerprint=config.fingerprint(),
        shard_set_hash=shard_set_hash([sid for sid, _ in shards]),
        seed=config.seed if seed is None else seed,
    )
    return Plan(key=key, spans=tuple(spans), order=tuple(permute(len(spans), key)), config=config)
