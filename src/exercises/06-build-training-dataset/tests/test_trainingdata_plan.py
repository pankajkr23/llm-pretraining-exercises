"""The odometer: a bijection, a disjoint partition, and a plan that can be re-derived.

Two invariants from `DECISIONS.md` are exercised here, and the distinction between them is the
point of the whole module:

- **C6, conservation** — every planned slot is covered exactly once.
- **C7, disjointness asserted on DATA, not coordinates.** Unique coordinates are arithmetic and
  prove nothing about the tokens. Two ranks can hold different coordinates that point at the same
  span, and that is the failure that would train twice while the ledger claims otherwise.
"""

import dataclasses

import pytest
from trainingdata.config import Config
from trainingdata.plan import (
    Coordinate,
    Plan,
    PlanKey,
    Span,
    build,
    build_span_table,
    decode,
    flat,
    permute,
    shard_set_hash,
)

SMALL = Config(sequence_length=8, microbatch=2, accumulation=2, ranks=2, steps=4)
SHARDS = [("shard-a", 200), ("shard-b", 160), ("shard-c", 88)]


def _all_coords(config: Config, steps: int) -> list[Coordinate]:
    """Every coordinate in the first `steps` steps.

    Args:
        config: Run shape.
        steps: How many steps to enumerate.

    Returns:
        The coordinates, in flat order.
    """
    return [
        Coordinate(step=s, rank=r, accum=a, seq=q)
        for s in range(steps)
        for r in range(config.ranks)
        for a in range(config.accumulation)
        for q in range(config.microbatch)
    ]


# --------------------------------------------------------------------------- the bijection


def test_flat_and_decode_round_trip_for_every_coordinate() -> None:
    """The odometer property. Without it, two coordinates could share an index."""
    for coord in _all_coords(SMALL, steps=6):
        assert decode(flat(coord, SMALL), SMALL) == coord


def test_flat_indices_are_dense_and_start_at_zero() -> None:
    """No gaps and no repeats — otherwise conservation cannot be checked at all."""
    indices = [flat(c, SMALL) for c in _all_coords(SMALL, steps=4)]
    assert indices == list(range(len(indices)))


def test_the_worked_example_from_the_documentation() -> None:
    """The example in the module docstring and the README, pinned.

    R=4, A=2, M=8 gives B=64; step 3, rank 2, accum 1, seq 5 -> 192+32+8+5 = 237. If the docs and
    the code ever disagree, a reader believes the docs.
    """
    config = Config(ranks=4, accumulation=2, microbatch=8)
    assert config.sequences_per_step == 64
    assert flat(Coordinate(step=3, rank=2, accum=1, seq=5), config) == 237
    assert decode(237, config) == Coordinate(step=3, rank=2, accum=1, seq=5)


@pytest.mark.parametrize(
    ("field", "value"),
    [("rank", 2), ("accum", 2), ("seq", 2), ("rank", -1), ("seq", -1)],
)
def test_a_digit_outside_its_place_is_refused(field: str, value: int) -> None:
    """A carry would alias two coordinates onto one index, and two ranks would train the same span.

    This is the sharp edge of a mixed-radix scheme, and it fails silently: nothing about the
    resulting integer looks wrong.
    """
    coord = dataclasses.replace(Coordinate(0, 0, 0, 0), **{field: value})
    with pytest.raises(ValueError, match=field):
        flat(coord, SMALL)


def test_decode_refuses_a_negative_index() -> None:
    """`divmod` on a negative would return a plausible-looking coordinate."""
    with pytest.raises(ValueError, match="non-negative"):
        decode(-1, SMALL)


# --------------------------------------------------------------------------- the span table


def test_the_span_table_is_non_overlapping_within_a_shard() -> None:
    """Spans are the unit a ledger names; two that overlap make that name ambiguous."""
    table = build_span_table(SHARDS, 8)
    by_shard: dict[str, list[Span]] = {}
    for span in table:
        by_shard.setdefault(span.shard_id, []).append(span)
    for spans in by_shard.values():
        for a, b in zip(spans, spans[1:], strict=False):
            assert not a.overlaps(b)
            assert a.end == b.start, "spans are not contiguous, so tokens are being skipped"


def test_the_ragged_tail_is_dropped_not_padded() -> None:
    """200//8 = 25, 160//8 = 20, 88//8 = 11.

    Padding would put tokens in the run that nothing put there; carrying the remainder into the next
    shard would make a span cross a shard boundary and stop being nameable as (shard, start, end).
    """
    table = build_span_table(SHARDS, 8)
    assert len(table) == 25 + 20 + 11
    assert all(s.length == 8 for s in table)


def test_a_shard_shorter_than_one_sequence_contributes_nothing() -> None:
    """It cannot supply a full window, and a partial one is not a training sample."""
    assert build_span_table([("tiny", 7)], 8) == []


def test_span_table_refuses_a_nonpositive_sequence_length() -> None:
    """Zero would divide by zero; negative would produce reversed spans."""
    with pytest.raises(ValueError, match="must be positive"):
        build_span_table(SHARDS, 0)


def test_overlaps_is_false_across_different_shards() -> None:
    """Same offsets in two shards are different tokens."""
    assert not Span("a", 0, 8).overlaps(Span("b", 0, 8))
    assert Span("a", 0, 8).overlaps(Span("a", 4, 12))


# --------------------------------------------------------------------------- the permutation


def test_the_permutation_is_a_permutation() -> None:
    """Every index exactly once. A 'shuffle' that dropped one would drop a span from the run."""
    key = PlanKey("cfg", "shards", 0)
    for n in (0, 1, 2, 57, 256):
        assert sorted(permute(n, key)) == list(range(n))


def test_the_permutation_is_derived_from_the_key_alone() -> None:
    """Same key, same order — across calls, and so across processes and machines."""
    key = PlanKey("cfg", "shards", 0)
    assert permute(64, key) == permute(64, key)


def test_a_different_key_gives_a_different_order() -> None:
    """Otherwise the key would not be recording anything."""
    base = PlanKey("cfg", "shards", 0)
    for changed in (
        dataclasses.replace(base, seed=1),
        dataclasses.replace(base, config_fingerprint="other"),
        dataclasses.replace(base, shard_set_hash="other"),
        dataclasses.replace(base, planner_version=2),
    ):
        assert permute(64, changed) != permute(64, base)
        assert changed.digest() != base.digest()


def test_the_permutation_actually_reorders() -> None:
    """A permutation that returned the identity would satisfy every test above and shuffle nothing.

    Then the run would read one shard at a time and the mixture would be whatever came next on disk.
    """
    assert permute(256, PlanKey("cfg", "shards", 0)) != list(range(256))


def test_the_shard_set_hash_is_order_sensitive() -> None:
    """Two runs over the same shards in a different order have different plans.

    Claiming the same key for both would make them look comparable when they are not.
    """
    assert shard_set_hash(["a", "b"]) != shard_set_hash(["b", "a"])
    assert shard_set_hash(["a", "b"]) == shard_set_hash(["a", "b"])


# --------------------------------------------------------------------------- the plan


def test_no_two_ranks_touch_overlapping_spans_in_a_step() -> None:
    """**C7, and the reason it is asserted on data rather than coordinates.**

    A coordinate bijection is arithmetic; it says nothing about which tokens each rank reads. This
    checks the spans themselves, which is the only version of the claim that could ever fail.
    """
    plan = build(SHARDS, SMALL)
    for step in range(SMALL.steps):
        seen: list[Span] = []
        for rank in range(SMALL.ranks):
            for accum in range(SMALL.accumulation):
                for seq in range(SMALL.microbatch):
                    seen.append(plan.span_for(Coordinate(step, rank, accum, seq)))
        for i, a in enumerate(seen):
            for b in seen[i + 1 :]:
                assert not a.overlaps(b), f"two slots in step {step} share tokens: {a} and {b}"


def test_every_span_is_used_before_any_is_reused() -> None:
    """Conservation: the run works through the corpus before it starts a second pass."""
    plan = build(SHARDS, SMALL)
    first_pass = [plan.span_for(decode(i, SMALL)) for i in range(plan.total_spans)]
    assert len({(s.shard_id, s.start) for s in first_pass}) == plan.total_spans


def test_the_pass_number_distinguishes_a_reread_from_a_first_exposure() -> None:
    """A re-read is legitimate; silently *calling* it a first exposure is not.

    The learning ledger's repeated-pass effect depends on being able to tell them apart.
    """
    plan = build(SHARDS, SMALL)
    n = plan.total_spans
    assert plan.pass_number(decode(0, SMALL)) == 1
    assert plan.pass_number(decode(n - 1, SMALL)) == 1
    assert plan.pass_number(decode(n, SMALL)) == 2
    assert plan.span_for(decode(n, SMALL)) == plan.span_for(decode(0, SMALL))


def test_the_same_inputs_build_the_same_plan() -> None:
    """The plan must be a pure function of its key. This is what replay rests on."""
    a, b = build(SHARDS, SMALL), build(SHARDS, SMALL)
    assert a.key == b.key and a.order == b.order and a.spans == b.spans


def test_changing_the_shard_set_changes_the_plan() -> None:
    """Adding a shard must not leave the plan claiming the same provenance."""
    a = build(SHARDS, SMALL)
    b = build([*SHARDS, ("shard-d", 80)], SMALL)
    assert a.key.digest() != b.key.digest()


def test_changing_the_seed_changes_the_order_but_not_the_spans() -> None:
    """A different seed is a different *order over the same data*, and should look like one."""
    a = build(SHARDS, SMALL, seed=0)
    b = build(SHARDS, SMALL, seed=1)
    assert a.spans == b.spans
    assert a.order != b.order


def test_a_plan_with_no_spans_refuses_rather_than_returning_nothing() -> None:
    """The corpus being too small is a setup error, and it should say so at the first request."""
    plan = build([("tiny", 3)], SMALL)
    assert plan.total_spans == 0
    with pytest.raises(ValueError, match="no shard was long enough"):
        plan.span_for(Coordinate(0, 0, 0, 0))


def test_the_plan_is_addressable_without_materialising_the_run() -> None:
    """`span_for` is a lookup, which is what keeps this O(1) at any corpus size.

    A plan that had to enumerate every step to answer one question would not survive 100B tokens,
    and the design is meant to be honest about that scale even though it is never run there.
    """
    plan = build(SHARDS, SMALL)
    far = Coordinate(step=10_000_000, rank=1, accum=1, seq=1)
    assert isinstance(plan.span_for(far), Span)


def test_the_plan_is_frozen() -> None:
    """A plan that changed mid-run would make every ledger entry before the change a lie."""
    plan = build(SHARDS, SMALL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.order = ()  # type: ignore[misc]
    assert isinstance(plan, Plan)
