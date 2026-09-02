"""What the order shows, and whether it shows what the brief says it shows.

Question 2 asks what the timeline reveals that a list cannot. The brief supplies an answer —
exactness, then memory, then length, then memory again — and the temptation is to print that
sentence over a chart and call it a finding.

These tests exist so the exercise cannot do that. `pressure_by_period` counts what each window
actually contains, and `Period.dominant` returns `None` on a tie rather than picking a winner. If
the neat arc were real, no window would tie. Two of them do.
"""

from attention.catalogue import load
from attention.timeline import bills_addressed, gaps, in_order, pressure_by_period

MECHANISMS = load()


def test_the_order_is_chronological_and_stable() -> None:
    """The assignment's central requirement, and the one thing the page must not get wrong."""
    ordered = in_order(MECHANISMS)
    assert [m.date for m in ordered] == sorted(m.date for m in MECHANISMS)
    assert in_order(ordered) == ordered, "the sort is not stable across a second application"


def test_two_mechanisms_sharing_a_date_do_not_reorder_between_runs() -> None:
    """Sinusoidal encoding and scaled dot-product attention share a paper and a date.

    A sort keyed on date alone would let them swap between builds, so the key includes the mechanism
    key. This pins the tie-break rather than leaving it to dictionary order.
    """
    same_day = [m.key for m in in_order(MECHANISMS) if str(m.date) == "2017-06-12"]
    assert same_day == sorted(same_day)
    assert len(same_day) >= 2, "expected the transformer paper to contribute two entries"


def test_the_timeline_opens_before_the_transformer() -> None:
    """The finding a list hides: attention is three years older than "Attention Is All You Need".

    Bahdanau introduced learned soft alignment in 2014; the 2017 paper removed the recurrence around
    it. Ordering by date makes that obvious and ordering by teaching order conceals it.
    """
    first = in_order(MECHANISMS)[0]
    assert first.date.year == 2014
    assert first.bill == "origin"


def test_learned_positions_predate_the_transformer() -> None:
    """A second thing only the date order shows.

    Learned absolute position embeddings are usually told as a Transformer-era idea. The source
    the Transformer paper itself cites for them is a month older than the Transformer.
    """
    by_key = {m.key: m for m in MECHANISMS}
    assert by_key["learned_absolute"].date < by_key["standard_attention"].date


def test_the_field_left_the_cost_alone_for_years_after_the_transformer() -> None:
    """The longest silence in the middle of the timeline, and a real answer to Question 2.

    Between the Transformer and the first serious attempt to make attention cheaper there is a gap
    of well over a year. The field spent that time using attention rather than paying for it — which
    is invisible in any list and obvious on a date axis.
    """
    by_key = {m.key: m for m in MECHANISMS}
    days = (by_key["sparse_attention"].date - by_key["standard_attention"].date).days
    assert days > 600, f"expected a long quiet stretch after the transformer, found {days} days"


def test_some_periods_have_no_single_dominant_pressure() -> None:
    """**The test that keeps the exercise honest.**

    The brief describes a clean sequence of one pressure after another. The data does not show that:
    there are windows in which the field was attacking several bills at once, and `dominant` reports
    `None` for them rather than manufacturing a winner.

    If this ever fails, the arc really did become clean — and the page's conclusion must change to
    match, rather than the test being relaxed.
    """
    periods = pressure_by_period(MECHANISMS, window=2)
    assert periods, "no periods derived; the catalogue may be empty"
    ties = [p for p in periods if p.dominant is None]
    assert ties, (
        "every window has a single dominant pressure, which would make the brief's arc exactly "
        "right — check the data before believing it"
    )


def test_every_period_accounts_for_every_mechanism_in_it() -> None:
    """The windows partition the catalogue; none is dropped between them."""
    periods = pressure_by_period(MECHANISMS, window=2)
    covered = [key for p in periods for key in p.mechanisms]
    assert sorted(covered) == sorted(m.key for m in MECHANISMS)
    for period in periods:
        assert sum(period.counts.values()) == len(period.mechanisms)


def test_both_bills_are_attacked_and_position_is_its_own_story() -> None:
    """Exercise 08's organising claim: attention charges twice, and position is a third thread."""
    bills = bills_addressed(MECHANISMS)
    for bill in ("compute", "cache", "position", "both"):
        assert bills.get(bill, 0) >= 3, f"only {bills.get(bill, 0)} mechanisms address {bill}"


def test_the_gaps_are_computable_and_never_negative() -> None:
    """A negative gap would mean the ordering broke; the whole timeline rests on this."""
    for before, after, days in gaps(MECHANISMS):
        assert days >= 0, f"{before} -> {after} runs backwards by {days} days"


def test_the_arc_verdict_answers_the_question_that_was_asked() -> None:
    """The page published "the claimed arc holds in 6 of these 7 windows" and it was wrong.

    `held` counted windows that produced *a* clear winner. The claim under test is not "does each
    window decide" — it is "do the windows decide in the order the brief predicts". Six of seven do
    decide, and the order is not the claimed one, so the honest verdict is the opposite of the
    published one. What made it convincing was that the number was derived; deriving a number does
    not make it an answer to the question you asked.
    """
    from attention.timeline import CLAIMED_ARC, arc_verdict

    v = arc_verdict(load())

    assert v.decided + v.undecided == len(v.observed)
    assert v.decided > 0, "no window decided, so there is nothing to compare against the claim"

    decided = tuple(x for x in v.observed if x)
    assert v.matches == (decided == CLAIMED_ARC), "matches must compare the ORDER, not the count"

    #: The specific failure, pinned. If a future catalogue makes the arc true this test should be
    #: rewritten to say so — but it must never pass by accident because `matches` stopped comparing
    #: sequences, which is exactly how the published claim went wrong.
    assert not v.matches, "the arc now matches; rewrite the verdict rather than loosening this"
    assert "cache" in v.never_dominates, (
        "the claim has the field returning to the cache bill twice; if some window now decides on "
        "cache alone, the verdict's central sentence is stale"
    )
    assert v.settles_on == "both" and v.settles_from is not None


def test_the_old_measurement_would_still_look_convincing() -> None:
    """The broken version, reconstructed, so the difference is visible rather than asserted.

    Counting decided windows gives a large, derived, confident number that says nothing about the
    claim. This is here because the failure was not sloppiness — it was a real count answering an
    adjacent question, which is the hardest kind of wrong number to notice.
    """
    from attention.timeline import arc_verdict, pressure_by_period

    periods = pressure_by_period(load())
    old_number = sum(1 for p in periods if p.dominant)
    v = arc_verdict(load())

    assert old_number == v.decided, "the reconstruction must match what the page used to print"
    assert old_number >= len(periods) - 1, "the old number was large, which is why it convinced"
    assert not v.matches, "and the claim it was offered as evidence for is false"


def test_the_arc_findings_are_reported_only_if_they_survive_moving_the_buckets() -> None:
    """The bucket edges are arbitrary, so every count drawn from them needs a noise floor.

    They start in 2014 because attention does, not because the field turned on that boundary. The
    page asserted its count was "not noise" and offered no evidence; re-running with the edges
    shifted one year is the cheapest test available and it immediately cost a finding — the claim
    that the field settles on both bills from 2020 does not survive. Two findings do.
    """
    from attention.timeline import arc_robustness, arc_verdict

    r = arc_robustness(load())
    assert len(r.offsets) >= 2, "one slicing is not a noise floor"
    assert len({tuple(s) for s in r.sequences}) > 1, (
        "the offsets produce identical sequences, so this measures nothing — pick edges that move"
    )

    #: The two that survive. If either ever stops surviving, the verdict's central claims are stale
    #: and must be rewritten rather than this loosened.
    assert not r.matches_anywhere, (
        "the claimed arc now matches under some slicing; rewrite the verdict"
    )
    assert r.cache_never_dominates, (
        "the cache bill now wins a window; the verdict's key sentence is stale"
    )

    #: And the one that does not. This is asserted so that if a future catalogue makes the settling
    #: robust, the test fails and someone upgrades the prose from "one reading" to a finding.
    assert r.settles_everywhere is None, (
        "the settling now survives every slicing — promote it from a reading to a finding"
    )
    assert arc_verdict(load()).settles_on is not None, (
        "the unshifted slicing no longer settles at all; the paragraph describing it is stale"
    )
