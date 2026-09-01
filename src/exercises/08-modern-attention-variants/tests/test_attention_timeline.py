"""What the order shows, and whether it shows what the brief says it shows.

Question 2 asks what the timeline reveals that a list cannot. The brief supplies an answer —
*"first it wants exactness, then it wants memory back, then it wants length, then it wants memory
back again"* — and the temptation is to print that sentence over a chart and call it a finding.

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
    """Session 8's organising claim: attention charges twice, and position is a third thread."""
    bills = bills_addressed(MECHANISMS)
    for bill in ("compute", "cache", "position", "both"):
        assert bills.get(bill, 0) >= 3, f"only {bills.get(bill, 0)} mechanisms address {bill}"


def test_the_gaps_are_computable_and_never_negative() -> None:
    """A negative gap would mean the ordering broke; the whole timeline rests on this."""
    for before, after, days in gaps(MECHANISMS):
        assert days >= 0, f"{before} -> {after} runs backwards by {days} days"
