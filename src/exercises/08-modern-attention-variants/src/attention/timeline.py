"""What falls out of the order, computed rather than asserted.

Question 2 of the assignment asks: *"What does the timeline actually show? Write what you saw once
the mechanisms were in date order that you could not see as a list."* The instructor's own answer,
in the brief, is a claimed arc:

    "first it wants exactness, then it wants memory back, then it wants length, then it wants
     memory back again"

and in `s8.md`, longer: exact global attention -> cheaper decoding memory -> better position
handling -> longer contexts -> recurrent state returning -> sparsity returning -> compression
becoming more aggressive.

**That claim is testable, so this module tests it rather than repeating it.** `pressure_by_period`
groups the catalogue into windows and reports which bill each window is dominated by. If the arc is
real it appears in the counts; if it does not appear, this exercise says so. An arc quoted from the
brief and printed over a chart that does not show it would be the same failure as an unsourced date.
"""

from collections import Counter
from dataclasses import dataclass

from attention.catalogue import Mechanism


@dataclass(frozen=True)
class Period:
    """One window of the timeline, and what the field was buying in it.

    Attributes:
        start: First year in the window, inclusive.
        end: Last year, inclusive.
        counts: How many mechanisms in the window addressed each bill.
        mechanisms: Their keys, in date order.
    """

    start: int
    end: int
    counts: dict[str, int]
    mechanisms: tuple[str, ...]

    @property
    def dominant(self) -> str | None:
        """The bill most mechanisms in this window addressed, or None on a tie or an empty window.

        Returns None rather than picking a winner, because a tie is a real answer about a period in
        which the field was doing two things at once — and quietly breaking it would manufacture an
        arc the data does not show.
        """
        if not self.counts:
            return None
        ranked = Counter(self.counts).most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]


def in_order(mechanisms: list[Mechanism]) -> list[Mechanism]:
    """The catalogue by date, oldest first — the order the assignment requires.

    Ties break on key so the order is stable across runs; two mechanisms genuinely sharing a date
    (sinusoidal and standard attention share a paper) must not reorder between builds.
    """
    return sorted(mechanisms, key=lambda m: (m.date, m.key))


def pressure_by_period(mechanisms: list[Mechanism], window: int = 2) -> list[Period]:
    """Group the timeline into windows and count which bill each one was paying down.

    Args:
        mechanisms: The catalogue.
        window: Years per period.

    Returns:
        Periods in chronological order, covering only years the catalogue actually spans.
    """
    ordered = in_order(mechanisms)
    if not ordered:
        return []

    first, last = ordered[0].date.year, ordered[-1].date.year
    periods: list[Period] = []
    for start in range(first, last + 1, window):
        end = start + window - 1
        inside = [m for m in ordered if start <= m.date.year <= end]
        if not inside:
            continue
        periods.append(
            Period(
                start=start,
                end=end,
                counts=dict(Counter(m.bill for m in inside)),
                mechanisms=tuple(m.key for m in inside),
            )
        )
    return periods


def gaps(mechanisms: list[Mechanism]) -> list[tuple[str, str, int]]:
    """The quiet stretches, as `(before, after, days)` triples.

    A long gap is as informative as a cluster: it says the field was satisfied, or busy elsewhere.
    A list cannot show this at all, which is part of the answer to Question 2.
    """
    ordered = in_order(mechanisms)
    return [
        (a.key, b.key, (b.date - a.date).days) for a, b in zip(ordered, ordered[1:], strict=False)
    ]


def bills_addressed(mechanisms: list[Mechanism]) -> dict[str, int]:
    """How many mechanisms addressed each bill, over the whole catalogue."""
    return dict(Counter(m.bill for m in mechanisms))
