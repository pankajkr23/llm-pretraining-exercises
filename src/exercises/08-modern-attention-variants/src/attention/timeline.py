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


def pressure_by_period(
    mechanisms: list[Mechanism], window: int = 2, offset: int = 0
) -> list[Period]:
    """Group the timeline into windows and count which bill each one was paying down.

    Args:
        mechanisms: The catalogue.
        window: Years per period.
        offset: Years to shift the first bucket boundary back by. **The bucket edges are an
            arbitrary choice** — they start at 2014 because that is when attention starts, not
            because anything happened to the field on that boundary. Shifting them is how the
            noise floor of every count derived from these windows gets measured, which this repo
            requires before a comparison is quoted: re-run it under a different arbitrary choice
            and check the effect survives.

    Returns:
        Periods in chronological order, covering only years the catalogue actually spans.
    """
    ordered = in_order(mechanisms)
    if not ordered:
        return []

    first, last = ordered[0].date.year - offset, ordered[-1].date.year
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


#: The instructor's arc, in this module's own labels. "Exactness" is the compute bill (exact global
#: attention), "memory" is the cache bill, "length" is position. Written down so the comparison is
#: against a stated claim rather than a remembered one.
CLAIMED_ARC: tuple[str, ...] = ("compute", "cache", "position", "cache")


@dataclass(frozen=True)
class ArcVerdict:
    """Whether the claimed arc survives the dates.

    Attributes:
        claimed: The arc as the brief states it, in our labels.
        observed: The dominant bill of each window in order, `None` where no bill dominated.
        decided: Windows that produced a clear winner.
        undecided: Windows that came back a tie.
        matches: Whether the decided winners, in order, are the claimed arc.
        never_dominates: Bills the claim relies on that never dominate a window alone.
        settles_on: The bill every window from `settles_from` onward is dominated by, if any.
        settles_from: The first year of that run.
    """

    claimed: tuple[str, ...]
    observed: tuple[str | None, ...]
    decided: int
    undecided: int
    matches: bool
    never_dominates: tuple[str, ...]
    settles_on: str | None
    settles_from: int | None


def arc_verdict(mechanisms: list[Mechanism], offset: int = 0) -> ArcVerdict:
    """Test the claimed arc against the windows, and report what actually happened.

    **The page used to print "the claimed arc holds in 6 of these 7 windows", and that number was
    measuring the wrong thing.** It counted windows that produced *a* clear winner — not windows
    whose winner was the one the arc predicts. Six of seven windows do decide; the sequence they
    decide on is not the claimed one, and the cache bill the story has the field returning to twice
    never dominates a single window on its own. So the honest verdict is the opposite of the one
    that was published, and it was published with a derived number attached, which is what made it
    convincing.

    Returns:
        The verdict, with every field derived so the page can render it without asserting anything.
    """
    periods = pressure_by_period(mechanisms, offset=offset)
    observed = tuple(p.dominant for p in periods)
    winners = [p for p in observed if p]

    tail: list[str] = []
    for bill in reversed(observed):
        if bill is None:
            continue
        if tail and bill != tail[0]:
            break
        tail.insert(0, bill)
    settles_on = tail[0] if len(tail) > 1 else None
    settles_from = None
    if settles_on:
        run = len(tail)
        idx = [i for i, p in enumerate(observed) if p is not None][-run]
        settles_from = periods[idx].start

    return ArcVerdict(
        claimed=CLAIMED_ARC,
        observed=observed,
        decided=len(winners),
        undecided=len(observed) - len(winners),
        matches=tuple(winners) == CLAIMED_ARC,
        never_dominates=tuple(b for b in dict.fromkeys(CLAIMED_ARC) if b not in winners),
        settles_on=settles_on,
        settles_from=settles_from,
    )


@dataclass(frozen=True)
class ArcRobustness:
    """Which conclusions survive moving the bucket edges.

    Attributes:
        offsets: The offsets tested.
        sequences: The observed winner sequence at each.
        matches_anywhere: Whether the claimed arc matches under any of them.
        cache_never_dominates: Whether the cache bill fails to win a window under all of them.
        settles_everywhere: The bill every slicing settles on, or `None` if they disagree.
    """

    offsets: tuple[int, ...]
    sequences: tuple[tuple[str | None, ...], ...]
    matches_anywhere: bool
    cache_never_dominates: bool
    settles_everywhere: str | None


def arc_robustness(mechanisms: list[Mechanism], offsets: tuple[int, ...] = (0, 1)) -> ArcRobustness:
    """Re-run the whole verdict under a different arbitrary choice and report what survives.

    **The two-year buckets start in 2014 because attention does, not because anything happened to
    the field on that boundary.** That makes the edges arbitrary, and this repo's own rule is that
    an arbitrary choice must be varied before any comparison drawn from it is quoted. The page
    asserted its count was "not noise" and offered no evidence for that at all.

    Varying it matters here: shifting the edges by one year changes the fourth window's winner and
    drops a window entirely. Two conclusions survive anyway — the claimed arc matches under neither
    slicing, and the cache bill never wins a window under either — and one does not, which is why
    it must be reported as the weaker claim it is.

    Returns:
        What held across every offset, and what did not.
    """
    verdicts = [arc_verdict(mechanisms, offset=o) for o in offsets]
    settles = {v.settles_on for v in verdicts}
    return ArcRobustness(
        offsets=offsets,
        sequences=tuple(v.observed for v in verdicts),
        matches_anywhere=any(v.matches for v in verdicts),
        cache_never_dominates=all("cache" in v.never_dominates for v in verdicts),
        settles_everywhere=settles.pop() if len(settles) == 1 else None,
    )
