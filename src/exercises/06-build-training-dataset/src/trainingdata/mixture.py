"""Session 5's recipe, as data — so the fetcher and the compliance check cannot disagree.

**The problem.** Session 5 decided *how much of each kind of data*. Session 6 executes it. If the
fetcher sizes its download from one copy of those numbers and the compliance report checks against
another, the two drift and the report becomes a measurement of itself. Every number below is
declared once, here, and everything else derives from it.

**Why a corpus must be sized against the RUN, not against the ratios.** Getting the proportions
right and the total wrong does not shrink the experiment, it changes what the experiment *is*: the
run stops measuring a mixture and starts measuring repetition. The corpus on disk when this module
was written held 2,185,575 tokens against a run consuming 10,485,760 — 4.8 epochs flat, and once
shaped to these weights, **30.2 epochs of web against 0.41 of agentic**. A mixture-compliance check
over that is a check on thirty re-reads of the same text.

**And a fetcher must stop on TOKENS, never on rows or bytes.** Measured under the frozen
`s02-bpe-10000` vocabulary, bytes per token ranges from **1.98 (code) to 8.81 (indic)** — a 4.4×
spread. A fetcher that counts rows, as session 5's does, lands nowhere near the mixture it is
trying to reproduce.

**Two things the recipe asks for that this run cannot execute, stated rather than quietly dropped.**
Session 5's curriculum runs a context ladder of 4k → 8k → 16k → 32k; exercise 06 runs a flat 512, so
the ladder cannot be executed as written and the `long_context` lane — whose *only* delivery
mechanism was that ladder — stays at zero rather than being given tokens it would misrepresent.
Session 5 also retired that lane on its own evidence: 60 of its 100B was repo-packed code already
counted under `code`.
"""

from typing import Final

from . import spec
from .config import Config

#: Session 5's headline mixture, re-exported from `spec.py`.
#:
#: **Declared there, not here**, because the auditor needs them too and may not import this module
#: to get them — `verify.py` re-derives a run's mixture and compares it against the plan, and a
#: comparison against the producer's own copy of the plan would check nothing. `spec.py` is the one
#: place both sides may read: shared facts, no shared logic.
#:
#: `long_context` is deliberately **zero**. It is a schedule over the other lanes, not a corpus, and
#: a fetcher that gives it tokens is inventing a lane session 5 explicitly retired.
LANE_SHARES: Final[dict[str, float]] = spec.LANE_SHARES

#: The minimum share of every batch a lane keeps, whatever a selector would prefer.
#:
#: A floor is a **minimum, not the lane's share**: `indic` runs at 18% of which only 12 points are
#: protected, leaving 6 exposed. `agentic` sits *exactly* on its floor with zero headroom, which is
#: why any floor breach there is immediately visible rather than absorbed.
FLOORS: Final[dict[str, float]] = spec.FLOORS

#: Protected lanes may not claim more than this between them. Session 5's number.
FLOOR_CEILING: Final[float] = 0.20

#: Extra supply a protected lane is fetched with, above its planned share.
#:
#: **A floor is a minimum, and supply at exactly the minimum breaches it on any rounding.**
#: Measured: fetching `agentic` to precisely its 2.00% budget produced 1.99% of the built corpus —
#: one ten-thousandth under, and its floor read BREACHED, because that lane's floor equals its
#: share and so has no headroom at all. The scheduler enforces the floor per batch; this only
#: guarantees there is enough on disk for it to be able to.
FLOOR_HEADROOM: Final[float] = 0.05

#: Lanes a fetcher is expected to supply text for — every lane carrying budget.
FUNDED_LANES: Final[tuple[str, ...]] = tuple(
    lane for lane, share in LANE_SHARES.items() if share > 0
)


def sequence_targets(config: Config) -> dict[str, int]:
    """How many whole sequences each lane owes over one epoch of the run.

    Rounded to whole sequences and reconciled so the parts sum to the whole: no lane's share
    divides evenly into a 64-sequence step, so the mixture is exact over the **run**, never per
    step. Any compliance check must therefore assert against a cumulative total.

    Args:
        config: The run shape.

    Returns:
        Lane name to sequence count, summing exactly to the run's total.

    Raises:
        ValueError: If the shares do not sum to 1.
    """
    total_sequences = config.total_tokens // config.sequence_length
    _check_shares()

    counts = {lane: round(share * total_sequences) for lane, share in LANE_SHARES.items()}
    drift = total_sequences - sum(counts.values())
    if drift:
        # Give the rounding remainder to the largest lane: it is the one whose share moves least in
        # relative terms, and leaving the parts not summing to the whole would make every later
        # percentage quietly wrong.
        largest = max(LANE_SHARES, key=lambda lane: LANE_SHARES[lane])
        counts[largest] += drift
    return counts


def token_targets(config: Config, *, include_heldout: bool = True) -> dict[str, int]:
    """How many tokens a fetcher must supply per lane.

    Args:
        config: The run shape.
        include_heldout: Add the reserve `config.heldout_share` withholds. A fetcher that ignores
            it supplies exactly one epoch of training data and nothing to evaluate on, which reads
            as success until the split is taken out of the training tokens. Protected lanes also
            get `FLOOR_HEADROOM` on top, so their floor is satisfiable rather than knife-edge.

    Returns:
        Lane name to token count.
    """
    per_lane = sequence_targets(config)
    scale = 1.0 / (1.0 - config.heldout_share) if include_heldout else 1.0
    targets = {
        lane: int(round(count * config.sequence_length * scale)) for lane, count in per_lane.items()
    }
    for lane in FLOORS:
        targets[lane] = int(round(targets[lane] * (1.0 + FLOOR_HEADROOM)))
    return targets


def _check_shares() -> None:
    """Refuse a mixture that does not add up, or that breaches its own floors.

    Raises:
        ValueError: If the shares do not sum to 1, a floor exceeds its lane's share, or the
            protected lanes together exceed the ceiling.
    """
    total = sum(LANE_SHARES.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"lane shares sum to {total}, not 1.0")

    for lane, floor in FLOORS.items():
        share = LANE_SHARES.get(lane)
        if share is None:
            raise ValueError(f"{lane!r} has a protected floor but no share")
        if floor > share:
            raise ValueError(
                f"{lane!r} has a floor of {floor} above its share of {share} — the floor would be "
                f"breached on the first batch"
            )

    protected = sum(FLOORS.values())
    if protected > FLOOR_CEILING:
        raise ValueError(
            f"protected lanes claim {protected} of every batch, above the {FLOOR_CEILING} ceiling"
        )


def realised_shares(consumed: dict[str, int]) -> dict[str, float]:
    """What share of the tokens actually consumed each lane supplied.

    The other half of a mixture claim. `LANE_SHARES` is the plan; this is the outcome, computed
    from what the ledger recorded rather than from what was intended.

    Args:
        consumed: Lane name to tokens consumed, e.g. summed from `ConsumeEvent.lane_mix`.

    Returns:
        Lane name to share. Empty when nothing was consumed.
    """
    total = sum(consumed.values())
    if not total:
        return {}
    return {lane: count / total for lane, count in sorted(consumed.items())}


def compliance(
    consumed: dict[str, int], *, tolerance: float = spec.MIXTURE_TOLERANCE
) -> dict[str, dict]:
    """Planned versus actual share, per lane, with the floors checked.

    Args:
        consumed: Lane name to tokens consumed.
        tolerance: How far a lane may drift from its planned share before it is out of compliance.
            One point, matching the tolerance session 5 enforces on its own stage schedule.

    Returns:
        Lane name to `{planned, realised, drift, within_tolerance, floor, floor_held}`.
    """
    actual = realised_shares(consumed)
    report: dict[str, dict] = {}
    for lane, planned in sorted(LANE_SHARES.items()):
        got = actual.get(lane, 0.0)
        floor = FLOORS.get(lane)
        report[lane] = {
            "planned": planned,
            "realised": got,
            "drift": got - planned,
            "within_tolerance": abs(got - planned) <= tolerance,
            "floor": floor,
            "floor_held": True if floor is None else got >= floor,
        }
    return report
