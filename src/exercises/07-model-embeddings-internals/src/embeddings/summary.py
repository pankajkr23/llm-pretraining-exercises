"""Paired-seed statistics: the arithmetic that turns ten losses into one defensible claim.

**Why pairing, in one number.** Across seeds this exercise's control moves **0.469 nats**, which is
larger than every effect it set out to measure. Compare unpaired means and you are reading the seed,
not the architecture. Within a seed both arms share an initialisation and a data order, so taking
the difference first cancels that: the recorded paired standard deviation is **0.024**, twenty times
smaller than the spread it removes.

So every claim here is stated as a **gap**, never as a height, and the summary reports the seeds
that agreed alongside the mean — because five seeds agreeing at a small gap is stronger evidence
than one seed showing a large one.

**No torch.** This is `statistics` over a list of floats, so it runs in the plain CI job, and the
numbers a reader most wants to check are the ones that need nothing installed.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

#: Below this many nats a gap is reported as `inconclusive` however tidy the seeds look. It is the
#: resolution this setup has earned, not a significance threshold: `pairing.paired_sd` is 0.024, and
#: claiming a difference smaller than the noise that survives pairing would be false precision.
RESOLUTION_NATS = 0.03


@dataclass(frozen=True)
class Paired:
    """One arm measured against another, seed by seed.

    Attributes:
        gap: Mean of `treatment - control`. Negative means the treatment is better, matching the
            sign convention in `results/measurements.json`.
        sd: Sample standard deviation of the per-seed gaps.
        t: `gap / (sd / sqrt(n))`. Reported, not thresholded — with five seeds it is a description
            of consistency rather than a hypothesis test anybody should lean on.
        seeds: How many seeds were compared.
        agreeing: How many of them moved in the direction of `gap`.
        sign_p: One-sided sign test, `0.5 ** seeds` when every seed agrees and `None` otherwise —
            spelled out rather than approximated, because the only case this run can reach is
            unanimity or nothing.
        per_seed: The individual gaps, kept so a reader can see the spread rather than trust `sd`.
    """

    gap: float
    sd: float
    t: float
    seeds: int
    agreeing: int
    sign_p: float | None
    per_seed: tuple[float, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> str:
        """`supported`, `refuted` or `inconclusive`, by the rule stated in the module docstring.

        `supported` requires every seed to agree AND the gap to clear `RESOLUTION_NATS`. A unanimous
        but tiny gap is `inconclusive`, which is the honest reading: this setup cannot resolve it.
        """
        if abs(self.gap) < RESOLUTION_NATS or self.agreeing != self.seeds:
            return "inconclusive"
        return "supported" if self.gap < 0 else "refuted"

    def as_dict(self) -> dict[str, object]:
        """A JSON-encodable view. Every field is a float, an int or None, deliberately.

        Three experiments in exercise 05 trained to completion and died writing their results,
        because the bundle carried an object `json` cannot encode. Nothing here can carry one.
        """
        return {
            "gap": self.gap,
            "sd": self.sd,
            "t": self.t,
            "seeds": self.seeds,
            "agreeing": self.agreeing,
            "sign_p": self.sign_p,
            "per_seed": list(self.per_seed),
            "verdict": self.verdict,
        }


def paired(control: Sequence[float], treatment: Sequence[float]) -> Paired:
    """Compare two arms seed by seed.

    Args:
        control: The reference arm's loss, one per seed.
        treatment: The other arm's loss, in the SAME seed order.

    Returns:
        The paired summary.

    Raises:
        ValueError: When the two differ in length, or fewer than two seeds are given. Both would
            otherwise produce a number: `zip` truncates silently, and a standard deviation over one
            sample is undefined rather than zero.
    """
    if len(control) != len(treatment):
        raise ValueError(
            f"paired comparison needs the same seeds on both sides, got {len(control)} and "
            f"{len(treatment)}. zip would silently truncate to the shorter one."
        )
    if len(control) < 2:
        raise ValueError("a paired comparison needs at least two seeds")

    gaps = [t - c for c, t in zip(control, treatment, strict=True)]
    gap = statistics.fmean(gaps)
    sd = statistics.stdev(gaps)
    direction = -1 if gap < 0 else 1
    agreeing = sum(1 for g in gaps if (g < 0) == (direction < 0))
    n = len(gaps)
    return Paired(
        gap=gap,
        sd=sd,
        t=gap / (sd / math.sqrt(n)) if sd else math.inf,
        seeds=n,
        agreeing=agreeing,
        sign_p=0.5**n if agreeing == n else None,
        per_seed=tuple(gaps),
    )


def unpaired_spread(losses: Sequence[float]) -> float:
    """How far one arm moves across seeds — the noise pairing exists to remove.

    Print it beside any gap. A gap smaller than this is not thereby wrong, but a reader who is not
    shown both will read an unpaired comparison into a paired one.
    """
    if len(losses) < 2:
        raise ValueError("a spread needs at least two seeds")
    return max(losses) - min(losses)
