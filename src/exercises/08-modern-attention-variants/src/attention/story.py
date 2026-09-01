"""The feature's six chapters, and the rule that every mechanism belongs to exactly one.

The page tells the chronology as six *wells* — a well being a magazine feature's chapter. The
grouping is editorial and it is a claim, so it lives here as tracked data rather than as prose
inside the page's JavaScript, and :func:`check` refuses a partition that does not cover the
catalogue exactly once.

Why that guard is load-bearing: the wells are how a reader navigates twenty-three mechanisms,
and the two ways this can rot are both silent. A mechanism named in no well simply stops being
shown, and the page still renders twenty-two entries with no gap where the missing one was. A
mechanism named in two wells is told twice, which reads as an editing mistake rather than as the
data error it is. Neither shows up as a broken page, so neither would be found by looking at one.

The ordering claim is separate and stronger: a well is a contiguous run of the date-ordered
catalogue only where the storyline says it is, and Wells III to VI deliberately interleave --
that interleaving is the finding those chapters exist to show. So :func:`check` asserts coverage,
not contiguity, and :func:`span` reports each well's real first and last date rather than assuming
the two are adjacent in the catalogue.
"""

from dataclasses import dataclass
from datetime import date

from attention.catalogue import Mechanism


@dataclass(frozen=True)
class Well:
    """One chapter of the feature.

    Attributes:
        numeral: The Roman numeral the page sets as the chapter's kicker.
        headline: The chapter's problem, stated in plain words rather than named as a technique.
        standfirst: One sentence of orientation, set under the headline.
        pull_quote: A line lifted from the catalogue's own text, set large. Not authored here --
            every one of these is a phrase that already appears in a mechanism's fields, so the
            page's loudest typography is quoting its own evidence.
        keys: The mechanisms in this chapter, in the order the chapter tells them.
    """

    numeral: str
    headline: str
    standfirst: str
    pull_quote: str
    keys: tuple[str, ...]


WELLS: tuple[Well, ...] = (
    Well(
        numeral="I",
        headline="By the end of the sentence it had forgotten the beginning.",
        standfirst=(
            "One fixed vector had to carry a whole source sentence. Attention's first form let "
            "the decoder look at every input position directly -- and it arrives three years "
            "before the architecture everyone now associates it with."
        ),
        pull_quote=(
            "Attention existed for three years before anyone removed the recurrence around it."
        ),
        keys=("bahdanau_attention",),
    ),
    Well(
        numeral="II",
        headline="The hardware was parallel and the model was not.",
        standfirst=(
            "Dropping recurrence bought parallel training and cost the model any idea of order. "
            "One paper ships the fix, the architecture, and both of the bills this page is about."
        ),
        pull_quote=(
            "Everything after this on the timeline is somebody paying less of one of those two."
        ),
        keys=("learned_absolute", "standard_attention", "sinusoidal"),
    ),
    Well(
        numeral="III",
        headline="Two bills, two crowds.",
        standfirst=(
            "The compute bill and the cache bill were attacked by different people for different "
            "reasons, and a date-ordered list interleaves them into apparent nonsense. Read as "
            "two crowds, the entries here are two arguments running in parallel."
        ),
        pull_quote="It moves along the same line rather than leaving it.",
        keys=(
            "sparse_attention",
            "topk_attention",
            "reformer",
            "sliding_window",
            "mqa",
            "gqa",
            "mla",
            "msa",
        ),
    ),
    Well(
        numeral="IV",
        headline="We shipped a position scheme in 2021 and we are still arguing about it.",
        standfirst=(
            "Rotary embeddings solved relative distance elegantly and left one bomb: run past the "
            "trained length and the rotation keeps going. Three repairs follow. Then one paper "
            "concludes the answer is to delete positional embeddings entirely -- and the next one "
            "concludes the answer is to make them richer. Both cannot be right."
        ),
        pull_quote="Stop repairing it and remove it.",
        keys=("rope", "alibi", "ntk_aware", "yarn", "drope", "hd_rope"),
    ),
    Well(
        numeral="V",
        headline="Two things we were wrong about.",
        standfirst=(
            "Neither of these is an optimisation. Both are discoveries about what was already "
            "happening -- one about where the cost actually was, one about what models had "
            "quietly been doing with the first few tokens all along."
        ),
        pull_quote="Nothing mathematically - which is why it is on this list as the exception.",
        keys=("flashattention", "attention_sinks"),
    ),
    Well(
        numeral="VI",
        headline="Then stop keeping everything.",
        standfirst=(
            "If the cache is the bill, refuse to hold a cache. Fold the past into a fixed-size "
            "state instead. Four generations of that idea are here, each fixing the last one's "
            "way of forgetting -- and every one of them pays in the same single way."
        ),
        pull_quote="The state is a lossy summary, and what it lost is not recoverable.",
        keys=(
            "linear_attention",
            "delta_rule",
            "mamba",
            "deltanet_parallel",
            "gated_deltanet",
            "kda",
            "nsa",
            "mamba3",
            "deepseek_csa",
            "gated_deltanet2",
        ),
    ),
)


def check(mechanisms: list[Mechanism]) -> None:
    """Raise unless the wells partition the catalogue exactly.

    Args:
        mechanisms: The loaded catalogue.

    Raises:
        ValueError: If any mechanism belongs to no well or to more than one, or if a well names
            a key the catalogue does not contain.
    """
    catalogue = [m.key for m in mechanisms]
    assigned: list[str] = [k for w in WELLS for k in w.keys]

    unknown = sorted(set(assigned) - set(catalogue))
    if unknown:
        raise ValueError(f"wells name mechanisms that are not in the catalogue: {unknown}")

    twice = sorted({k for k in assigned if assigned.count(k) > 1})
    if twice:
        raise ValueError(f"mechanisms assigned to more than one well: {twice}")

    orphaned = sorted(set(catalogue) - set(assigned))
    if orphaned:
        raise ValueError(f"mechanisms in no well, so shown nowhere on the page: {orphaned}")


def well_of(key: str) -> Well:
    """The well a mechanism belongs to.

    Args:
        key: A catalogue key.

    Returns:
        The well naming it.

    Raises:
        KeyError: If no well names it.
    """
    for well in WELLS:
        if key in well.keys:
            return well
    raise KeyError(key)


def span(well: Well, mechanisms: list[Mechanism]) -> tuple[date, date]:
    """The first and last date a well covers.

    Computed from the mechanisms rather than written down, because a well is not necessarily a
    contiguous run of the catalogue -- Wells III to VI overlap in time on purpose.

    Args:
        well: The chapter.
        mechanisms: The loaded catalogue.

    Returns:
        ``(earliest, latest)`` over the well's own mechanisms.
    """
    by_key = {m.key: m for m in mechanisms}
    dates = sorted(by_key[k].date for k in well.keys)
    return dates[0], dates[-1]
