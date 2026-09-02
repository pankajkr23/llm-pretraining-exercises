"""The feature's six chapters, and the rule that every mechanism belongs to exactly one.

The page tells the chronology as six *wells* — a well being a magazine feature's chapter. The
grouping is editorial and it is a claim, so it lives here as tracked data rather than as prose
inside the page's JavaScript, and :func:`check` refuses a partition that does not cover the
catalogue exactly once.

Why that guard is load-bearing: the wells are how a reader navigates the whole catalogue, and
the two ways this can rot are both silent. A mechanism named in no well simply stops being shown,
and the page still renders one entry short with no gap where the missing one was. A mechanism
named in two wells is told twice, which reads as an editing mistake rather than as the data error
it is. Neither shows up as a broken page, so neither would be found by looking at one.

What the guard does NOT check is whether a well's headline is true of its members, and that is
where this file has been wrong. Well VI is headed "keep a fixed-size state" and its standfirst
promised "every one of them pays in the same single way" -- while it held NSA and DeepSeek's
compressed sparse attention, both of which build a score grid and keep a KV cache. They select
from the cache; they do not replace it. They now sit in Well III with the other entries that
attack a bill without abandoning the grid, which leaves Well VI as exactly the STATE family --
the same eight the glossary counts as refusing to build a grid at all. The chapter and the shape
are now one object, so a future disagreement between them is visible rather than editorial.

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
        subject: What the chapter is ABOUT, in three or four plain words. The hooks below are the
            chapter titles and they are deliberately oblique -- "Two bills, two crowds." is a good
            hook and tells a reader scanning the longest section on the page nothing about what is
            in it. The subject rides beside the numeral so the rail, the kicker and a returning
            reader all have something to navigate by.
        headline: The chapter's problem, stated in plain words rather than named as a technique.
        standfirst: One sentence of orientation, set under the headline.
        keys: The mechanisms in this chapter, in the order the chapter tells them.
    """

    numeral: str
    subject: str
    headline: str
    standfirst: str
    keys: tuple[str, ...]


WELLS: tuple[Well, ...] = (
    Well(
        numeral="I",
        subject="Attention before the Transformer",
        headline="By the end of the sentence it had forgotten the beginning.",
        standfirst=(
            "One fixed vector had to carry a whole source sentence. Attention's first form let "
            "the decoder look at every input position directly -- and it arrives three years "
            "before the architecture everyone now associates it with."
        ),
        keys=("bahdanau_attention",),
    ),
    Well(
        numeral="II",
        subject="The Transformer, and the two bills it opens",
        headline="The hardware was parallel and the model was not.",
        standfirst=(
            "Dropping recurrence bought parallel training and cost the model any idea of order. "
            "One paper ships the fix, the architecture, and both of the bills this page is about."
        ),
        keys=("learned_absolute", "standard_attention", "sinusoidal"),
    ),
    Well(
        numeral="III",
        subject="Compute and cache split the field",
        headline="Two bills, two crowds.",
        standfirst=(
            "The compute bill and the cache bill were attacked by different people for different "
            "reasons, and a date-ordered list interleaves them into apparent nonsense. Read as "
            "two crowds, the entries here are two arguments running in parallel -- until the last "
            "two, which stop choosing and go after both at once."
        ),
        keys=(
            "sparse_attention",
            "topk_attention",
            "reformer",
            "sliding_window",
            "mqa",
            "gqa",
            "mla",
            "msa",
            "nsa",
            "deepseek_csa",
        ),
    ),
    Well(
        numeral="IV",
        subject="Rotary embeddings, and the three repairs",
        headline="We shipped a position scheme in 2021 and we are still arguing about it.",
        standfirst=(
            "The worked example above leaves one thing out on purpose: position. Both copies of "
            '"the" get identical vectors there, and this chapter is the gap that opens. Rotary '
            "embeddings solved relative distance elegantly and left one bomb: run past the trained "
            "length and the rotation keeps going. Three repairs follow. Then one paper concludes "
            "the answer is to delete positional embeddings entirely -- and the next one concludes "
            "the answer is to make them richer. Both cannot be right."
        ),
        keys=("rope", "alibi", "ntk_aware", "yarn", "drope", "hd_rope"),
    ),
    Well(
        numeral="V",
        subject="Two discoveries, not two optimisations",
        headline="Two things we were wrong about.",
        standfirst=(
            "Neither of these is an optimisation. Both are discoveries about what was already "
            "happening \u2014 one about where the cost actually was, one about what models had "
            "quietly been doing with the first few tokens all along. The first is FlashAttention. "
            "Everyone had assumed attention was slow because of the arithmetic; it was actually "
            "slow because of shuttling the score grid out to memory and back, so reordering the "
            "same maths to keep the grid on chip made it several times faster with a bit-for-bit "
            "identical result. Nothing was approximated \u2014 the bill had simply been misread."
        ),
        keys=("flashattention", "attention_sinks"),
    ),
    Well(
        numeral="VI",
        subject="Throw the cache away, keep a fixed-size state",
        headline="Then stop keeping everything.",
        standfirst=(
            "If the cache is the bill, refuse to hold a cache. Fold the past into a fixed-size "
            "state instead. Each entry here fixes the last one's way of forgetting -- and every "
            "one of them pays in the same single way."
        ),
        keys=(
            "linear_attention",
            "delta_rule",
            "mamba",
            "deltanet_parallel",
            "gated_deltanet",
            "kda",
            "mamba3",
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
