"""Scoring: per-language fertility, cross-language spread, and the assignment's final score.

The graded denominator is the **faithful unit** — one contiguous Unicode letter/mark/number run,
or one visible non-space punctuation/symbol character. It is the countable atom of the
faithfulness rule (every visible non-whitespace character must survive ``decode(encode(x))``, so
every visible character is counted). Whitespace is excluded from both the rule and the count:
tokenizers legitimately transform it (Metaspace ``▁``, byte-level ``Ġ``), and counting it would
invite denominator padding.

Fertility can land **below 1.0** under this denominator, because BPE merges frequent punctuation
runs (``](``, ``|-``) into single tokens — one token covering two or three units.

:func:`count_words` (whitespace split) is retained and reported alongside, but nothing is scored
on it. It exists so the report can show what the denominator choice is worth: the same tokenizer
measured both ways.
"""

import math
from dataclasses import dataclass

import regex

# One contiguous letter/mark/number run, or one visible non-space punctuation/symbol character.
# ``\p{M}`` is load-bearing: it keeps Devanagari/Telugu combining marks attached to their base
# character, so ``भारत`` counts as one unit rather than fragmenting at every matra.
_UNIT = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

# Fertility above which Hindi counts as degraded; calibrated to the faithful-unit denominator.
HINDI_THRESHOLD = 1.2


def count_units(text: str) -> int:
    """Number of faithful units in ``text`` — the graded denominator."""
    return len(_UNIT.findall(text))


def count_words(text: str) -> int:
    """Number of whitespace-separated words in ``text`` — reported for contrast, never scored.

    Whitespace splitting is script-agnostic: a word-character regex wrongly splits Indic words at
    combining vowel marks. This is the denominator our earlier experiments used; on wiki-faithful
    Markdown it counts roughly a quarter as many atoms as :func:`count_units`, which is why
    numbers denominated in words and in units are not comparable.
    """
    return len(text.split())


def count_denominator(text: str, denominator: str) -> int:
    """Count ``text`` in a profile's denominator — ``"units"`` or ``"words"``.

    Which one is in play is a property of the *measurement*, never of the tokenizer, so it is
    selected by :class:`~tokenization.config.EvalProfile` rather than guessed here.
    """
    if denominator == "units":
        return count_units(text)
    if denominator == "words":
        return count_words(text)
    msg = f"unknown denominator {denominator!r}"
    raise ValueError(msg)


@dataclass(frozen=True)
class LangScore:
    """One language's measurement: its denominator count and its token count.

    ``units`` holds whichever denominator the profile scores in — faithful units under v2,
    whitespace words under v1. The field keeps one name because every formula downstream is the
    same shape; what differs is what was counted, which is why the two are never ranked together.
    """

    code: str
    units: int
    tokens: int

    @property
    def ratio(self) -> float:
        """Fertility X = tokens / denominator count. Lower is better; ``0.0`` when empty."""
        return self.tokens / self.units if self.units else 0.0


def spread(langs: list[LangScore]) -> float:
    """Gap between the largest and smallest fertility across languages."""
    ratios = [x.ratio for x in langs]
    return max(ratios) - min(ratios)


def score(langs: list[LangScore]) -> float:
    """Raw score = 1000 / (max fertility − min fertility); ``inf`` when all fertilities match."""
    gap = spread(langs)
    return 1000.0 / gap if gap else float("inf")


def mean_ratio(langs: list[LangScore]) -> float:
    """Corpus-wide fertility: total tokens / total units. ``0.0`` for an empty list.

    This is the honest counterweight to :func:`score`. Spread rewards *convergence* whichever way
    it is bought — a tokenizer that makes every language equally mediocre scores perfectly — so a
    config that shrinks the spread while raising this number has flattened the languages rather
    than improved them. It is deliberately corpus-wide rather than a mean of per-language ratios,
    so a 5,808-unit language cannot sway it as much as a 186,367-unit one.
    """
    total_units = sum(x.units for x in langs)
    return sum(x.tokens for x in langs) / total_units if total_units else 0.0


def hindi_penalty(langs: list[LangScore]) -> float:
    """Penalty factor ``exp(max(0, X_hi / 1.2 − 1))``, or ``1.0`` when Hindi is absent.

    This exists to block the exploit of *degrading the best language* to shrink the spread. It
    only bites above :data:`HINDI_THRESHOLD` — on the wiki-faithful corpus every fertility sits
    near 0.6–0.75, so the penalty is inert and that exploit is unguarded. :func:`degrades_best`
    is the check that actually holds us to it.
    """
    hindi = next((x for x in langs if x.code == "hi"), None)
    if hindi is None:
        return 1.0
    return math.exp(max(0.0, hindi.ratio / HINDI_THRESHOLD - 1.0))


def adjusted_score(langs: list[LangScore]) -> float:
    """Hindi-adjusted score = raw score / :func:`hindi_penalty` — the graded number."""
    return score(langs) / hindi_penalty(langs)


def degrades_best(langs: list[LangScore], baseline: list[LangScore]) -> bool:
    """Whether ``langs`` shrank the spread by making ``baseline``'s *best* language worse.

    The published penalty only fires above :data:`HINDI_THRESHOLD`, which nothing on this corpus
    reaches — so a run could collapse the spread purely by degrading whichever language sits at
    the minimum, and be rewarded for it. Any experiment flagged here is rejected on honesty
    grounds whatever it scores.
    """
    before = {x.code: x.ratio for x in baseline}
    best_code = min(before, key=lambda c: before[c])
    now = {x.code: x.ratio for x in langs}
    return now.get(best_code, before[best_code]) > before[best_code]
