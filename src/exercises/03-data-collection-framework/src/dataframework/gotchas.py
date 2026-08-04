"""Parse an Atlas *Risk & Notes* field into typed records.

This is the richest content in the catalogue and the easiest to lose. As prose in a table cell,
"HPLT still has 17.3% byte-exact duplicates" is invisible; typed, it joins every other `DEDUP`
caveat in one filterable view — the screen that saves someone a month.

**The field is "Risk *& Notes*", and the distinction matters.** Roughly a third of the 145 rows
record a genuine caveat ("CSAM — absolute blocklist"); the rest are observations and upsides
("matches FineWeb-2 with 6x fewer tokens"). Typing the second kind as a risk would satisfy INV-5's
letter while gutting its purpose: a `#gotchas` screen where most datasets carry a meaningless
`PROVENANCE` badge is worse than no screen. So a note is parsed into three buckets —

* `gotchas` — genuine caveats, typed for filtering;
* `opportunity` — the stated upside, rendered under ✦ OPPORTUNITY on the card;
* `note` — residual observations, kept verbatim.

INV-5 is then enforced as *nothing is dropped*: a non-empty field must produce at least one of the
three, and `catalog.py` fails the build otherwise. That is a stricter reading than "≥1 gotcha",
because it also forbids quietly discarding the half of the field that isn't a risk.

Classification is keyword-driven over sentences, ordered most-specific first so that, e.g., "CSAM"
is `SAFETY` rather than merely a `PROVENANCE` worry. `docs/TODO.md` task 1.3 names ten reference
datasets whose classification must be right; `tests/test_gotchas.py` pins each of them.
"""

import re
from dataclasses import dataclass

from .models import Gotcha, GotchaType, Severity

# Phrases that mark a caveat as bar-the-door rather than proceed-with-care.
_BLOCKING_MARKERS: tuple[str, ...] = (
    "absolute blocklist",
    "blocklist",
    "do not use",
    "avoid as a source",
    "exclude from",
    "pirated",
    "csam",
    "strip books3",
    "noncommercial",
    "non-commercial",
    "cc by-nc",
    "cc-by-nc",
)

# (type, patterns) — ordered most specific first; the first match per sentence wins.
_RULES: tuple[tuple[GotchaType, tuple[str, ...]], ...] = (
    (
        "SAFETY",
        (
            r"\bcsam\b",
            r"child sexual",
            r"\btoxic",
            r"\bharmful\b",
            r"privacy (failure|leakage)",
            r"medical-record",
            r"\bpii\b",
        ),
    ),
    (
        "PROVENANCE",
        (
            r"\bpirated\b",
            r"books3",
            r"libgen|anna'?s archive|z-library",
            r"openai[- ]output|alpaca|distilled from",
            r"litigation|lawsuit|\bsued\b|\bani\b v\.",
            r"generation provenance|licen[cs]e[- ]chain|model-output provenance",
            r"provenance (is )?(unclear|unknown)",
            r"scraped without|unclear origin",
        ),
    ),
    (
        "LICENCE",
        (
            r"noncommercial|non-commercial|cc[- ]?by[- ]?nc|\bnc\b",
            r"licen[cs]e (complexity|review|string|chain|is not|not stated"
            r"|unstated|unclear|ambiguous|murky)",
            r"(bespoke|community|custom) licen[cs]e",
            r"inherited licen[cs]e|licen[cs]e inherit",
            r"opt-?out",
            r"copyleft|share-?alike|\bgpl\b|\blgpl\b|\bmpl\b|\bepl\b",
            r"requires legal review|legal sign-?off",
            r"terms of service|\btos\b|restricts bulk",
            r"not osi|openrail",
            # NB: bare "redistribut" belongs to AVAILABILITY — docs/DESIGN.md §5 files
            # "Bharat Data Sagar (not redistributable)" there, not under LICENCE.
        ),
    ),
    (
        "DEDUP",
        (
            r"duplicat",
            r"\bdedup",
            r"overlap",
            r"subsumed by|superseded by",
        ),
    ),
    (
        "COMPOSITION",
        (
            r"machine-translated|machine translated",
            r"do not count.*natural|not natural|as natural",
            r"synthetic portion|labelled synthetic|counted as natural",
            r"paragraph-split|breaks long context",
            r"vendor claim|unaudited|ceo statement|figure is a",
            r"tokenizer counts|token counts differ",
            r"rephras|synthetic (rephrasing|diverse)",
            r"translated into|translation-derived|translated from",
            r"is not what|isn'?t what",
        ),
    ),
    (
        "ATTRIBUTION",
        (
            r"attribution",
            r"maintain a manifest",
            r"weights?-as-derivative",
        ),
    ),
    (
        "AVAILABILITY",
        (
            r"not (publicly )?redistributable|contribution is invited",
            r"will release|verify (the )?upload|not yet (released|available)",
            r"url dead|download.*dead|no longer hosted|taken down",
            r"\bgated\b|request access",
            r"announced but|unavailable|does not exist yet",
        ),
    ),
    (
        "SOURCING",
        (
            r"use \w+ instead",
            r"better (legal )?posture",
            r"prefer\b.*instead",
            r"a better route",
        ),
    ),
    (
        "HETEROGENEITY",
        (
            r"heterogene",
            r"per-(dataset|item|record) (licen[cs]e|review|ingestion)",
            r"do not bulk|don'?t bulk|bulk[- ]download",
            r"mixed terms|mixed licen[cs]",
        ),
    ),
)

# Fragments that read as an upside rather than a caveat.
_OPPORTUNITY_MARKERS: tuple[str, ...] = (
    "arguably as valuable",
    "companion pipeline",
    "the anchor corpus",
    "largest",
    "most permissive",
    "directly serves",
    "value is",
    "highest-quality",
    "cleanest",
    "best-in-class",
    "best ",
    "punches far above",
    "key scarcity datapoint",
    "directly relevant",
    "doubly useful",
    "the audit is the value",
    "validated recipe",
    "port to indic",
    "curriculum spine",
    "fewer tokens",
    "useful for",
    "gold standard",
    "state of the art",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.;!?])\s+|\s+-\s+(?=[A-Z])")


@dataclass(frozen=True, slots=True)
class ParsedNotes:
    """Everything one *Risk & Notes* field yielded.

    Attributes:
        gotchas: Genuine caveats, typed for filtering.
        opportunity: The stated upside, if any.
        note: Residual observations that are neither caveat nor upside, kept verbatim.
    """

    gotchas: tuple[Gotcha, ...]
    opportunity: str | None
    note: str | None

    @property
    def is_empty(self) -> bool:
        """Whether the field yielded nothing at all (only true for a blank field)."""
        return not self.gotchas and not self.opportunity and not self.note


def _sentences(note: str) -> list[str]:
    """Split a *Risk & Notes* field into candidate sentences.

    Args:
        note: The raw field.

    Returns:
        Non-empty, stripped fragments.
    """
    return [s.strip() for s in _SENTENCE_SPLIT.split(note) if s and s.strip()]


def _severity_for(text: str) -> Severity:
    """Classify a fragment as blocking or advisory.

    Args:
        text: The fragment.

    Returns:
        ``"blocking"`` if it carries a bar-the-door marker, else ``"advisory"``.
    """
    lowered = text.lower()
    return "blocking" if any(m in lowered for m in _BLOCKING_MARKERS) else "advisory"


def classify(text: str) -> GotchaType | None:
    """Classify one fragment into a gotcha type.

    Args:
        text: The fragment.

    Returns:
        The first matching type, or `None` if the fragment states no caveat.
    """
    lowered = text.lower()
    for gotcha_type, patterns in _RULES:
        for pattern in patterns:
            if re.search(pattern, lowered):
                return gotcha_type
    return None


def _is_opportunity(text: str) -> bool:
    """Whether a fragment reads as an upside.

    Args:
        text: The fragment.

    Returns:
        True if it carries an opportunity marker.
    """
    lowered = text.lower()
    return any(m in lowered for m in _OPPORTUNITY_MARKERS)


def parse(note: str | None) -> ParsedNotes:
    """Split a *Risk & Notes* field into caveats, upside, and residual observations.

    At most one gotcha per type: the first fragment that triggers a type carries it, so a card shows
    one clear `DEDUP` line rather than three restatements. A blocking marker anywhere in the field
    escalates every caveat in it.

    Args:
        note: The raw field; may be `None` or blank.

    Returns:
        The parsed buckets. Empty only when the field is.
    """
    if not note or not note.strip():
        return ParsedNotes(gotchas=(), opportunity=None, note=None)

    found: dict[GotchaType, Gotcha] = {}
    opportunity: str | None = None
    residual: list[str] = []

    for sentence in _sentences(note):
        gotcha_type = classify(sentence)
        if gotcha_type is not None:
            if gotcha_type not in found:
                found[gotcha_type] = Gotcha(
                    type=gotcha_type, text=sentence, severity=_severity_for(sentence)
                )
        elif opportunity is None and _is_opportunity(sentence):
            opportunity = sentence
        else:
            residual.append(sentence)

    if _severity_for(note) == "blocking":
        found = {
            t: (g if g.is_blocking else Gotcha(type=g.type, text=g.text, severity="blocking"))
            for t, g in found.items()
        }

    return ParsedNotes(
        gotchas=tuple(found.values()),
        opportunity=opportunity,
        note=" ".join(residual) or None,
    )
