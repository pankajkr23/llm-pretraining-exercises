"""Where a date came from, recorded so a reader can check it rather than trust it.

The assignment's one warning is the reason this module exists: an agent asked for a launch date
will supply a confident one it has half remembered, so every date must be checked against the paper
or release itself rather than recalled.

A date with no source is not a weaker claim than a sourced one -- it is a different kind of object,
and this exercise refuses to publish it. So every date carries the URL it was read from, the
**verbatim** string it was read from, and the day somebody looked. `quoted_date` is the field that
does the work: a reader can compare it against `date` without leaving the page, and a transcription
error shows up as a disagreement between two fields rather than as a number nobody can check.

**`confidence` is allowed to say `unverified`, and that is the point.** A catalogue that cannot
express doubt will express confidence it has not earned. One mechanism in Session 8 -- DroPE -- is
described in the course with no paper named and a garbled title, so it may genuinely have no
findable primary source. Recording that honestly is a result; inventing an arXiv id is a
fabrication.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime

#: arXiv's submission-history format, e.g. `[v1] Mon, 12 Jun 2017 17:57:34 UTC (1,102 KB)`.
#:
#: Parsed rather than trusted so the recorded ISO date can be checked against the string it was
#: read from. A transcription slip -- reading 2102.11174 as "11 Feb" when v1 is 22 Feb -- is exactly
#: the error the assignment warns about, and it is invisible unless something compares the two.
_ARXIV_QUOTE = re.compile(r"\[v(\d+)\]\s+\w{3},\s+(\d{1,2})\s+(\w{3})\s+(\d{4})")


def parse_quoted(quoted: str) -> date | None:
    """The date inside an arXiv submission-history line, or None if it is not one.

    Returns None rather than raising: not every source is arXiv, and a forum post or a model release
    quotes its date in its own format. A `None` here means "cannot cross-check automatically", which
    callers must handle rather than read as agreement.
    """
    match = _ARXIV_QUOTE.search(quoted)
    if not match:
        return None
    _, day, month, year = match.groups()
    try:
        return datetime.strptime(f"{day} {month} {year}", "%d %b %Y").date()
    except ValueError:
        return None


#: What kind of artifact a date was read from.
#:
#: Not every mechanism arrives in a paper. NTK-aware RoPE scaling began as a forum post; sliding
#: window attention as a decoder default arrived with a model release. Flattening those into
#: "paper" would be tidier and false.
SOURCE_KINDS: frozenset[str] = frozenset({"paper", "post", "release", "repo", "none"})

#: Values `Source.confidence` may take.
CONFIDENCE: frozenset[str] = frozenset({"verified", "unverified"})


@dataclass(frozen=True)
class Source:
    """The artifact a date was read from.

    Attributes:
        kind: One of `SOURCE_KINDS`.
        title: The artifact's own title, verbatim.
        url: What was fetched. A reader should be able to open this and see `quoted_date`.
        quoted_date: The date string **as the source states it** — for arXiv, the submission-history
            line for v1, e.g. `"[v1] Tue, 20 Apr 2021 09:54:06 UTC"`.
        verified_on: The day somebody actually opened `url`.
        arxiv_id: Present when `kind == "paper"` and the paper is on arXiv.
        confidence: `verified` only when `url` was opened and `quoted_date` read from it.
        note: Anything a careful reader needs — a shared paper, a contested attribution, a
            mechanism whose origin is not a single artifact.
    """

    kind: str
    title: str
    url: str
    quoted_date: str
    verified_on: date
    arxiv_id: str | None = None
    confidence: str = "verified"
    note: str = ""

    def __post_init__(self) -> None:
        """Reject a source that cannot be checked, at construction rather than at render time."""
        if self.kind not in SOURCE_KINDS:
            raise ValueError(
                f"unknown source kind {self.kind!r}; expected one of {sorted(SOURCE_KINDS)}"
            )
        if self.confidence not in CONFIDENCE:
            raise ValueError(
                f"confidence must be one of {sorted(CONFIDENCE)}, got {self.confidence!r}"
            )
        if self.confidence == "verified":
            if not self.url:
                raise ValueError(f"{self.title!r} claims to be verified but records no url")
            if not self.quoted_date:
                raise ValueError(
                    f"{self.title!r} claims to be verified but quotes no date from the source. "
                    "The quoted string is what makes the claim checkable."
                )

    @property
    def is_checkable(self) -> bool:
        """Whether a reader could open this and confirm the date themselves."""
        return bool(self.url) and bool(self.quoted_date) and self.confidence == "verified"
