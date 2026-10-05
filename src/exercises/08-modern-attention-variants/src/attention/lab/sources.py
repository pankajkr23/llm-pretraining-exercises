"""Facts the lab needs that `results/mechanisms.json` does not carry, each with its quote.

The catalogue drives the published page, so the lab does not add to it. Anything else the lab
builds on is recorded here in the same shape as a catalogue size: a value, the sentence it was read
from, and where. **Nothing here is trusted because it is written here** — every entry is re-found in
the downloaded source by `tools/verify_lab_sources.py`, and the result is the ledger
`verified.json`.

How to add one: find the source by search (never from memory), confirm the id on its own page,
copy the sentence exactly, run the verifier, and only then use the value.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LabSource:
    """One sourced fact.

    Attributes:
        id: Stable identifier, `<mechanism>.<param>`.
        value: The value the sentence states.
        quote: The sentence, copied exactly from the source.
        where: Section, table or equation it appears in.
        url: The document the quote is checked against (a versioned arXiv HTML page, a raw
            config file, or an archived post).
        title: The document's title.
        unit: Optional unit for the value.
    """

    id: str
    value: int | float | str | bool
    quote: str
    where: str
    url: str
    title: str
    unit: str = ""

    def __post_init__(self) -> None:
        """Refuse an entry with no quote, no location or no checkable URL."""
        if not self.quote.strip() or not self.where.strip() or not self.url.startswith("https://"):
            raise ValueError(f"lab source {self.id!r} needs a quote, a location and an https URL")


#: Filled in only after each entry has been found in its downloaded source.
LAB_SOURCES: dict[str, LabSource] = {}


def add(source: LabSource) -> LabSource:
    """Register a sourced fact; a duplicate id is an error."""
    if source.id in LAB_SOURCES:
        raise ValueError(f"lab source {source.id!r} added twice")
    LAB_SOURCES[source.id] = source
    return source
