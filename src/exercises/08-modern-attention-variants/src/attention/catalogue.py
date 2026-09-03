"""The mechanisms, what each one traded, and when it actually appeared.

`results/mechanisms.json` is the tracked evidence this exercise exists to produce. Everything a
future page renders comes from there, so the page cannot disagree with the catalogue and the
catalogue cannot disagree with the sources it cites.

**The shape is the instructor's, not ours.** The requirements specifies the narrative each entry
must carry:

    what came before -> the problem it ran into -> the mechanism introduced
    -> what that fixed -> what it cost in exchange

and three questions each entry has to answer: what it buys, what it gives up, and when it is
actually the right choice. Those are fields below, not prose conventions, because a field can be
checked for emptiness and a paragraph cannot.

**`MANDATED` is the requirements' own list, quoted.** The instructor said he will score zero for a
missing mechanism, so "did we cover everything" is a test rather than a memory. Keep the wording as
he wrote it; the mapping from his phrase to our key lives beside it and is the part allowed to
change.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from attention.sources import Source

EXERCISE = Path(__file__).resolve().parents[2]
CATALOGUE = EXERCISE / "results" / "mechanisms.json"

#: The coverage list, verbatim from the requirements text, mapped to catalogue keys.
#:
#: Left side is the instructor's phrase exactly as written; right side is every key that phrase
#: requires. Splitting them means a rename on our side can never quietly drop one of his items:
#: the test reads the left.
#:
#: **The right side is a tuple because two of his phrases name two mechanisms each**, and that is
#: not a formatting detail -- it is where this guard failed. "sparse and top-k attention" mapped to
#: `sparse_attention` alone, and `sparse_attention.aka` additionally claimed "top-k attention" as
#: an alias, so the catalogue asserted the two were the same technique and the guard agreed.
#: They are not the same technique: a fixed sparse pattern decides which pairs can ever interact
#: before the model sees any data, while top-k decides per query from the scores themselves, and
#: the reference notes teach the difference at length ("How do we know which keys are best?").
#: Covering half a phrase and passing is exactly the "missing or mis-explained mechanism" the
#: requirement scores zero for.
MANDATED: dict[str, tuple[str, ...]] = {
    "standard attention": ("standard_attention",),
    "absolute learned positions": ("learned_absolute",),
    "sinusoidal": ("sinusoidal",),
    "RoPE": ("rope",),
    "ALiBi": ("alibi",),
    "MQA": ("mqa",),
    "GQA": ("gqa",),
    "sliding window": ("sliding_window",),
    "attention sinks": ("attention_sinks",),
    "NTK-aware scaling": ("ntk_aware",),
    "YaRN": ("yarn",),
    "linear attention": ("linear_attention",),
    "the delta rule": ("delta_rule",),
    "Gated DeltaNet": ("gated_deltanet",),
    "MLA": ("mla",),
    "sparse and top-k attention": ("sparse_attention", "topk_attention"),
    "compressed and sparse attention as DeepSeek does it": ("nsa",),
    "DroPE": ("drope",),
}

#: How a mechanism's glyph is drawn. Four generators cover all twenty-three.
#:
#: `field` — a T x T support: which scores survive. `stack` — how many key/value heads are kept.
#: `state` — a fixed-size store, the same size at any context. `bands` — rotary frequency bands.
GLYPH_KINDS: frozenset[str] = frozenset({"field", "stack", "state", "bands"})

#: How much of a glyph's shape is sourced.
#:
#: **This distinction is the whole reason the field exists.** The catalogue records no window size,
#: sink count, stride, block size, top-k, latent width or state dimension for any entry, so a glyph
#: drawn to specific numbers would be inventing them — the exact fabrication this exercise is built
#: to prevent. `schematic` means the *shape* is faithful and the *numbers* are ours, and the page
#: says so on the plate rather than leaving a reader to assume otherwise.
GLYPH_SCALES: frozenset[str] = frozenset({"illustrative", "schematic"})


#: Which bill a mechanism pays down. Exercise 08's organising idea: attention charges twice, and
#: everything after the original is somebody paying less of one of them.
#:
#: `origin` exists because the first entries on the timeline do not pay a bill — they *create* the
#: situation the rest of the list responds to. Forcing them into `compute` or `cache` would make the
#: era counts in `timeline.pressure_by_period` claim an optimisation pressure that did not yet
#: exist.
BILLS: frozenset[str] = frozenset({"origin", "compute", "cache", "position", "both"})


@dataclass(frozen=True)
class Glyph:
    """How one mechanism is drawn, and how much of that drawing is sourced.

    Attributes:
        kind: One of `GLYPH_KINDS`.
        params: Generator arguments — window widths, head counts, and so on.
        scale: `illustrative` when the numbers come from a source, `schematic` when they are ours.
        source: Why this shape, in words a reader can weigh. Required.
        sizes: Real quantities the drawing may use, each carrying its own provenance. A number may
            enter this catalogue only with a citation attached -- see `_check_sizes`.
    """

    kind: str
    params: dict
    scale: str
    source: str
    sizes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """A glyph with no stated provenance is a picture presenting itself as evidence."""
        self._check_sizes()
        if self.kind not in GLYPH_KINDS:
            raise ValueError(f"unknown glyph kind {self.kind!r}; expected {sorted(GLYPH_KINDS)}")
        if self.scale not in GLYPH_SCALES:
            raise ValueError(
                f"glyph scale must be one of {sorted(GLYPH_SCALES)}, got {self.scale!r}"
            )
        if len(self.source.split()) < 6:
            raise ValueError(
                "every glyph must say where its shape comes from; a drawing with no provenance "
                "reads as measured and is not"
            )

    def _check_sizes(self) -> None:
        """A number may enter only with a citation attached.

        `GLYPH_SCALES` explains why the catalogue held no sizes at all: *a glyph drawn to specific
        numbers would be inventing them - the exact fabrication this exercise is built to prevent.*
        Real sizes are allowed now because the diagrams need them, and the guarantee is preserved by
        making provenance the price of entry, exactly as `Source` refuses a verified citation with
        no URL. A `stated` size quotes the sentence it was read from; an `ours` size says why we
        chose it. Neither may be a bare number.
        """
        for name, spec in (self.sizes or {}).items():
            if not isinstance(spec, dict) or "value" not in spec:
                raise ValueError(f"size {name!r} must be a mapping with a 'value'")
            origin = spec.get("from")
            if origin == "stated":
                quote = str(spec.get("quote", "")).strip()
                if not quote:
                    raise ValueError(f"size {name!r} claims to be stated but quotes nothing")
                #: The quote must CONTAIN the number it is evidence for. That is a far stronger
                #: check than a word count, and it is the right one: a hyperparameter is quoted as
                #: a fragment ("sliding stride d=16") rather than a sentence, so a length floor
                #: rejects honest evidence while still admitting a long quote that never mentions
                #: the value. This admits the fragment and rejects the mismatch.
                if not _quote_evidences(spec["value"], quote):
                    raise ValueError(
                        f"size {name!r} is {spec['value']}, but the quote offered as evidence does "
                        f"not contain that number: {quote!r}"
                    )
                if not spec.get("where"):
                    raise ValueError(f"size {name!r} is stated but says nowhere in the source")
            elif origin == "ours":
                if len(str(spec.get("note", "")).split()) < 6:
                    raise ValueError(
                        f"size {name!r} is ours and must say why we chose it, in a sentence"
                    )
            else:
                raise ValueError(
                    f"size {name!r} must declare from: 'stated' or 'ours', got {origin!r}"
                )


@dataclass(frozen=True)
class Adoption:
    """One shipped model that uses this mechanism, and the sentence saying so.

    **A model name is a claim, so it is sourced exactly like a date.** The page names real models
    because a reader otherwise cannot tell whether it is describing history, a research frontier or
    the thing inside the chatbot they used this morning — and "almost every open model uses them"
    asks for trust while offering nothing to check. Every entry here was read out of that model's
    own paper and the quote is verified as a contiguous substring of it.

    **An empty list is a result, not a gap.** Reformer and top-k attention have no entry because no
    model paper we read claims them, and that absence is one of the more informative things on the
    plate: it separates the mechanisms the field adopted from the ones it admired.

    Attributes:
        model: The model's name as its own paper writes it.
        quote: The sentence from that paper, verbatim.
        where: Section or table, plus the arXiv id.
        url: The paper.
        confidence: `explicit` when the paper names the mechanism; `implied` when it describes it
            without the name and the description is unambiguous.
        note: Scope that a bare name would misrepresent -- most often that only some model sizes
            use it.
    """

    model: str
    quote: str
    where: str
    url: str
    confidence: str = "explicit"
    note: str = ""

    def __post_init__(self) -> None:
        """Refuse a model name with nothing behind it."""
        if not self.quote.strip():
            raise ValueError(f"adoption by {self.model!r} must quote the paper that says so")
        if not self.url.strip():
            raise ValueError(f"adoption by {self.model!r} must link the paper")
        if self.confidence not in {"explicit", "implied"}:
            raise ValueError(f"adoption by {self.model!r} has confidence {self.confidence!r}")


@dataclass(frozen=True)
class Mechanism:
    """One entry on the timeline.

    Attributes:
        key: Stable identifier, used by the page and by `MANDATED`.
        name: Display name.
        date: When it first appeared publicly, from `source`.
        source: Where that date was read.
        bill: Which of the two bills it addresses, or `position`.
        what_existed: The state of the art it arrived into.
        problem: The problem that existed *at that moment*.
        mechanism: What it actually does, mechanically.
        what_it_fixed: What became cheaper or possible.
        new_tradeoff: What it made worse. Required — see `__post_init__`.
        buys: One clause: what you get.
        gives_up: One clause: what you pay.
        when_to_choose: The workload it is right for.
        taught_in_source: Whether Exercise 08 covered it, or whether we sourced it from outside.
        bonus: True for a mechanism the instructor did not list at all.
        shipped_in: Models that use it, each with the sentence from its own paper.
            Empty where no paper we read claims it, which is itself a finding.
    """

    key: str
    name: str
    date: date
    source: Source
    bill: str
    what_existed: str
    problem: str
    mechanism: str
    what_it_fixed: str
    new_tradeoff: str
    buys: str
    gives_up: str
    when_to_choose: str
    taught_in_source: bool = True
    bonus: bool = False
    aka: tuple[str, ...] = field(default_factory=tuple)
    shipped_in: tuple[Adoption, ...] = ()
    glyph: Glyph | None = None

    def __post_init__(self) -> None:
        """Refuse an entry that has only upside.

        The requirements are explicit that a technique written down with only upside has not been
        understood yet. An empty `new_tradeoff` or `gives_up` is how that failure would enter
        the catalogue, so it is rejected at construction rather than noticed in review.
        """
        if self.bill not in BILLS:
            raise ValueError(
                f"{self.key}: unknown bill {self.bill!r}; expected one of {sorted(BILLS)}"
            )
        for name in ("new_tradeoff", "gives_up", "when_to_choose"):
            if not getattr(self, name).strip():
                raise ValueError(
                    f"{self.key}: {name} is empty. Every mechanism here is a trade; an entry with "
                    f"no stated cost has not been understood yet."
                )


def load(path: Path = CATALOGUE) -> list[Mechanism]:
    """Read the catalogue, newest last.

    Args:
        path: The tracked JSON.

    Returns:
        Mechanisms in date order.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = [_mechanism(entry) for entry in raw["mechanisms"]]
    return sorted(out, key=lambda m: (m.date, m.key))


def _mechanism(entry: dict) -> Mechanism:
    """Build one `Mechanism` from its JSON form."""
    src = dict(entry["source"])
    source = Source(
        kind=src["kind"],
        title=src["title"],
        url=src.get("url", ""),
        quoted_date=src.get("quoted_date", ""),
        verified_on=date.fromisoformat(src["verified_on"]),
        arxiv_id=src.get("arxiv_id"),
        confidence=src.get("confidence", "verified"),
        note=src.get("note", ""),
    )
    pattern = entry.get("pattern")
    glyph = (
        Glyph(
            kind=pattern["kind"],
            params=pattern.get("params", {}),
            scale=pattern["scale"],
            source=pattern["source"],
            sizes=pattern.get("sizes", {}),
        )
        if pattern
        else None
    )
    known = {
        f
        for f in Mechanism.__dataclass_fields__
        if f not in {"date", "source", "aka", "glyph", "shipped_in"}
    }
    fields = {k: v for k, v in entry.items() if k in known}
    return Mechanism(
        date=date.fromisoformat(entry["date"]),
        source=source,
        aka=tuple(entry.get("aka", ())),
        glyph=glyph,
        shipped_in=tuple(Adoption(**a) for a in entry.get("shipped_in", ())),
        **fields,
    )


def _quote_evidences(value: object, quote: str) -> bool:
    """Does this quote actually contain the number it is offered as evidence for?

    Taught the notation the sources use, rather than loosened. Papers write context lengths as
    "32k" and "1M", not "32768" and "1000000", so a literal substring test rejects an honest quote.
    It still rejects the case that matters: a quote that never mentions the value at all, which is
    a number attributed to a paper on nothing more than our say-so.
    """
    flat = quote.replace(",", "").lower()
    if str(value).lower() in flat:
        return True
    if isinstance(value, int):
        for unit, scale in (("k", 1024), ("k", 1000), ("m", 1024**2), ("m", 1000**2)):
            if value % scale != 0:
                continue
            #: Both spacings, because papers use both -- "1M tokens" and "At 1 M tokens, the FLOPs
            #: reduction reaches 28x" are the same claim, and rejecting the second would have
            #: thrown away MSA's context length on a typographic detail. Teaching the check the
            #: notation is what its docstring describes; a literal test rejects honest evidence.
            if any(f"{value // scale}{sep}{unit}" in flat for sep in ("", " ")):
                return True
    return False


def missing_mandated(mechanisms: list[Mechanism]) -> list[str]:
    """Which of the instructor's required mechanisms are absent.

    Returns:
        His phrases, not our keys -- so a failure reads in the words he graded against.
    """
    have = {m.key for m in mechanisms}
    return [phrase for phrase, keys in MANDATED.items() if not set(keys) <= have]


def undrawn(mechanisms: list[Mechanism]) -> list[Mechanism]:
    """Entries the page cannot draw a glyph for.

    The plate shows all twenty-three or it is not the plate; an entry with no glyph would be a
    silent hole in a figure whose whole claim is completeness.
    """
    return [m for m in mechanisms if m.glyph is None]


def unverified(mechanisms: list[Mechanism]) -> list[Mechanism]:
    """Entries whose date a reader could not check for themselves."""
    return [m for m in mechanisms if not m.source.is_checkable]
