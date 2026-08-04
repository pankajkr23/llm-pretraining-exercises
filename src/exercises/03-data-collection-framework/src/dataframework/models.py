"""The type system — where the invariants are structural rather than documented.

Three record primitives back every judgment the framework makes:

* :class:`Value` — a number that knows where it came from. Ground rule 4 ("every numeric value is
  provenance-typed") and rule 7 ("never invent figures") are enforced here, not by review.
* :class:`Gotcha` — a typed caveat parsed from an Atlas *Risk & Notes* field, so caveats can be
  filtered across the whole catalogue instead of being prose nobody reads.
* :class:`Gate` — the verdict of one of the framework's five questions, carrying its reasoning and
  confidence (INV-3).

`Literal` annotations are erased at runtime, so each class re-checks its own enumerations in
`__post_init__`. Without that, `Value(1.2, "tok/word", "measurd")` would sail through the type
checker and ship a silently untyped number.

**Split of responsibility:** this module enforces what must be true of *any* record (a verdict is
one of four strings; a stated number has a unit and a source). Dataset-level policy — "a catalogue
row with a Risk & Notes field must yield at least one gotcha" — belongs to `catalog.py`, which can
see the whole corpus.
"""

from dataclasses import dataclass
from typing import Literal, get_args

Provenance = Literal["measured", "estimated", "unknown"]
"""How a number came to be known. Anything not measured or derived is `unknown`, never invented."""

GotchaType = Literal[
    "LICENCE",
    "DEDUP",
    "COMPOSITION",
    "PROVENANCE",
    "SAFETY",
    "ATTRIBUTION",
    "AVAILABILITY",
    "SOURCING",
    "HETEROGENEITY",
]
"""The nine caveat kinds, so the catalogue is filterable by failure mode."""

Severity = Literal["blocking", "advisory"]
"""`blocking` stops a dataset entering a mix; `advisory` only downgrades its grade."""

Verdict = Literal["PASS", "CONDITIONAL", "FAIL", "UNKNOWN"]
"""The outcome of one framework question for one dataset."""

Confidence = Literal["high", "medium", "low"]
"""How much weight the judgment can bear."""

PROVENANCES: frozenset[str] = frozenset(get_args(Provenance))
GOTCHA_TYPES: frozenset[str] = frozenset(get_args(GotchaType))
SEVERITIES: frozenset[str] = frozenset(get_args(Severity))
VERDICTS: frozenset[str] = frozenset(get_args(Verdict))
CONFIDENCES: frozenset[str] = frozenset(get_args(Confidence))


def _require_member(value: object, allowed: frozenset[str], field: str) -> None:
    """Raise unless ``value`` is one of ``allowed``.

    Args:
        value: The value to check.
        allowed: The permitted strings.
        field: Field name, for the error message.

    Raises:
        ValueError: If ``value`` is not in ``allowed``.
    """
    if value not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}, got {value!r}")


def _require_text(value: str, field: str) -> None:
    """Raise unless ``value`` is a non-blank string.

    Args:
        value: The value to check.
        field: Field name, for the error message.

    Raises:
        ValueError: If ``value`` is not a string or is blank.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string, got {value!r}")


@dataclass(frozen=True, slots=True)
class Value:
    """A number that carries its own provenance.

    A known number must state its unit and where it came from; an unknown one must admit it by
    holding `None`. That biconditional is what stops an estimate quietly hardening into a fact.

    Attributes:
        value: The magnitude, or `None` when the provenance is `unknown`.
        unit: What the magnitude counts, e.g. `"tokens"`, `"tok/word"`, `"USD"`.
        provenance: `measured`, `estimated`, or `unknown`.
        source: URL, arXiv id, or `"computed:<fn>"`. Required unless the value is unknown.
    """

    value: float | int | None
    unit: str
    provenance: Provenance
    source: str | None = None

    def __post_init__(self) -> None:
        """Validate the enumeration and the known/unknown biconditional.

        Raises:
            ValueError: If the provenance is unrecognised, the unit is blank, an unknown value
                carries a magnitude, or a known value lacks a magnitude or a source.
        """
        _require_member(self.provenance, PROVENANCES, "provenance")
        _require_text(self.unit, "unit")
        if self.provenance == "unknown":
            if self.value is not None:
                raise ValueError(f"provenance 'unknown' must carry value=None, got {self.value!r}")
        else:
            if self.value is None:
                raise ValueError(f"provenance {self.provenance!r} requires a value; use 'unknown'")
            _require_text(self.source or "", "source")

    @property
    def is_known(self) -> bool:
        """Whether a magnitude is actually available."""
        return self.value is not None

    @classmethod
    def unknown(cls, unit: str, source: str | None = None) -> "Value":
        """Build an explicitly-unknown value.

        The honest alternative to inventing a figure (ground rule 7).

        Args:
            unit: What the magnitude would have counted.
            source: Optional note on why it is unknown.

        Returns:
            A `Value` with `value=None` and `provenance="unknown"`.
        """
        return cls(value=None, unit=unit, provenance="unknown", source=source)


@dataclass(frozen=True, slots=True)
class Gotcha:
    """A typed caveat extracted from an Atlas *Risk & Notes* field.

    Typing these is what makes the catalogue's richest content queryable — "show every dataset with
    a DEDUP problem" — instead of prose buried in a table cell.

    Attributes:
        type: Which failure mode this is.
        text: The caveat itself, in plain language.
        severity: `blocking` bars the dataset from a mix; `advisory` only downgrades it.
    """

    type: GotchaType
    text: str
    severity: Severity

    def __post_init__(self) -> None:
        """Validate the enumerations and require non-empty text.

        Raises:
            ValueError: If the type or severity is unrecognised, or the text is blank.
        """
        _require_member(self.type, GOTCHA_TYPES, "type")
        _require_text(self.text, "text")
        _require_member(self.severity, SEVERITIES, "severity")

    @property
    def is_blocking(self) -> bool:
        """Whether this caveat bars the dataset from a mix outright."""
        return self.severity == "blocking"


@dataclass(frozen=True, slots=True)
class Gate:
    """The verdict of one framework question for one dataset (INV-3).

    Reasoning and confidence are non-optional by construction: a verdict nobody can audit is worse
    than no verdict, because it looks like evidence.

    `citations` is a tuple rather than a list so the record is genuinely immutable — a `list` field
    on a frozen dataclass can still be mutated in place.

    Attributes:
        verdict: `PASS`, `CONDITIONAL`, `FAIL`, or `UNKNOWN`.
        reasoning: Why this verdict — never blank.
        confidence: How much weight the judgment bears.
        citations: Supporting URLs, arXiv ids, or `"computed:<fn>"` markers.
    """

    verdict: Verdict
    reasoning: str
    confidence: Confidence
    citations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the enumerations, the reasoning, and every citation.

        Raises:
            ValueError: If the verdict or confidence is unrecognised, the reasoning is blank, or
                any citation is blank.
        """
        _require_member(self.verdict, VERDICTS, "verdict")
        _require_text(self.reasoning, "reasoning")
        _require_member(self.confidence, CONFIDENCES, "confidence")
        if not isinstance(self.citations, tuple):
            raise TypeError(f"citations must be a tuple, got {type(self.citations).__name__}")
        for i, citation in enumerate(self.citations):
            _require_text(citation, f"citations[{i}]")

    @property
    def is_blocking_failure(self) -> bool:
        """Whether this gate fails outright (as opposed to passing with conditions)."""
        return self.verdict == "FAIL"
