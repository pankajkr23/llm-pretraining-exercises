"""Unit tests for the record primitives (pure, fast, no network).

The point of these types is that bad records cannot be constructed, so most of these tests assert
that construction *raises*.
"""

import dataclasses

import pytest
from dataframework.models import (
    CONFIDENCES,
    GOTCHA_TYPES,
    PROVENANCES,
    SEVERITIES,
    VERDICTS,
    Gate,
    Gotcha,
    Value,
)

# --------------------------------------------------------------------------- Value


def test_known_value_round_trips():
    v = Value(1.85, "tok/word", "measured", source="computed:fertility")
    assert v.value == 1.85
    assert v.is_known
    assert v.provenance == "measured"


def test_unknown_helper_builds_a_none_value():
    v = Value.unknown("tokens", source="never measured by anyone")
    assert v.value is None
    assert not v.is_known
    assert v.provenance == "unknown"


@pytest.mark.parametrize("provenance", sorted(PROVENANCES))
def test_every_declared_provenance_is_constructible(provenance):
    kwargs = {"value": None} if provenance == "unknown" else {"value": 1, "source": "s"}
    assert Value(unit="tokens", provenance=provenance, **kwargs).provenance == provenance


def test_typo_in_provenance_raises_rather_than_shipping():
    # Literal is erased at runtime; without the __post_init__ check this would silently pass.
    with pytest.raises(ValueError, match="provenance"):
        Value(1.2, "tok/word", "measurd", source="s")  # type: ignore[arg-type]


def test_unknown_value_may_not_carry_a_magnitude():
    with pytest.raises(ValueError, match="must carry value=None"):
        Value(42, "tokens", "unknown")


def test_known_value_may_not_be_none():
    with pytest.raises(ValueError, match="requires a value"):
        Value(None, "tokens", "estimated", source="s")


def test_known_value_requires_a_source():
    # "Never invent figures" — a stated number must be traceable.
    with pytest.raises(ValueError, match="source"):
        Value(300, "tokens", "measured")


def test_unit_is_required():
    with pytest.raises(ValueError, match="unit"):
        Value(1, "  ", "estimated", source="s")


def test_value_is_frozen():
    v = Value(1, "tokens", "estimated", source="s")
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.value = 2


# --------------------------------------------------------------------------- Gotcha


def test_gotcha_round_trips_and_reports_blocking():
    g = Gotcha("SAFETY", "LAION-5B contains CSAM; use Re-LAION.", "blocking")
    assert g.is_blocking
    assert Gotcha("DEDUP", "17.3% byte-exact dupes remain.", "advisory").is_blocking is False


@pytest.mark.parametrize("gotcha_type", sorted(GOTCHA_TYPES))
def test_all_nine_gotcha_types_are_constructible(gotcha_type):
    assert Gotcha(gotcha_type, "text", "advisory").type == gotcha_type


@pytest.mark.parametrize("severity", sorted(SEVERITIES))
def test_both_severities_are_constructible(severity):
    assert Gotcha("LICENCE", "text", severity).severity == severity


def test_unknown_gotcha_type_raises():
    with pytest.raises(ValueError, match="type"):
        Gotcha("VIBES", "text", "advisory")  # type: ignore[arg-type]


def test_empty_gotcha_text_raises():
    with pytest.raises(ValueError, match="text"):
        Gotcha("DEDUP", "   ", "advisory")


def test_unknown_severity_raises():
    with pytest.raises(ValueError, match="severity"):
        Gotcha("DEDUP", "text", "kinda bad")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- Gate


def test_gate_round_trips():
    g = Gate("PASS", "CC-BY-4.0 permits commercial training.", "high", ("2403.06350",))
    assert g.verdict == "PASS"
    assert g.citations == ("2403.06350",)
    assert not g.is_blocking_failure


@pytest.mark.parametrize("verdict", sorted(VERDICTS))
def test_every_verdict_is_constructible(verdict):
    assert Gate(verdict, "reasoning", "low").verdict == verdict


@pytest.mark.parametrize("confidence", sorted(CONFIDENCES))
def test_every_confidence_is_constructible(confidence):
    assert Gate("UNKNOWN", "reasoning", confidence).confidence == confidence


def test_gate_without_reasoning_raises():
    # INV-3: an unauditable verdict is worse than none, because it looks like evidence.
    with pytest.raises(ValueError, match="reasoning"):
        Gate("FAIL", "", "high")


def test_unknown_verdict_raises():
    with pytest.raises(ValueError, match="verdict"):
        Gate("PROBABLY", "reasoning", "high")  # type: ignore[arg-type]


def test_unknown_confidence_raises():
    with pytest.raises(ValueError, match="confidence"):
        Gate("PASS", "reasoning", "certain")  # type: ignore[arg-type]


def test_blank_citation_raises():
    with pytest.raises(ValueError, match=r"citations\[1\]"):
        Gate("PASS", "reasoning", "high", ("ok", " "))


def test_citations_must_be_a_tuple_so_the_record_is_really_immutable():
    with pytest.raises(TypeError, match="tuple"):
        Gate("PASS", "reasoning", "high", ["mutable"])  # type: ignore[arg-type]


def test_gate_is_frozen():
    g = Gate("PASS", "reasoning", "high")
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.verdict = "FAIL"
