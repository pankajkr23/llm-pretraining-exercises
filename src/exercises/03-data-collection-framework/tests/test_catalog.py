"""Unit tests for the spine validator (pure, fast, no network).

A validator that only ever passes is worthless, so these tests mostly feed it broken records and
assert it complains.
"""

from dataframework.catalog import (
    _check_gate,
    _check_row_ids,
    _check_value,
    validate_benchmark,
    validate_dataset,
)

GOOD_VALUE = {"value": 251e9, "unit": "tokens", "provenance": "estimated", "source": "seed"}
GOOD_GATE = {"verdict": "PASS", "reasoning": "licence permits it", "confidence": "high"}


def _dataset(**overrides):
    record = {
        "id": "IND-01",
        "name": "Sangraha",
        "tier": "GREEN",
        "size": {"tokens": GOOD_VALUE, "naturalness": {}},
        "gates": {"provenance": GOOD_GATE},
        "gotchas": [{"type": "DEDUP", "text": "dupes", "severity": "advisory"}],
        "opportunity": None,
        "note": None,
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------------- values


def test_a_clean_value_passes():
    errors = []
    _check_value(GOOD_VALUE, "x", errors)
    assert errors == []


def test_bare_number_is_rejected():
    errors = []
    _check_value(251e9, "x", errors)
    assert any("provenance-typed" in e for e in errors)


def test_value_without_a_source_is_rejected():
    errors = []
    _check_value({**GOOD_VALUE, "source": ""}, "x", errors)
    assert any("never invent figures" in e for e in errors)


def test_null_value_must_declare_unknown():
    errors = []
    _check_value({**GOOD_VALUE, "value": None}, "x", errors)
    assert any("unknown" in e for e in errors)


# --------------------------------------------------------------------------- gates


def test_gate_without_reasoning_is_rejected():
    errors = []
    _check_gate({**GOOD_GATE, "reasoning": "  "}, "g", errors)
    assert any("INV-3" in e for e in errors)


def test_unrecognised_verdict_is_rejected():
    errors = []
    _check_gate({**GOOD_GATE, "verdict": "PROBABLY"}, "g", errors)
    assert any("verdict" in e for e in errors)


# --------------------------------------------------------------------------- datasets


def test_a_clean_dataset_passes():
    errors = []
    validate_dataset(_dataset(), errors)
    assert errors == []


def test_dropped_research_content_is_rejected():
    # INV-5: a Risk & Notes field that yielded no caveat, upside or note was lost.
    errors = []
    validate_dataset(_dataset(gotchas=[], opportunity=None, note=None), errors)
    assert any("INV-5" in e for e in errors)


def test_a_gap_row_with_no_tier_is_allowed():
    errors = []
    validate_dataset(_dataset(tier=None), errors)
    assert errors == []


def test_bogus_tier_is_rejected():
    errors = []
    validate_dataset(_dataset(tier="PURPLE"), errors)
    assert any("tier" in e for e in errors)


def test_unrecognised_gotcha_type_is_rejected():
    errors = []
    validate_dataset(
        _dataset(gotchas=[{"type": "VIBES", "text": "t", "severity": "advisory"}]), errors
    )
    assert any("not recognised" in e for e in errors)


# --------------------------------------------------------------------------- benchmarks


def test_benchmark_carrying_eval_items_is_rejected():
    # INV-1: eval text must never enter the repo, let alone the web bundle.
    errors = []
    validate_benchmark({"name": "MILU", "split_policy": "TEST", "items": ["Q1"]}, errors)
    assert any("INV-1" in e for e in errors)


def test_benchmark_without_a_split_policy_is_rejected():
    errors = []
    validate_benchmark({"name": "MILU", "split_policy": ""}, errors)
    assert any("split policy" in e for e in errors)


# --------------------------------------------------------------------------- record ids


def test_duplicate_reference_ids_are_rejected():
    errors = []
    _check_row_ids([{"id": "R1"}, {"id": "R1"}], "risks", errors)
    assert any("duplicate" in e for e in errors)


def test_languages_may_key_on_iso_code_instead_of_id():
    errors = []
    _check_row_ids([{"code": "hi"}, {"code": "bn"}], "languages", errors)
    assert errors == []


def test_row_without_any_id_is_rejected():
    errors = []
    _check_row_ids([{"name": "nameless"}], "risks", errors)
    assert any("no id" in e for e in errors)
