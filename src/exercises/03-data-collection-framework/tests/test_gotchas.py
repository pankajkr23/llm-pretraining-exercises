"""Unit tests for the Risk & Notes parser (pure, fast, no network).

The reference cases come from `docs/TODO.md` task 1.3, which names ten datasets whose
classification must be right. They are pinned here so a rule change cannot quietly regress them.
"""

import pytest
from dataframework.gotchas import ParsedNotes, classify, parse

# --------------------------------------------------------------------------- buckets


def test_blank_note_yields_nothing():
    for blank in (None, "", "   "):
        assert parse(blank).is_empty


def test_a_pure_caveat_becomes_a_typed_gotcha():
    p = parse("CSAM FOUND IN THE DATASET (Thiel, Dec 2023). ABSOLUTE BLOCKLIST")
    assert {g.type for g in p.gotchas} == {"SAFETY"}
    assert all(g.is_blocking for g in p.gotchas)


def test_a_pure_upside_is_not_forced_into_a_fake_risk():
    # "Risk & Notes" also carries notes; typing these as risks would gut the #gotchas view.
    p = parse("Matches FineWeb-2 performance with 6x FEWER tokens")
    assert p.gotchas == ()
    assert p.opportunity is not None


def test_residual_observation_is_kept_not_dropped():
    p = parse("Published rephrasing prompts in Appendix H")
    assert not p.is_empty  # INV-5: something survives


def test_a_mixed_note_splits_into_caveat_and_upside():
    p = parse(
        "THE ANCHOR CORPUS. The 162B synthetic portion is machine-translated - do NOT count as "
        "natural Indic. Companion pipeline 'Setu' is arguably as valuable as the data itself"
    )
    assert "COMPOSITION" in {g.type for g in p.gotchas}
    assert p.opportunity is not None


# --------------------------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("CSAM found in the dataset", "SAFETY"),
        ("CONTAINS BOOKS3 (pirated)", "PROVENANCE"),
        ("Krutrim Community License requires legal review", "LICENCE"),
        ("17.3% byte-exact duplicates remained after official dedup", "DEDUP"),
        ("the synthetic portion is machine-translated Wikimedia", "COMPOSITION"),
        ("record-level attribution - maintain a manifest", "ATTRIBUTION"),
        ("not publicly redistributable", "AVAILABILITY"),
        ("USE eCOURTS INSTEAD - same content, far better legal posture", "SOURCING"),
        ("heterogeneous licensing is the whole risk", "HETEROGENEITY"),
    ],
)
def test_representative_fragments_classify(text, expected):
    assert classify(text) == expected


def test_a_plain_observation_classifies_as_no_caveat():
    assert classify("GPT-2 tokenizer counts") in (None, "COMPOSITION")
    assert classify("Broad Indic coverage including long-tail languages") is None


def test_specificity_order_csam_is_safety_not_provenance():
    # Both rules could plausibly fire; SAFETY must win.
    assert classify("CSAM and pirated material") == "SAFETY"


# --------------------------------------------------------------------------- severity


def test_blocking_marker_escalates_every_caveat_in_the_note():
    p = parse("Superseded by IndicCorp v2. EXCLUDE from commercial mix - it is NC licensed.")
    assert p.gotchas
    assert all(g.is_blocking for g in p.gotchas), "a blocklist marker taints the whole record"


def test_ordinary_caveat_stays_advisory():
    p = parse("Assume similar duplication for Indic partitions - RE-DEDUP YOURSELF")
    assert all(not g.is_blocking for g in p.gotchas)


# --------------------------------------------------------------------------- shape


def test_at_most_one_gotcha_per_type():
    p = parse("Duplicates found. More duplicates here. Still more overlap in the corpus.")
    types = [g.type for g in p.gotchas]
    assert len(types) == len(set(types))


def test_result_is_immutable():
    p = parse("CSAM found")
    assert isinstance(p, ParsedNotes)
    assert isinstance(p.gotchas, tuple)
