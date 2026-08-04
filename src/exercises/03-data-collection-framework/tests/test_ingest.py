"""Unit tests for the CSV → record expansion (pure, fast, no network, no seed files needed)."""

from dataframework.ingest import (
    build_benchmark_record,
    build_dataset_record,
    derive_gates,
    parse_licence,
    parse_naturalness,
    parse_tokens,
    slugify,
)

SANGRAHA = {
    "ID": "IND-01",
    "Category": "Indic Text (PT)",
    "Dataset": "Sangraha",
    "Owner_Steward": "AI4Bharat, IIT Madras",
    "Tier": "GREEN",
    "License": "CC-BY-4.0 (data); MIT (tooling). Explicitly permits commercial use",
    "Size_Scale": "251B tokens total: Verified 64B / Unverified 24B / Synthetic 162B",
    "Stage": "PT",
    "Languages": "22 Indian languages",
    "Access": "https://huggingface.co/datasets/ai4bharat/sangraha ; paper arXiv:2403.06350",
    "Used_By": "Foundation for most Indic LLM work post-2024",
    "Risk_Notes": (
        "THE ANCHOR CORPUS. The 162B synthetic portion is machine-translated - do NOT count as "
        "natural Indic. Dedup against Varta before mixing."
    ),
}


def test_slugify():
    assert slugify("Sangraha") == "sangraha"
    assert slugify("The Stack v2") == "the-stack-v2"
    assert slugify("AIKosh (IndiaAI Datasets Platform)") == "aikosh-indiaai-datasets-platform"


# --------------------------------------------------------------------------- sizes


def test_token_count_is_read_and_typed():
    v = parse_tokens("251B tokens total")
    assert v.value == 251_000_000_000
    assert v.unit == "tokens"
    assert v.provenance == "estimated"


def test_size_without_a_token_count_is_unknown_not_guessed():
    v = parse_tokens("35,000 hours of speech")
    assert v.value is None
    assert v.provenance == "unknown"


def test_naturalness_split_is_extracted():
    # The headline 251B hides that only 88B is natural text — the whole point of the split.
    parts = parse_naturalness(SANGRAHA["Size_Scale"])
    assert parts["verified"]["value"] == 64_000_000_000
    assert parts["unverified"]["value"] == 24_000_000_000
    assert parts["synthetic"]["value"] == 162_000_000_000


def test_missing_split_yields_no_parts():
    assert parse_naturalness("900B tokens") == {}


# --------------------------------------------------------------------------- licence


def test_permissive_licence_flags_commercial():
    assert parse_licence("CC-BY-4.0 (data); MIT (tooling)")["commercial"] is True


def test_noncommercial_licence_is_flagged_false():
    assert parse_licence("CC BY-NC-SA 4.0 - NONCOMMERCIAL")["commercial"] is False


def test_share_alike_is_detected():
    assert parse_licence("CC BY-SA 4.0")["share_alike"] is True


def test_unstated_licence_is_none_not_assumed():
    assert parse_licence("")["commercial"] is None


# --------------------------------------------------------------------------- gates


def test_every_gate_carries_reasoning_and_a_citation():
    gates = derive_gates(SANGRAHA, {"COMPOSITION", "DEDUP"}, blocking=False)
    assert set(gates) == {"provenance", "composition", "contamination", "yield", "evidence"}
    for name, gate in gates.items():
        assert gate["reasoning"].strip(), f"{name} has no reasoning (INV-3)"
        assert gate["citations"], f"{name} cites nothing"


def test_blocking_note_fails_the_provenance_gate():
    gates = derive_gates(SANGRAHA, {"SAFETY"}, blocking=True)
    assert gates["provenance"]["verdict"] == "FAIL"


def test_silent_row_yields_unknown_not_a_pass():
    row = dict.fromkeys(SANGRAHA, "")
    gates = derive_gates(row, set(), blocking=False)
    assert gates["provenance"]["verdict"] == "UNKNOWN"
    assert gates["evidence"]["verdict"] == "UNKNOWN"


def test_contamination_is_never_asserted_as_pass_in_phase_one():
    # Decontamination is measured in phase 2; claiming PASS here would be inventing a result.
    gates = derive_gates(SANGRAHA, set(), blocking=False)
    assert gates["contamination"]["verdict"] == "UNKNOWN"


# --------------------------------------------------------------------------- records


def test_dataset_record_has_the_card_fields():
    r = build_dataset_record(SANGRAHA)
    for key in ("id", "name", "tier", "licence", "size", "gotchas", "gates", "access"):
        assert key in r
    assert r["id"] == "IND-01"
    assert r["is_gap"] is False
    assert {g["type"] for g in r["gotchas"]} >= {"COMPOSITION"}
    assert "arXiv:2403.06350" in r["access"]["links"]


def test_untiered_row_is_marked_a_gap():
    # "-" means the dataset does not exist yet; that slot is the differentiation argument.
    row = {**SANGRAHA, "Tier": "-", "Dataset": "Indic-commented code"}
    r = build_dataset_record(row)
    assert r["tier"] is None
    assert r["is_gap"] is True


def test_benchmark_record_never_carries_items():
    r = build_benchmark_record(
        {
            "Benchmark": "MILU",
            "Owner": "AI4Bharat / IBM Research",
            "Type": "Indic knowledge (MCQ)",
            "Coverage": "11 Indic languages",
            "Size": "~85K MCQs",
            "Split_Policy": "TEST (locked) + dedicated VAL set",
            "Access": "https://github.com/AI4Bharat/MILU",
            "Notes": "Sourced from 1,500+ Indian competitive exams.",
        }
    )
    assert r["held_out"] is True
    assert r["trust_band"] == "native-sourced"
    for banned in ("items", "questions", "examples", "samples"):
        assert banned not in r, "eval text must never enter the repo (INV-1)"
