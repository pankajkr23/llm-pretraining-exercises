"""Unit tests for the phase-2 computation (pure, fast, no network)."""

import pytest
from dataframework.coverage import build_matrix, capabilities_for
from dataframework.fertility import (
    PARITY_TARGET,
    cost_delta,
    measure,
    meets_parity,
    parity_ratio,
    tokens_for_corpus,
    training_cost,
    unmeasured,
)
from dataframework.fetch_benchmarks import MIN_WORDS as FETCH_MIN_WORDS
from dataframework.fetch_benchmarks import _pick_question_column, _row_to_item
from dataframework.grade import grade_dataset, is_commercially_usable, score_gates
from dataframework.mix import MAX_EPOCHS_HARD, check, compose, effective_tokens, is_buildable
from dataframework.orphans import find_orphans
from dataframework.shingles import (
    build_attributed_index,
    build_index,
    gram_width,
    is_contaminated,
    overlap,
    shingle,
)
from dataframework.sourcing import blockers, build_plan
from dataframework.vocab_sweep import find_peak, round_to_multiple, sweep


def _gates(**verdicts):
    return {
        name: {"verdict": verdict, "reasoning": "because", "confidence": "high"}
        for name, verdict in verdicts.items()
    }


# --------------------------------------------------------------------------- grade


def test_design_worked_example_grades_b():
    # docs/DESIGN.md §5 shows exactly this gate combination as grade B.
    record = {
        "gates": _gates(
            provenance="PASS",
            composition="CONDITIONAL",
            contamination="PASS",
            yield_="UNKNOWN",
            evidence="PASS",
        ),
        "gotchas": [],
    }
    assert grade_dataset(record)[0] == "B"
    assert score_gates(record["gates"]) == 7


def test_failed_provenance_is_excluded_not_merely_penalised():
    record = {"gates": _gates(provenance="FAIL", evidence="PASS"), "gotchas": []}
    assert grade_dataset(record)[0] == "X"
    assert not is_commercially_usable("X")


def test_failed_contamination_is_also_disqualifying():
    assert grade_dataset({"gates": _gates(contamination="FAIL"), "gotchas": []})[0] == "X"


def test_blocking_gotcha_excludes_even_with_clean_gates():
    record = {
        "gates": _gates(provenance="PASS", evidence="PASS"),
        "gotchas": [{"type": "SAFETY", "text": "CSAM", "severity": "blocking"}],
    }
    assert grade_dataset(record)[0] == "X"


def test_unknown_gates_score_nothing():
    # Ignorance must not be mistaken for a pass.
    assert score_gates(_gates(a="UNKNOWN", b="UNKNOWN")) == 0


# --------------------------------------------------------------------------- fertility


def test_parity_ratio_and_target():
    assert parity_ratio(1.78, 1.2) == pytest.approx(1.483, abs=1e-3)
    assert meets_parity(1.48)
    assert not meets_parity(8.0)  # the tokenizer tax the atlas measures under cl100k
    assert PARITY_TARGET == 1.5


def test_training_cost_scales_with_tokens_and_params():
    cost = training_cost(1e9, 40e9)
    assert cost["flops"] == 6 * 40e9 * 1e9
    assert cost["gpu_hours"] > 0
    assert cost["inr"] > cost["usd"]


def test_better_fertility_saves_money():
    saving = cost_delta(1e9, baseline_fertility=2.4, improved_fertility=1.85, n_params=40e9)
    assert saving["gpu_hours"] > 0
    assert saving["tokens"] == tokens_for_corpus(1e9, 2.4) - tokens_for_corpus(1e9, 1.85)


def test_measurement_requires_attribution():
    # INV-4: a fertility value with no tokenizer and no run is an annotation, not a measurement.
    with pytest.raises(ValueError, match="INV-4"):
        measure(lambda t: [1], {"hi": "क ख ग"}, tokenizer_ref="", run_id="r1")


def test_measured_values_carry_measured_provenance():
    out = measure(
        lambda t: [0] * 10, {"hi": "one two three four five"}, tokenizer_ref="tk", run_id="r1"
    )
    assert out["hi"]["provenance"] == "measured"
    assert out["hi"]["value"] == pytest.approx(2.0)


def test_unmeasured_is_unknown_never_estimated():
    # Ground rule 8 rules out "estimated" for fertility specifically.
    out = unmeasured(["hi", "ta"], "task 2.2b has not run")
    assert all(v["provenance"] == "unknown" and v["value"] is None for v in out.values())


# --------------------------------------------------------------------------- vocab sweep


def test_sweep_emits_the_whole_curve_and_a_peak():
    curve = sweep(reference_vocab=100_000, reference_fertility=2.4)
    assert len(curve) == 33
    peak = find_peak(curve)
    assert peak in curve
    assert all(row["fertility"] >= 1.0 for row in curve)


def test_larger_vocab_lowers_fertility_but_costs_more_softmax():
    curve = sweep(100_000, 2.4)
    assert curve[-1]["fertility"] < curve[0]["fertility"]
    assert curve[-1]["softmax_cost"] > curve[0]["softmax_cost"]


def test_vocab_is_rounded_for_tensor_cores():
    assert round_to_multiple(208_000) % 128 == 0
    assert round_to_multiple(208_896) == 208_896  # 1,632 x 128


# --------------------------------------------------------------------------- mix


def test_effective_tokens_is_pool_times_epochs():
    # Rule R1 — the correction to the "Chinchilla-brained" single-pass assumption.
    assert effective_tokens(300e9, 4) == 1.2e12


def test_repetition_lifts_a_thin_pool_without_new_data():
    once = compose(
        [
            {"name": "en", "unique_tokens": 3e12, "epochs": 1},
            {"name": "in", "unique_tokens": 300e9, "epochs": 1, "is_indic": True},
        ]
    )
    repeated = compose(
        [
            {"name": "en", "unique_tokens": 3e12, "epochs": 1},
            {"name": "in", "unique_tokens": 300e9, "epochs": 4, "is_indic": True},
        ]
    )
    assert repeated["indic_share"] > once["indic_share"]


def test_epochs_past_the_hard_ceiling_is_an_error():
    mix = compose([{"name": "tiny", "unique_tokens": 1e9, "epochs": MAX_EPOCHS_HARD + 1}])
    findings = check(mix)
    assert any(f["level"] == "error" for f in findings)
    assert not is_buildable(findings)


def test_mostly_synthetic_indic_warns():
    mix = compose(
        [
            {"name": "nat", "unique_tokens": 100e9, "epochs": 1, "is_indic": True},
            {
                "name": "syn",
                "unique_tokens": 900e9,
                "epochs": 1,
                "is_indic": True,
                "is_synthetic": True,
            },
        ]
    )
    assert any("synthetic" in f["message"] for f in check(mix))


def test_missing_always_on_lane_warns():
    mix = compose([{"name": "en", "unique_tokens": 1e12, "epochs": 1}])
    assert any("Always-ON" in f["message"] for f in check(mix))


# --------------------------------------------------------------------------- coverage


def test_code_mixed_is_not_programming():
    # In an Indic corpus "code-mixed" means Hindi-English switching; counting it as `code`
    # would hide a real hole.
    assert "code" not in capabilities_for({"name": "IndicKLAR", "type": "code-mixed evaluation"})
    assert "code" in capabilities_for({"name": "HumanEval", "type": "code generation"})


def test_hmmt_is_not_machine_translation():
    caps = capabilities_for({"name": "GSM8K / MATH / HMMT / MathArena", "type": "Math"})
    assert "translation" not in caps
    assert "math-reasoning" in caps


def test_matrix_reports_holes_explicitly():
    matrix = build_matrix([{"name": "OnlyMath", "type": "Math"}])
    assert "indic-language" in matrix["holes"]
    assert matrix["benchmark_count"] == 1


# --------------------------------------------------------------------------- orphans


def test_a_tier_no_benchmark_can_see_is_reported_and_priced():
    mix = compose(
        [
            {
                "name": "india-context",
                "unique_tokens": 500e9,
                "epochs": 1,
                "capabilities": ["india-context"],
            }
        ]
    )
    orphans = find_orphans(mix, [{"name": "OnlyMath", "type": "Math"}])
    assert len(orphans) == 1
    assert orphans[0]["usd"] > 0  # the cost is attached, so the trade-off is explicit


def test_a_covered_tier_is_not_an_orphan():
    mix = compose([{"name": "code", "unique_tokens": 1e11, "epochs": 1, "capabilities": ["code"]}])
    assert find_orphans(mix, [{"name": "HumanEval", "type": "code generation"}]) == []


# --------------------------------------------------------------------------- shingles


def test_identical_text_collides():
    text = " ".join(f"w{i}" for i in range(20))
    assert is_contaminated(text, shingle(text))


def test_unrelated_text_does_not_collide():
    a = " ".join(f"a{i}" for i in range(20))
    b = " ".join(f"b{i}" for i in range(20))
    assert not is_contaminated(b, shingle(a))


def test_a_planted_item_is_caught_inside_a_larger_document():
    item = "the quick brown fox jumps over the lazy dog again and again today"
    document = "unrelated preamble " * 10 + item + " unrelated tail " * 10
    assert overlap(document, shingle(item))


def test_a_short_item_hashes_to_one_whole_text_gram():
    """Shorter than the window means one gram, not thirteen — and it is only findable at width 4.

    The old name for this test claimed short items were "still indexed", which read as "still
    detectable" and was not true of the case that matters. See `ShingleIndex`.
    """
    assert len(shingle("only four words here")) == 1
    assert gram_width("only four words here") == 4
    assert gram_width("the quick brown fox jumps over the lazy dog again and again today") == 13


def test_an_index_records_every_width_it_used():
    index = build_attributed_index(
        {"B": ["only five words here now", " ".join(f"w{i}" for i in range(20))]}
    )
    assert index.widths == frozenset({5, 13})
    assert not index.unindexable


def test_an_index_refuses_items_narrower_than_the_floor():
    index = build_attributed_index({"B": ["far too short"]})
    assert index.widths == frozenset()
    assert index.unindexable == {"B": 1}


def test_missing_corpus_reports_unchecked_never_clean(tmp_path):
    from dataframework.config import Config

    index = build_index(Config(data_dir=tmp_path, web_dir=tmp_path))
    assert index["coverage"] == "none"
    assert "UNCHECKED" in index["note"]
    assert index["shingle_count"] == 0


# --------------------------------------------------------------------------- benchmark fetch


def test_question_column_is_found_across_plausible_schemas():
    """MILU's schema is not knowable until the dataset is reachable, so the guess must be broad."""
    assert _pick_question_column(["question", "option1", "answer"]) == "question"
    assert _pick_question_column(["question_text", "choices"]) == "question_text"
    assert _pick_question_column(["id", "Stem", "A"]) == "Stem"


def test_an_unknown_schema_exits_saying_what_it_saw():
    """Guessing wrong must be loud: a silent miss would write an empty index that reads as clean."""
    with pytest.raises(SystemExit, match="id.*lang.*payload"):
        _pick_question_column(["id", "lang", "payload"])


def test_an_item_carries_its_options():
    """A leaked document carries the question and its options together, so index them so."""
    row = {"question": "Which river", "option1": "Ganges", "option2": "Nile", "subject": "Geo"}
    assert _row_to_item(row, "question", True) == "Which river Ganges Nile"
    assert _row_to_item(row, "question", False) == "Which river"
    assert _row_to_item({"question": "   "}, "question", True) == ""


def test_the_fetch_floor_matches_the_index_floor():
    """Writing items the index would refuse anyway is just a confusing count."""
    from dataframework.shingles import MIN_SHINGLE_N

    assert FETCH_MIN_WORDS == MIN_SHINGLE_N


# --------------------------------------------------------------------------- sourcing


def _ds(**kw):
    base = {
        "id": "X",
        "name": "n",
        "category": "English Web",
        "grade": "B",
        "licence_commercial": True,
        "size_tokens": {"value": 1e9},
    }
    return {**base, **kw}


def test_unknown_licence_blocks_a_commitment():
    """Unknown is not permission. The whole framework rests on that distinction."""
    assert blockers(_ds(licence_commercial=None)) == ["licence"]
    assert blockers(_ds(licence_commercial=False)) == ["licence"]
    assert blockers(_ds()) == []


def test_a_budget_you_cannot_add_up_is_not_a_budget():
    assert "size" in blockers(_ds(size_tokens={"value": None}))
    assert "size" in blockers(_ds(size_tokens=None))


def test_a_rejected_dataset_can_never_be_committed():
    """INV-2 again, from the sourcing side: grade X is not a shortfall to be filled."""
    assert "grade" in blockers(_ds(grade="X"))
    assert "grade" in blockers(_ds(grade="C"))


def test_a_gap_is_not_a_candidate():
    assert "does not exist" in blockers(_ds(is_gap=True))


def test_the_plan_counts_only_what_it_may_count():
    mix = {"tiers": [{"name": "english-web-hq", "unique_tokens": 4e9}]}
    plan = build_plan(
        [_ds(id="ok"), _ds(id="nolic", licence_commercial=None), _ds(id="nosize", size_tokens={})],
        mix,
    )
    tier = plan["tiers"][0]
    assert tier["committed"] == ["ok"]
    assert tier["committed_tokens"] == 1e9
    assert tier["shortfall_tokens"] == 3e9
    assert tier["candidates_total"] == 3


def test_the_work_queue_ranks_paperwork_above_the_unmeasured():
    """A dataset blocked only on a licence counts the moment it clears; an unsized one does not."""
    mix = {"tiers": [{"name": "english-web-hq", "unique_tokens": 1e12}]}
    plan = build_plan(
        [
            _ds(id="small", licence_commercial=None, size_tokens={"value": 1e9}),
            _ds(id="huge", licence_commercial=None, size_tokens={"value": 9e12}),
            _ds(id="unsized", licence_commercial=None, size_tokens={}),
        ],
        mix,
    )
    assert [b["id"] for b in plan["blocked"]] == ["huge", "small", "unsized"]
    assert plan["blocked"][-1]["unlocks_tokens"] is None


def test_over_supply_is_not_a_surplus_to_spend_elsewhere():
    mix = {"tiers": [{"name": "english-web-hq", "unique_tokens": 1e9}]}
    tier = build_plan([_ds(size_tokens={"value": 9e9})], mix)["tiers"][0]
    assert tier["shortfall_tokens"] == 0
    assert tier["covered_share"] == 1.0
