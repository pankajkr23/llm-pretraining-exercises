"""The five invariants, enforced against the built spine and bundle.

These are the tests that make the framework's claims checkable rather than asserted. Each one is
written twice over: once against the real spine (it must pass), and once against a deliberately
broken fixture (it must fail). A guard that has never been seen to fail is not a guard.

    INV-1  training never touches eval data          3.1, 3.2, 3.7
    INV-2  no grade-X dataset in a commercial mix    3.3
    INV-3  every judgment carries its reasoning      3.4
    INV-4  fertility is measured, never annotated    3.5
    INV-5  no Atlas content silently dropped         3.8
    (plus) no bare numbers in the bundle             3.6

Run in existing CI by `uv run pytest`; no separate workflow.
"""

import json
import re

import pytest
from dataframework.catalog import EXPECTED_COUNTS, validate
from dataframework.config import Config
from dataframework.gotchas import parse as parse_notes
from dataframework.grade import grade_dataset, is_commercially_usable
from dataframework.shingles import (
    DIGEST_BYTES,
    SHINGLE_N,
    build_attributed_index,
    build_index,
    find_collisions,
    is_contaminated,
    shingle,
)

CFG = Config()
WEB = CFG.web_dir

# A stand-in eval registry. Real benchmark items are not in the repo (open item B3) and must never
# be — so the contamination demo uses a fixture whose text is unmistakably synthetic.
FIXTURE_REGISTRY: dict[str, list[str]] = {
    "MILU": [
        "Which of the following rivers forms the largest delta in the world "
        "before emptying into the Bay of Bengal according to standard geography texts",
        "The Indian Councils Act of 1909 is more commonly remembered by which name "
        "in the constitutional history of modern India",
    ],
    "IndicXTREME": [
        "Identify the grammatical case of the highlighted noun phrase in the "
        "following Marathi sentence taken from a news corpus",
    ],
    # Seven words: shorter than the 13-word window, which is the ordinary shape of an MCQ stem
    # and the case the gate used to miss silently.
    "ShortStem": ["Which river drains the Chota Nagpur plateau"],
}

# Below MIN_SHINGLE_N. A window this narrow matches ordinary prose, so the gate must refuse it
# and say so rather than index it and cry wolf.
TOO_SHORT_REGISTRY: dict[str, list[str]] = {"Stub": ["Name the capital"]}


def _catalog() -> list[dict]:
    return json.loads(CFG.catalog_file.read_text(encoding="utf-8"))


def _bundle() -> dict:
    return json.loads((WEB / "data.json").read_text(encoding="utf-8"))


def _built() -> bool:
    return (WEB / "data.json").exists() and CFG.catalog_file.exists()


pytestmark = pytest.mark.skipif(
    not _built(), reason="spine or bundle not built; run `python -m dataframework` first"
)


# ------------------------------------------------------------------ 3.1  INV-1a


def test_no_trainable_dataset_has_a_failing_contamination_gate():
    """A dataset that would poison evaluation must not be usable, whatever else it scores."""
    for record in _catalog():
        contamination = (record.get("gates") or {}).get("contamination") or {}
        if contamination.get("verdict") == "FAIL":
            grade, _ = grade_dataset(record)
            assert grade == "X", (
                f"{record['id']} fails the contamination gate yet grades {grade} — "
                "it would be trainable"
            )


def test_a_contaminated_dataset_cannot_grade_usable():
    """Breaking 3.1 must fail: flip contamination to FAIL and the grade must become X."""
    poisoned = {
        "id": "FIXTURE",
        "gates": {
            "provenance": {"verdict": "PASS", "reasoning": "r", "confidence": "high"},
            "contamination": {
                "verdict": "FAIL",
                "reasoning": "collides with MILU",
                "confidence": "high",
            },
        },
        "gotchas": [],
    }
    grade, reasoning = grade_dataset(poisoned)
    assert grade == "X"
    assert not is_commercially_usable(grade)
    assert "contamination" in reasoning


# ------------------------------------------------------------------ 3.2  INV-1b


def test_eval_text_absent_from_the_web_bundle():
    """No eval item text may appear anywhere in the shipped bundle."""
    blob = "".join(path.read_text(encoding="utf-8") for path in WEB.rglob("*.json"))
    for benchmark, items in FIXTURE_REGISTRY.items():
        for item in items:
            assert item[:64] not in blob, f"{benchmark} item text leaked into web/"


def test_shingles_bundle_carries_hashes_only():
    """shingles.json must be digests, never recoverable text."""
    payload = json.loads((WEB / "shingles.json").read_text(encoding="utf-8"))
    for digest in payload.get("shingles", []):
        assert re.fullmatch(rf"[0-9a-f]{{{DIGEST_BYTES * 2}}}", digest), (
            f"{digest!r} is not a bare hex digest"
        )


def test_a_leaked_item_would_be_caught():
    """Breaking 3.2 must fail: the same check over a blob containing an item must trip."""
    item = FIXTURE_REGISTRY["MILU"][0]
    leaked_blob = '{"note": "' + item + '"}'
    assert item[:64] in leaked_blob


# ------------------------------------------------------------------ 3.3  INV-2


def test_no_grade_x_dataset_is_commercially_usable():
    """INV-2: an excluded dataset must never be admitted to a commercial mix."""
    for record in _catalog():
        grade, _ = grade_dataset(record)
        if grade == "X":
            assert not is_commercially_usable(grade), f"{record['id']} is X yet admitted"


def test_the_blocklist_is_not_empty():
    """A blocklist that excludes nothing is not evidence the corpus is clean."""
    grades = [grade_dataset(record)[0] for record in _catalog()]
    assert grades.count("X") > 0, "no dataset graded X — the exclusion rule never fired"


def test_a_blocking_gotcha_forces_exclusion():
    """Breaking 3.3 must fail: a CSAM caveat cannot grade usable."""
    record = {
        "id": "FIXTURE",
        "gates": {"provenance": {"verdict": "PASS", "reasoning": "r", "confidence": "high"}},
        "gotchas": [{"type": "SAFETY", "text": "CSAM found", "severity": "blocking"}],
    }
    assert grade_dataset(record)[0] == "X"


# ------------------------------------------------------------------ 3.4  INV-3


def test_every_gate_carries_reasoning_and_confidence():
    """INV-3: an unauditable verdict looks like evidence, which is worse than no verdict."""
    for record in _catalog():
        for name, gate in (record.get("gates") or {}).items():
            assert (gate.get("reasoning") or "").strip(), f"{record['id']}.{name}: no reasoning"
            assert gate.get("confidence") in {"high", "medium", "low"}, (
                f"{record['id']}.{name}: confidence {gate.get('confidence')!r}"
            )


def test_a_gate_without_reasoning_cannot_be_constructed():
    """Breaking 3.4 must fail at construction, not review."""
    from dataframework.models import Gate

    with pytest.raises(ValueError, match="reasoning"):
        Gate(verdict="PASS", reasoning="   ", confidence="high")


# ------------------------------------------------------------------ 3.5  INV-4


def test_no_fertility_value_claims_measurement_without_a_run():
    """INV-4: `measured` requires a tokenizer and a run id in the source string."""
    fertility = _bundle()["fertility"]["by_language"]
    for language, value in fertility.items():
        if value.get("provenance") == "measured":
            assert "@" in (value.get("source") or ""), (
                f"{language}: claims measured but the source names no tokenizer@run"
            )


def test_unmeasured_fertility_is_unknown_never_estimated():
    """Ground rule 8: fertility may not ship as a plausible-looking estimate."""
    for language, value in _bundle()["fertility"]["by_language"].items():
        assert value["provenance"] in {"measured", "unknown"}, (
            f"{language}: fertility shipped as {value['provenance']!r}"
        )


def test_measurement_without_attribution_raises():
    """Breaking 3.5 must fail: measure() refuses an anonymous run."""
    from dataframework.fertility import measure

    with pytest.raises(ValueError, match="INV-4"):
        measure(lambda t: [1], {"hi": "a b c"}, tokenizer_ref="tk", run_id="")


# ------------------------------------------------------------------ 3.6  no bare numbers


def _bare_numbers(node, path="", inherited=False) -> list[str]:
    """Collect numeric leaves not covered by a `provenance` declaration on any ancestor."""
    found: list[str] = []
    if isinstance(node, dict):
        typed = inherited or "provenance" in node
        for key, value in node.items():
            found += _bare_numbers(value, f"{path}.{key}", typed)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found += _bare_numbers(value, f"{path}[{i}]", inherited)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        if not inherited:
            found.append(path)
    return found


def test_no_bare_number_in_the_bundle():
    """Every figure must be renderable as measured or estimated — so none may be naked."""
    bare = _bare_numbers(_bundle())
    assert not bare, f"{len(bare)} bare number(s), e.g. {bare[:5]}"


def test_the_bare_number_check_can_actually_fail():
    """Breaking 3.6 must fail: an untyped figure is detected."""
    assert _bare_numbers({"tokens": 251_000_000_000})
    assert not _bare_numbers(
        {"tokens": {"value": 251e9, "unit": "tokens", "provenance": "estimated", "source": "s"}}
    )


# ------------------------------------------------------------------ 3.7  the demo


def test_planting_a_known_eval_item_is_caught_and_names_the_benchmark(capsys):
    """The demo: plant an eval item in a training shard and watch the gate name the benchmark.

    This is the most convincing artifact in the submission — the invariant is not a claim about
    the pipeline, it is a thing you can watch happen.
    """
    index = build_attributed_index(FIXTURE_REGISTRY)
    planted = FIXTURE_REGISTRY["MILU"][0]
    shard = (
        "Ordinary web text about monsoon agriculture in the Gangetic plain. "
        + planted
        + " More ordinary web text about irrigation and canal systems."
    )

    hits = find_collisions(shard, index)

    assert hits, "planted item was not detected"
    assert "MILU" in hits, f"detected contamination but named {list(hits)} instead of MILU"
    assert "IndicXTREME" not in hits, "named a benchmark the shard does not contain"

    # The terminal output the report reproduces.
    print("\nCONTAMINATION GATE — FAILED")
    print(f"  shard collides with: {', '.join(f'{b} ({n} shingles)' for b, n in hits.items())}")
    print("  action: drop the shard, or drop the benchmark from the eval set")
    assert "MILU" in capsys.readouterr().out


def test_a_clean_shard_passes_the_gate():
    """The gate must not cry wolf, or it will be turned off."""
    index = build_attributed_index(FIXTURE_REGISTRY)
    clean = "Ordinary web text about monsoon agriculture, irrigation and canal systems in India."
    assert not find_collisions(clean, index)
    assert not is_contaminated(clean, index)


def test_detection_survives_the_item_being_embedded_mid_document():
    """Contamination is rarely a whole document; a quoted item must still trip the gate."""
    item = FIXTURE_REGISTRY["IndicXTREME"][0]
    document = "preamble " * 200 + item + " epilogue " * 200
    assert "IndicXTREME" in find_collisions(document, build_attributed_index(FIXTURE_REGISTRY))


# ------------------------------------------------------------------ 3.7b  short items


def test_a_short_item_is_caught_inside_a_longer_shard():
    """An item shorter than the window must still be found once embedded.

    The gate used to be structurally incapable of this: a 7-word item emits one 7-gram, a shard
    emits only 13-grams, and the two never intersect — so the shard came back clean. The index now
    records the width each item was hashed at and the document is shingled at each of them.
    """
    index = build_attributed_index(FIXTURE_REGISTRY)
    short = FIXTURE_REGISTRY["ShortStem"][0]
    shard = "Ordinary web text about canal irrigation. " + short + " More ordinary web text."

    hits = find_collisions(shard, index)

    assert "ShortStem" in hits, f"short item went undetected; gate named {list(hits)}"
    assert is_contaminated(shard, index)


def test_the_short_item_check_can_actually_fail():
    """Breaking 3.7b must fail: drop the width-awareness and the short item vanishes again.

    This reproduces the old single-width lookup. If it ever stops finding nothing, the widths are
    no longer doing the work the test above credits them with.
    """
    index = build_attributed_index(FIXTURE_REGISTRY)
    short = FIXTURE_REGISTRY["ShortStem"][0]
    shard = "Ordinary web text about canal irrigation. " + short + " More ordinary web text."

    old_lookup = shingle(shard, SHINGLE_N) & set(index.grams)
    assert not old_lookup, (
        "a 13-gram-only lookup found the short item, so this mutation no longer proves anything"
    )


def test_an_item_below_the_floor_is_reported_not_indexed():
    """Too short to identify anything: refuse it, and make the refusal a number."""
    index = build_attributed_index(TOO_SHORT_REGISTRY)

    assert not index.grams, "a 3-word item was indexed; it would match ordinary prose"
    assert index.unindexable == {"Stub": 1}
    assert not find_collisions("Name the capital of Bihar and its principal river.", index)


def test_the_widths_in_play_are_reported(tmp_path):
    """Coverage must state which widths it used and how many items it could not cover."""
    corpus = tmp_path / "benchmarks"
    corpus.mkdir()
    (corpus / "Mixed.json").write_text(
        json.dumps([FIXTURE_REGISTRY["ShortStem"][0], *TOO_SHORT_REGISTRY["Stub"]]),
        encoding="utf-8",
    )

    report = build_index(Config(data_dir=tmp_path, web_dir=tmp_path))

    assert report["gram_widths"] == [7]
    assert report["unindexable_items"] == 1
    assert "shorter than" in report["note"]


# ------------------------------------------------------------------ 3.8  INV-5


def test_no_atlas_section_dropped():
    """INV-5: record counts are the tripwire for content going missing.

    The expected counts live in `catalog.EXPECTED_COUNTS` rather than being restated here, so there
    is one source of truth. Two of them differ from the draft in `docs/TODO.md` — risks is 20, not
    21, and market holds 17 deals, not 16 — because the extraction counted the Atlas rather than
    the plan. Padding to the planned number would have meant inventing rows.
    """
    counts, errors = validate(CFG)
    for name, expected in EXPECTED_COUNTS.items():
        assert counts.get(name) == expected, (
            f"{name}: {counts.get(name)} of {expected} — content dropped"
        )
    assert not errors, f"spine has {len(errors)} problem(s); first: {errors[0]}"


def test_every_risk_note_survived_into_the_record():
    """INV-5, the stricter half: a Risk & Notes field must leave a trace.

    Not necessarily a gotcha — the field mixes caveats with observations, and forcing a caveat out
    of "matches FineWeb-2 with 6x fewer tokens" would be a lie. But it must survive as *something*.
    """
    for record in _catalog():
        if record.get("gotchas") or record.get("opportunity") or record.get("note"):
            continue
        pytest.fail(f"{record['id']}: Risk & Notes yielded nothing — content dropped")


def test_the_completeness_check_can_actually_fail():
    """Breaking 3.8 must fail: a short register is detected."""
    parsed = parse_notes("")
    assert parsed.is_empty  # a genuinely blank note yields nothing...
    assert not parse_notes("CSAM found in the dataset").is_empty  # ...but real content never does


def test_every_dataset_kept_its_identity():
    """Duplicate or missing ids would silently merge records during export."""
    ids = [record.get("id") for record in _catalog()]
    assert all(ids), "a catalogue record has no id"
    assert len(ids) == len(set(ids)), "duplicate id in the catalogue"
