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
    normalise,
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
    assert not is_commercially_usable(poisoned)
    assert "contamination" in reasoning


# ------------------------------------------------------------------ 3.2  INV-1b


def _leaks(blob: str, registry: dict[str, list[str]]) -> list[str]:
    """Every registry item whose text appears in `blob`.

    Extracted so the check below and its mutation proof run the same code. The proof used to build
    a string from an item and assert the item was in it, which touched no project code and could
    not fail under any change to this repository.
    """
    return [
        f"{benchmark}: {item[:40]}"
        for benchmark, items in registry.items()
        for item in items
        if item[:64] in blob
    ]


def test_eval_text_absent_from_the_web_bundle():
    """INV-1b: no eval item text may appear anywhere in the shipped bundle.

    Scans the real benchmark corpus where it is present, not only the synthetic fixtures. The
    fixtures were never within reach of the pipeline, so a scan for them alone could never have
    detected the risk it exists to detect: an actual MILU item reaching `web/`.
    """
    blob = "".join(path.read_text(encoding="utf-8") for path in WEB.rglob("*.json"))

    assert not _leaks(blob, FIXTURE_REGISTRY)

    real = CFG.data_dir / "benchmarks" / "MILU.json"
    if not real.exists():
        pytest.skip("no benchmark corpus present (data/benchmarks/ is git-ignored)")
    items = json.loads(real.read_text(encoding="utf-8"))
    texts = [i if isinstance(i, str) else (i.get("text") or i.get("question") or "") for i in items]
    assert not _leaks(blob, {"MILU": [t for t in texts if len(t) >= 64]})


def test_shingles_bundle_carries_hashes_only():
    """shingles.json must be digests, never recoverable text."""
    payload = json.loads((WEB / "shingles.json").read_text(encoding="utf-8"))
    assert "shingles" not in payload, "the digest index must not ship to the browser"
    index_path = CFG.data_dir / "shingle_index.json"
    if not index_path.exists():
        pytest.skip("no shingle index built (data/benchmarks/ is empty)")
    for digest in json.loads(index_path.read_text(encoding="utf-8")).get("shingles", []):
        assert re.fullmatch(rf"[0-9a-f]{{{DIGEST_BYTES * 2}}}", digest), (
            f"{digest!r} is not a bare hex digest"
        )


def test_a_leaked_item_would_be_caught():
    """Breaking 3.2 must fail: the real scanner, over a bundle carrying an item, must name it."""
    item = FIXTURE_REGISTRY["MILU"][0]
    clean = "".join(path.read_text(encoding="utf-8") for path in WEB.rglob("*.json"))

    assert not _leaks(clean, FIXTURE_REGISTRY), "the bundle is not clean to begin with"
    assert _leaks(clean + '{"note": "' + item + '"}', FIXTURE_REGISTRY) == [f"MILU: {item[:40]}"]


# ------------------------------------------------------------------ 3.3  INV-2


def test_no_excluded_dataset_reaches_a_commercial_mix():
    """INV-2: an excluded dataset must never be admitted to a commercial mix.

    The old version of this asked `is_commercially_usable(grade)` inside `if grade == "X"`, and
    that function was `grade != "X"` — so it reduced to `assert not ("X" != "X")` and could not
    fail for any catalogue. It restated the implementation instead of testing the claim. INV-2 is
    about what reaches the mix, so this reads the plan the pipeline actually builds. Correction X18.
    """
    catalogue = _catalog()
    excluded = {r["id"] for r in catalogue if grade_dataset(r)[0] == "X"}
    assert excluded, "no dataset graded X — the exclusion rule never fired, so this proves nothing"

    # Read the shipped plan rather than rebuilding one: that bundle is what the page renders and
    # what a reader would train from, so it is the surface the invariant is a promise about.
    sourcing = _bundle()["sourcing"]
    committed = {ident for tier in sourcing["tiers"] for ident in tier["committed"]}

    assert not (committed & excluded), (
        f"grade-X datasets reached the commercial mix: {sorted(committed & excluded)}"
    )


def test_the_exclusion_check_can_actually_fail():
    """Breaking 3.3 must fail: commit an excluded dataset and the guard must object.

    The mutation forges the plan rather than the catalogue, because that is the surface INV-2
    protects — what ends up in the mix, not what the grader thought of it.
    """
    catalogue = _catalog()
    excluded = {r["id"] for r in catalogue if grade_dataset(r)[0] == "X"}
    sourcing = _bundle()["sourcing"]
    forged = {ident for tier in sourcing["tiers"] for ident in tier["committed"]}
    forged |= {sorted(excluded)[0]}

    assert forged & excluded, "the mutation no longer commits an excluded dataset"


def test_an_unestablished_licence_is_not_permission():
    """INV-2's other half: unknown is not permission, and the function that says so must say it.

    `is_commercially_usable` looked only at the grade, so it returned True for a dataset with no
    established licence — contradicting `sourcing.blockers` and the legal chapter, both of which
    treat an unknown licence as a blocker.
    """
    gates = {
        g: {"verdict": "PASS", "reasoning": "r", "confidence": "high"}
        for g in ("provenance", "composition", "contamination", "yield", "evidence")
    }
    assert is_commercially_usable({"id": "A", "gates": gates, "licence_commercial": True})
    assert not is_commercially_usable({"id": "B", "gates": gates, "licence_commercial": None})
    assert not is_commercially_usable({"id": "C", "gates": gates, "licence_commercial": False})


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


# Blocks whose numbers are sums over the catalogue's own `size_tokens`. Not one of the 145 records
# carries a measured size, so nothing derived from them may claim to be measured either.
_DERIVED_FROM_CATALOGUE_SIZES = ("sourcing", "lifecycle", "orphan_tiers")


def test_the_protocol_gap_note_matches_the_run_it_describes():
    """INV-4: the run must not misdescribe its own coverage.

    `protocol_gaps` was a hardcoded literal reading "three of the six tokenizers are unavailable",
    rendered on the page long after the run had measured five with one unavailable — a false
    statement about its own coverage, in the chapter whose subject is honest measurement.
    """
    record = json.loads((CFG.records_dir / "fertility.json").read_text(encoding="utf-8"))
    note = record["protocol_gaps"]

    assert f"{len(record['tokenizers_measured'])} tokenizers measured" in note, note
    assert f"{len(record['tokenizers_unavailable'])} unavailable" in note, note
    assert record["corpus"] in note


def test_no_measured_value_names_an_unresolved_run():
    """INV-4: a measurement must name the run that produced it — a placeholder is not a run.

    `main()` stamped every source with `pending-<timestamp>` and then substituted the real id, but
    only inside `by_tokenizer`. `conversational` was missed, so 115 values shipped in the public
    bundle claiming to be measured against an id prefixed `pending-` that matched no run and did not
    equal `run_id` in the same file. The old guard passed them because it only asked for an "@".
    Correction X17.
    """

    def unresolved(node: object) -> list[str]:
        """Every `source` naming a placeholder run.

        Walks the structure rather than grepping the text, because the corrections register
        *describes* this bug in prose and a substring search flags its own correction record.
        """
        out: list[str] = []
        if isinstance(node, dict):
            source = node.get("source")
            if isinstance(source, str) and "pending-" in source:
                out.append(source)
            for child in node.values():
                out += unresolved(child)
        elif isinstance(node, list):
            for child in node:
                out += unresolved(child)
        return out

    for path in (CFG.records_dir / "fertility.json", WEB / "records.json"):
        bad = unresolved(json.loads(path.read_text(encoding="utf-8")))
        assert not bad, f"{path.name} ships {len(bad)} unresolved run id(s), e.g. {bad[0]}"


def test_the_unresolved_run_check_can_actually_fail():
    """Breaking X17 must fail: a placeholder nested where the old patch never reached is caught.

    `conversational` is exactly where the 115 unresolved ids hid, so the mutation puts one back
    there rather than somewhere the original bug could not have occurred.
    """
    record = json.loads((CFG.records_dir / "fertility.json").read_text(encoding="utf-8"))
    forged = {
        **record,
        "conversational": {"ta": {"source": "cl100k_base|conv@pending-20260805T063758Z"}},
    }

    planted = [
        v["source"]
        for v in forged["conversational"].values()
        if isinstance(v, dict) and "pending-" in v.get("source", "")
    ]
    assert planted, "the mutation no longer plants a placeholder, so the guard proves nothing"


def test_nothing_derived_from_estimates_claims_to_be_measured():
    """INV-4, the direction that was being violated: a sum is as weak as its weakest input.

    `sourcing` declared `measured` over `committed_tokens`, which is 6.39T summed from catalogue
    sizes of which 24 are estimated and 121 unknown. The page tells the reader that mark means
    "somebody ran it". Nobody had. Correction X17.
    """
    bundle = _bundle()

    sizes = [(d.get("size_tokens") or {}).get("provenance") for d in bundle["datasets"]]
    assert "measured" not in sizes, (
        "a catalogue size is now measured — re-check whether the blocks below may claim it too"
    )

    for block in _DERIVED_FROM_CATALOGUE_SIZES:
        assert bundle[block]["provenance"] != "measured", (
            f"{block} claims measured; its token figures are sums of catalogue estimates"
        )


def test_the_derived_provenance_check_can_actually_fail():
    """Breaking X17 must fail: restore the blanket and the guard must object."""
    bundle = _bundle()
    forged = {**bundle, "sourcing": {**bundle["sourcing"], "provenance": "measured"}}

    assert forged["sourcing"]["provenance"] == "measured"
    with pytest.raises(AssertionError, match="sums of catalogue estimates"):
        for block in _DERIVED_FROM_CATALOGUE_SIZES:
            assert forged[block]["provenance"] != "measured", (
                f"{block} claims measured; its token figures are sums of catalogue estimates"
            )


def test_the_records_agree_about_any_model_they_both_describe():
    """Two registers describing one model must not describe it differently.

    `architectures.json` said Sarvam-30B was 30B where `scaling_reference.json` said 32B, and the
    model card says 32B; the same pair disagreed about the 105B's active parameters. Both are
    rendered on the page, in different chapters, so a reader comparing them saw two answers.
    Correction X19.
    """
    records = json.loads((WEB / "records.json").read_text(encoding="utf-8"))
    arch = {r["model"]: r for r in records.get("architectures", []) if r.get("params_total")}
    ref = {
        m["model"]: m
        for m in (records.get("scaling_reference") or {}).get("models", [])
        if m.get("params_total")
    }

    shared = set(arch) & set(ref)
    assert shared, "no model appears in both registers, so this guard is watching nothing"
    for model in sorted(shared):
        for field in ("params_total", "params_active"):
            if arch[model].get(field) is None or ref[model].get(field) is None:
                continue
            assert arch[model][field] == ref[model][field], (
                f"{model}: architectures says {arch[model][field]} for {field}, "
                f"scaling_reference says {ref[model][field]}"
            )


def test_counts_may_still_claim_measurement():
    """The correction must not over-swing: counting records in a catalogue we hold is a measurement.

    145 datasets is 145 datasets. Marking exact counts as estimates would be the same dishonesty
    pointing the other way, and would drain the mark of meaning.
    """
    bundle = _bundle()

    assert bundle["sourcing"]["counts"]["provenance"] == "measured"
    assert bundle["record_counts"]["provenance"] == "measured"
    assert bundle["grades"]["provenance"] == "measured"


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


# Indic prose for the tokenisation tests. Devanagari, Malayalam and Tamil, chosen because every one
# of these words carries vowel signs — the characters `\w` silently discards.
_HINDI = "भारत और पाकिस्तान के बीच"  # five words
_MALAYALAM = "ഇന്ത്യയും പാകിസ്ഥാനും തമ്മിലുള്ള"  # three words
_TAMIL_PROSE = (
    "இந்த தாக்குதல் இந்தியா மற்றும் பாகிஸ்தான் இடையேயான உறவில் "
    "பெரும் நெருக்கடியை ஏற்படுத்தியது என்று செய்தி நிறுவனம் தெரிவித்தது"
)


def test_indic_words_survive_tokenisation():
    r"""INV-1: a word is a word in every script, not only in the ones written without diacritics.

    Python's `\w` matches letters and digits but not combining marks, and every Indic vowel sign,
    virama and anusvara is one — so `\w+` split each word at every vowel sign and threw the sign
    away. 91% of the items this gate indexes are in Indic scripts, which made a "thirteen-word
    fingerprint" about five real words of consonant skeleton there. Correction X16.
    """
    assert normalise(_HINDI) == ["भारत", "और", "पाकिस्तान", "के", "बीच"]
    assert len(normalise(_MALAYALAM)) == 3
    # And the scripts that were never broken must stay unbroken.
    assert normalise("between India and Pakistan today") == [
        "between",
        "india",
        "and",
        "pakistan",
        "today",
    ]


def test_the_tokenisation_can_actually_fail():
    r"""Breaking X16 must fail: the old pattern shatters the same words.

    If `\w+` ever stops splitting these, this mutation has stopped proving anything and the test
    above is no longer evidence.
    """
    old = re.compile(r"\w+", re.UNICODE)

    assert len(old.findall(_HINDI.lower())) > len(normalise(_HINDI)), (
        "the old pattern no longer shatters Devanagari, so this proves nothing"
    )
    # The exact shape of the defect: three ordinary Malayalam words became a full 13-token window,
    # which is why unrelated prose collided with the index.
    assert len(old.findall(_MALAYALAM.lower())) == SHINGLE_N


def test_ordinary_indic_prose_does_not_collide_with_the_index():
    """INV-1, the false-positive direction: the gate must not delete innocent training text.

    A gate that drops clean documents is worse than no gate, because the loss is silent and lands
    on the scarcest tier in the mixture. Measured against 203,388 held-out FLORES-200 sentences the
    old tokeniser produced 5 such collisions, all Indic; this fixture is one of them.
    """
    index = build_attributed_index({"MILU": [_TAMIL_PROSE]})
    unrelated = (
        "ഇന്ത്യയും പാകിസ്ഥാനും തമ്മിലുള്ള ബന്ധത്തെ ആക്രമണം വലിയ രീതിയിൽ ബാധിച്ചു എന്ന് വാർത്താ ഏജൻസി റിപ്പോർട്ട് ചെയ്തു"
    )

    assert not find_collisions(unrelated, index), (
        "unrelated Indic prose collided with the eval index — the gate is deleting clean text"
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
