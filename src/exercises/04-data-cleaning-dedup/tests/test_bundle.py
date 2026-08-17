"""The bundle the page reads, and the invariants that hold it honest.

These run against `web/data.json` — the tracked artifact — rather than `artifacts/run.json`, which
does not exist in CI. When the bundle has not been built the suite skips rather than fails, so a
fresh checkout still works; the cost is that these guards protect the build only after someone has
run it, which is why `test_the_yield_chain_can_actually_fail` exists to prove they bite when they
do run.
"""

import json

import pytest
from datacleaning import pipeline
from datacleaning.config import Config
from datacleaning.records import Figure, StageStat

CFG = Config()
DATA_JSON = CFG.web_dir / "data.json"

pytestmark = pytest.mark.skipif(
    not DATA_JSON.exists(),
    reason="bundle not built; run `uv run python -m datacleaning --profile lite` first",
)


@pytest.fixture(scope="module")
def bundle() -> dict:
    return json.loads(DATA_JSON.read_text(encoding="utf-8"))


# ---- size and encoding ------------------------------------------------------------------------


def test_the_bundle_is_within_its_size_budget():
    """A reader downloads this before seeing anything. Prose belongs in chapters.js."""
    size_kb = DATA_JSON.stat().st_size / 1024
    assert size_kb <= CFG.data_json_budget_kb, (
        f"data.json is {size_kb:.1f} KB, over the {CFG.data_json_budget_kb:.0f} KB budget"
    )


def test_the_bundle_stores_indic_text_as_utf8_not_escapes():
    """`ensure_ascii=True` would triple the byte cost of exactly the fields carrying Indic text."""
    raw = DATA_JSON.read_text(encoding="utf-8")
    assert "\\u0900" not in raw and "\\u0c00" not in raw


# ---- the yield chain --------------------------------------------------------------------------


def test_every_stage_hands_its_documents_to_the_next(bundle):
    """A stage that loses documents without reporting them would break this chain."""
    stages = bundle["stages"]
    assert stages, "no stages in the bundle"
    for earlier, later in zip(stages, stages[1:], strict=False):
        assert later["docs_in"] == earlier["docs_out"], (
            f"{later['id']} received {later['docs_in']} but {earlier['id']} emitted "
            f"{earlier['docs_out']}"
        )


def test_the_published_descent_matches_the_stage_records(bundle):
    """The page renders `yield`; it must not be able to disagree with `stages`."""
    assert bundle["yield"]["docs"] == [s["docs_out"] for s in bundle["stages"]]
    assert bundle["yield"]["labels"] == [s["name"] for s in bundle["stages"]]


def test_the_yield_chain_can_actually_fail():
    """Break the chain on purpose and confirm the check above notices.

    Without this, the two tests above pass trivially on a pipeline where every stage is a
    pass-through — which is exactly what the pipeline is right now.
    """
    figure = Figure(1, "tokens", "measured", "test")
    good = StageStat("1", "a", "A", True, 10, 8, figure, figure)
    broken = StageStat("2", "b", "B", True, 10, 5, figure, figure)  # claims 10 in, but A emitted 8

    stages = [s.as_json() for s in (good, broken)]
    mismatched = [
        (later["docs_in"], earlier["docs_out"])
        for earlier, later in zip(stages, stages[1:], strict=False)
        if later["docs_in"] != earlier["docs_out"]
    ]
    assert mismatched, "a deliberately broken chain was not detected"


# ---- provenance -------------------------------------------------------------------------------


def test_no_token_count_is_published_without_naming_its_tokenizer(bundle):
    """A token count without a tokenizer is not a fact about a corpus (DECISIONS.md §D3)."""
    for stage in bundle["stages"]:
        for key in ("tokens_in", "tokens_out"):
            figure = stage[key]
            assert set(figure) >= {"value", "unit", "provenance", "source"}
            if figure["value"] is not None:
                assert figure["provenance"] in {"measured", "derived", "inherited"}
                assert figure["source"], f"{stage['id']}.{key} has no source"


def test_a_stage_that_is_not_implemented_says_so(bundle):
    """A pass-through must never be mistaken for a stage that found nothing."""
    for stage in bundle["stages"]:
        if not stage["real"]:
            assert "not implemented" in stage["note"].lower()


def test_the_run_id_is_derived_from_content_rather_than_the_clock(bundle):
    """Identifiers that changed on every run is one of the three defects the audit found."""
    assert bundle["run"]["run_id"].startswith("s04-")
    for key in ("config_hash", "script_hash", "content_hash"):
        assert bundle["run"][key].startswith("sha256:")


def test_the_out_of_vocabulary_probe_is_excluded_from_the_budget(bundle):
    """Its counts are unusable by construction; budgeting them would be the demonstrated error."""
    probe = [c for c in bundle["corpora"] if not c["counts_toward_budget"]]
    assert probe, "the out-of-vocabulary probe is missing from the bundle"
    assert all(c["key"] == "oov" for c in probe)


def test_the_strategy_list_covers_the_sessions_stages(bundle):
    """Eight strategies, rendered as nine rows: Extract is inherited and 2b is never numbered."""
    ids = [s["id"] for s in bundle["strategies"]]
    assert ids == [s[1] for s in pipeline.STAGES]
    assert "formats" in ids and "extract" in ids
