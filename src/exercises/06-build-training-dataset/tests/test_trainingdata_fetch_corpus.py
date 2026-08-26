"""The corpus fetcher — the parts that must hold without touching the network.

A fetcher is mostly I/O, and I/O is mostly untestable offline. What *is* testable is the part that
decides **what may enter the corpus at all**: which lanes exist, which licences are acceptable, and
what happens to a row that declares something else. Those are the claims that would be expensive to
get wrong, because a corpus is not something you can un-download from a model.
"""

import importlib.util
from pathlib import Path

import pytest
from trainingdata import mixture

EXERCISE = Path(__file__).resolve().parents[1]
FETCHER = EXERCISE / "tools" / "fetch_corpus.py"


def _module():
    """Import the fetcher, which lives in `tools/` rather than the package.

    Returns:
        The imported module.
    """
    spec_ = importlib.util.spec_from_file_location("fetch_corpus", FETCHER)
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fetcher():
    """The fetcher module.

    Returns:
        The imported module.
    """
    return _module()


def test_the_fetcher_is_tracked() -> None:
    """**Unlike the notebook builder, this one must survive a clone.**

    `AGENTS.md`: a corpus needs a tracked way to fetch and licence-check it. A fetcher that only
    exists on one working tree means the corpus can never be reproduced or its licences re-checked.
    """
    assert FETCHER.is_file()
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(FETCHER.relative_to(EXERCISE.parents[2]))],
        cwd=EXERCISE.parents[2],
        capture_output=True,
    )
    assert tracked.returncode == 0, "fetch_corpus.py is not tracked by git"


# --- which lanes exist -----------------------------------------------------------------------


def test_every_source_belongs_to_a_funded_lane(fetcher) -> None:
    """A source for a lane the mixture does not fund would download text nothing can use."""
    for source in fetcher.SOURCES + fetcher.LOCAL_SOURCES:
        assert source.lane in mixture.FUNDED_LANES, f"{source.lane} is not a funded lane"


def test_every_funded_lane_has_at_least_one_source(fetcher) -> None:
    """**The failure mode `AGENTS.md` names: a missing input reads as passing.**

    Session 5's corpus silently falls back to three lanes when `data/proxy/` is absent, and a
    fresh clone reproduces a three-lane corpus with no error at all. A lane with no source here
    would fetch zero tokens and report a mixture computed over the lanes that happened to work.
    """
    covered = {s.lane for s in fetcher.SOURCES} | {s.lane for s in fetcher.LOCAL_SOURCES}
    missing = sorted(set(mixture.FUNDED_LANES) - covered)
    assert not missing, f"these funded lanes have no source: {missing}"


def test_the_retired_lane_has_no_source(fetcher) -> None:
    """`long_context` is a schedule over the other lanes, not a corpus.

    Giving it text would invent a lane session 5 retired, and double-count code.
    """
    lanes = {s.lane for s in fetcher.SOURCES} | {s.lane for s in fetcher.LOCAL_SOURCES}
    assert "long_context" not in lanes


def test_the_indic_lane_uses_only_scripts_the_tokenizer_can_read(fetcher) -> None:
    """**A lane can be present and still be unusable.**

    Sangraha covers many languages, and Bengali, Kannada, Gujarati and Tamil all measure above 80%
    `[UNK]` under the frozen vocabulary — they would pass every structural check and fail the 5%
    publication gate mid-build, after the download. Devanagari and Telugu measure 0.00-0.17%.
    """
    indic = [s for s in fetcher.SOURCES if s.lane == "indic"]
    assert {s.split for s in indic} == {"hin", "tel", "mai"}
    assert all(s.config == "verified" for s in indic), (
        "only the `verified` config carries Maithili; it exists in no other"
    )


# --- what may enter the corpus ----------------------------------------------------------------


def test_the_permissive_set_contains_no_copyleft(fetcher) -> None:
    """The whole point of the gate. A copyleft corpus is a licensing claim about the model's
    output that this repo is in no position to make."""
    forbidden = {"gpl", "agpl", "lgpl", "cc-by-sa", "cc-by-nc", "other", "unknown"}
    for allowed in fetcher.PERMISSIVE | fetcher.PERMISSIVE_CODE_FILES:
        assert not any(bad in allowed for bad in forbidden), f"{allowed} is not permissive"


def test_the_per_file_gate_is_narrower_than_the_dataset_gate(fetcher) -> None:
    """They answer different questions.

    The dataset licence covers the *packaging*; the per-file one covers somebody's actual source
    file. `odc-by` is a fine licence for a data collection and a meaningless one for a `.py`.
    """
    assert "odc-by" in fetcher.PERMISSIVE
    assert "odc-by" not in fetcher.PERMISSIVE_CODE_FILES


def test_a_copyleft_source_file_is_dropped(fetcher) -> None:
    """**The row-level check, which the dataset-level tag would wave straight through.**

    `codeparrot/github-code-clean` is Apache-2.0 as a packaging and mixes GPL/AGPL/LGPL files with
    permissive ones. Trusting the dataset tag would put copyleft source into the corpus.
    """
    code = next(s for s in fetcher.SOURCES if s.licence_column)
    assert fetcher._document({"code": "print(1)", "license": "gpl-3.0"}, code) is None
    assert fetcher._document({"code": "print(1)", "license": "agpl-3.0"}, code) is None
    assert fetcher._document({"code": "print(1)", "license": "mit"}, code) == ["print(1)"]


def test_a_row_with_no_licence_at_all_is_dropped(fetcher) -> None:
    """Absence is not permission. An unverifiable licence is not a permissive one."""
    code = next(s for s in fetcher.SOURCES if s.licence_column)
    assert fetcher._document({"code": "print(1)", "license": None}, code) is None
    assert fetcher._document({"code": "print(1)"}, code) is None


def test_a_row_missing_its_text_is_skipped_not_padded(fetcher) -> None:
    """A `None` rendered as the string "None" would put four junk tokens in the corpus."""
    web = next(s for s in fetcher.SOURCES if s.lane == "web")
    assert fetcher._document({"text": None}, web) is None
    assert fetcher._document({"text": ""}, web) is None


def test_multi_field_sources_concatenate_rather_than_pick_one(fetcher) -> None:
    """**The trap that silently fetches nothing.**

    `OpenR1-Math-220k` has no `text` field at all — the prose is in `problem` and `solution`.
    A fetcher pointed at `text` would return `None` for every row and report an empty lane.
    """
    reasoning = next(s for s in fetcher.SOURCES if s.lane == "reasoning")
    assert reasoning.fields == ("problem", "solution")
    assert "text" not in reasoning.fields
    rendered = fetcher._document({"problem": "2+2?", "solution": "4"}, reasoning)
    assert rendered == ["2+2?", "4"], (
        "the parts must be KEPT, not pre-joined — the boundary is unrecoverable afterwards, and "
        "81.9% of real reasoning documents contain more than one blank line"
    )


# --- sizing ------------------------------------------------------------------------------------


def test_the_token_count_includes_the_eos_the_shard_builder_will_add(fetcher) -> None:
    """A target that ignored it comes up short by one token per document.

    On the agentic lane that is ~16,000 tokens — 7% of its whole budget.
    """

    class _Fake:
        def encode(self, text):
            class _Encoded:
                ids = [1, 2, 3]

            return _Encoded()

    tokens, unknown = fetcher._measure("anything", _Fake())
    assert tokens == 4, "the EOS terminator was not counted"
    assert unknown == 0


def test_unknown_tokens_are_counted_from_id_zero(fetcher) -> None:
    """`[UNK]` is id 0 in the frozen tokenizer, so the share is computable straight from ids.

    It is also a perfectly legal shard token, so nothing but the manifest ever notices a lane that
    is mostly unknown — which is why the fetcher has to measure it.
    """

    class _Fake:
        def encode(self, text):
            class _Encoded:
                ids = [0, 5, 0, 7]

            return _Encoded()

    tokens, unknown = fetcher._measure("anything", _Fake())
    assert (tokens, unknown) == (5, 2)


def test_the_targets_come_from_the_mixture_not_from_a_second_copy(fetcher) -> None:
    """If the fetcher held its own numbers, the corpus and the compliance report would disagree
    and nothing would say so."""
    source = FETCHER.read_text()
    assert "mixture.token_targets" in source
    for share in ("0.32", "0.28", "0.18", "0.12", "0.08"):
        assert f"= {share}" not in source, f"the fetcher restates the {share} lane share"
