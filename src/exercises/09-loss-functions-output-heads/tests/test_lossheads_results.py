"""`RESULTS.md` must still describe the runs in `results/`, and the README must not restate them.

**A generated document that has drifted from its source is worse than no document**, because it
reads as maintained. This regenerates `RESULTS.md` in memory and fails when the tracked copy
differs — so a stale figure is a red test rather than something a reader eventually notices.

It needs no `torch`: it reads JSON and renders a string. That is deliberate, because the ordinary CI
job is where a stale document should be caught, and the `train` job is not guaranteed to run on
every change to a document.
"""

import json
import sys
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"
sys.path.insert(0, str(EXERCISE / "tools"))

pytest.importorskip("render_results", reason="tools/render_results.py is not importable from here")
import render_results  # noqa: E402


def _results_exist() -> bool:
    return (RESULTS / "harness.json").is_file() and (RESULTS / "training.json").is_file()


requires_results = pytest.mark.skipif(
    not _results_exist(), reason="results/ has not been generated on this checkout"
)


@requires_results
def test_results_md_is_current() -> None:
    """The whole point. Regenerate and compare, byte for byte."""
    harness = json.loads((RESULTS / "harness.json").read_text())
    training = json.loads((RESULTS / "training.json").read_text())
    expected = render_results.render(harness, training)
    actual = (EXERCISE / "RESULTS.md").read_text()

    assert actual == expected, (
        "RESULTS.md no longer matches results/*.json. Regenerate it:\n"
        "  uv run python src/exercises/09-loss-functions-output-heads/tools/render_results.py"
    )


@requires_results
def test_the_verdict_words_come_from_the_data_and_not_from_the_author() -> None:
    """The twin, and the more important half.

    Byte-equality would pass on a template with the word "above" hard-coded. This flips the data so
    the finding reverses, and asserts the rendered document reverses with it. A document that reads
    the same either way is a document telling a story rather than reporting a run.
    """
    harness = json.loads((RESULTS / "harness.json").read_text())
    training = json.loads((RESULTS / "training.json").read_text())

    real = render_results.render(harness, training)
    assert "sits **above**" in real

    flipped = json.loads(json.dumps(training))
    flipped["summary"]["further_head_is_harder"] = False
    flipped["summary"]["broken_shift_is_lower"] = False
    inverted = render_results.render(harness, flipped)

    assert "sits **below**" in inverted, "the verdict word is hard-coded, not read from the run"
    assert "**higher**" in inverted


@requires_results
def test_the_memory_ratio_is_only_reported_when_the_losses_agree() -> None:
    """A ratio between two different computations is not a saving, and must not read as one."""
    harness = json.loads((RESULTS / "harness.json").read_text())
    training = json.loads((RESULTS / "training.json").read_text())

    broken = json.loads(json.dumps(harness))
    broken["item_7_memory"]["losses_agree"] = False
    rendered = render_results.render(broken, training)

    assert "the ratio below means nothing" in rendered, (
        "a disagreement between the two loss paths rendered as though nothing were wrong"
    )


def test_the_readme_points_at_results_rather_than_restating_them() -> None:
    """Two documents, two jobs. The README argues; `RESULTS.md` is the evidence it points to.

    Restating a figure in the README creates a second place to keep it correct, and the second copy
    is the one that drifts. This does not run the renderer, so it holds on a fresh clone too.
    """
    readme = (EXERCISE / "README.md").read_text()
    assert "RESULTS.md" in readme, "the README does not send a reader to the measured evidence"
