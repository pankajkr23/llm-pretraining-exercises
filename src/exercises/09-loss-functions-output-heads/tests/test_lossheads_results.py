"""`RESULTS.md` must still describe the runs in `results/`, and the README must not restate them.

**A generated document that has drifted from its source is worse than no document**, because it
reads as maintained. This regenerates `RESULTS.md` in memory and fails when the tracked copy
differs — so a stale figure is a red test rather than something a reader eventually notices.

It needs no `torch`: it reads JSON and renders a string. That is deliberate, because the ordinary CI
job is where a stale document should be caught, and the `train` job is not guaranteed to run on
every change to a document.
"""

import importlib.util
import json
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"


def _load_renderer():
    """Import `tools/render_results.py` by path, without touching `sys.path`.

    The first version did `sys.path.insert(0, tools/)` and then `pytest.importorskip`. Both were
    wrong. The insert was never undone, so a generically-named `render_results` — and the gitignored
    `build_notebook.py` beside it — sat at the front of `sys.path` for every later import. And
    the importorskip could never fire, because the insert two lines above guaranteed the import
    would succeed; all it did was register this file in `OPTIONAL_DEPENDENCY_GATES`'s eyes as
    gated on an optional dependency, which turned the repo-wide coverage guard red.

    `render_results` is a tracked file in this exercise. If it stops importing, that is a defect and
    the test should fail, not skip.
    """
    spec = importlib.util.spec_from_file_location(
        "lossheads_render_results", EXERCISE / "tools" / "render_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_results = _load_renderer()


_NEEDED = ("harness.json", "training.json", "sensitivity.json")


def _results_exist() -> bool:
    return all((RESULTS / name).is_file() for name in _NEEDED)


def _load() -> tuple[dict, dict, dict]:
    return tuple(json.loads((RESULTS / name).read_text()) for name in _NEEDED)


requires_results = pytest.mark.skipif(
    not _results_exist(), reason="results/ has not been generated on this checkout"
)


@requires_results
def test_results_md_is_current() -> None:
    """The whole point. Regenerate and compare, byte for byte."""
    expected = render_results.render(*_load())
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
    harness, training, sensitivity = _load()

    real = render_results.render(harness, training, sensitivity)
    assert "sits **above**" in real

    flipped = json.loads(json.dumps(training))
    flipped["summary"]["further_head_is_harder"] = False
    flipped["summary"]["broken_shift_is_lower"] = False
    inverted = render_results.render(harness, flipped, sensitivity)

    assert "sits **below**" in inverted, "the verdict word is hard-coded, not read from the run"
    assert "**higher**" in inverted


@requires_results
def test_the_memory_ratio_is_only_reported_when_the_losses_agree() -> None:
    """A ratio between two different computations is not a saving, and must not read as one."""
    harness, training, sensitivity = _load()

    broken = json.loads(json.dumps(harness))
    broken["item_7_memory"]["losses_agree"] = False
    rendered = render_results.render(broken, training, sensitivity)

    assert "the ratio below means nothing" in rendered, (
        "a disagreement between the two loss paths rendered as though nothing were wrong"
    )


def test_the_readme_sends_the_reader_to_the_measured_evidence() -> None:
    """Two documents, two jobs. The README argues; `RESULTS.md` is the evidence it points to."""
    readme = (EXERCISE / "README.md").read_text()
    assert "RESULTS.md" in readme, "the README does not send a reader to the measured evidence"


@requires_results
def test_every_figure_the_readme_quotes_matches_the_run_it_came_from() -> None:
    """The README **does** restate figures, and this is what keeps them true.

    An earlier version of this test was named for a property it did not check — it asserted only
    that the string "RESULTS.md" appeared, while the README quoted eight numbers none of which were
    verified. A test named for a property it does not test is worse than no test, because it reads
    as coverage.

    The honest fix is not to strip the numbers out of the README: a document that argues needs its
    headline figures in it. It is to check the ones it quotes against the runs they came from.
    """
    _, training, sensitivity = _load()
    readme = (EXERCISE / "README.md").read_text()
    summary = training["summary"]
    memory = sensitivity["memory"]

    expected = {
        f"{summary['final_broken_shift']:.2f}": "the broken shift's final loss",
        f"{summary['final_correct_shift']:.2f}": "the correct shift's final loss",
        f"{summary['steps_where_further_head_was_higher']} of {summary['total_steps']}": (
            "how often the further head was higher"
        ),
        f"{memory['min']:.2f}x": "the lowest memory ratio measured",
        f"{memory['max']:.2f}x": "the highest memory ratio measured",
    }
    missing = {value: what for value, what in expected.items() if value not in readme}
    assert not missing, (
        "the README quotes figures that no longer match the runs, or has stopped quoting them:\n"
        + "\n".join(f"  {value} — {what}" for value, what in missing.items())
    )
