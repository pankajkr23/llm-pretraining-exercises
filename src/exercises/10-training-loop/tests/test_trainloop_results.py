"""`RESULTS.md` must still describe the run, and the README's own figures must still match it.

**The second half exists because the first half is not enough, and this exercise proved it.** Its
`RESULTS.md` was generated correctly and its README quoted `27.89%` — a figure from an earlier run,
higher than the true `27.64%`, sitting in a document whose headline is that a number was caught
being wrong in the flattering direction. Nothing was red. A reader found it by doing arithmetic.

Exercise 09 hit the same class of defect and fixed it the same way. The lesson is not "stop putting
numbers in the README" — a document that argues needs its headline figures — it is that a figure
quoted anywhere is a figure something has to check.

Needs no `torch`: it reads JSON and renders a string. Deliberate, because a stale document should be
caught in the ordinary CI job rather than only where the `train` extra is installed.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"


def _load_renderer():
    """Import `tools/render_results.py` by path, without touching `sys.path`.

    Exercise 09 did this with a `sys.path.insert` plus a `pytest.importorskip`. Both were wrong: the
    insert left a generically-named module at the front of the path for every later import, and the
    importorskip could never fire — but did register the file, repo-wide, as gated on an optional
    dependency, which turned the CI coverage guard red.
    """
    spec = importlib.util.spec_from_file_location(
        "trainloop_render_results", EXERCISE / "tools" / "render_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_results = _load_renderer()

requires_results = pytest.mark.skipif(
    not (RESULTS / "run.json").is_file(),
    reason="results/run.json has not been generated on this checkout",
)


def _run() -> dict:
    return json.loads((RESULTS / "run.json").read_text())


@requires_results
def test_results_md_is_current() -> None:
    """Regenerate and compare, byte for byte."""
    expected = render_results.render(_run())
    actual = (EXERCISE / "RESULTS.md").read_text()
    assert actual == expected, (
        "RESULTS.md no longer matches results/run.json. Regenerate it:\n"
        "  uv run python src/exercises/10-training-loop/tools/render_results.py"
    )


@requires_results
def test_every_figure_the_readme_quotes_matches_the_run_it_came_from() -> None:
    """The guard this exercise needed and did not have.

    Its README quoted an MFU of 27.89% while the generated document said 27.64% — a stale figure
    from an earlier run, and the higher of the two. Byte-equality on `RESULTS.md` cannot see that,
    because the README is not generated.
    """
    run = _run()
    readme = (EXERCISE / "README.md").read_text()
    five, three = run["item_5_mfu"], run["item_3_accumulation"]
    two = run["item_2_gradient"]

    expected = {
        f"{three['relative_gap']:.1%}": "the accumulation gap on the worked arithmetic",
        f"{three['curves']['final_gap']:.4f}": "the accumulation gap on the real run",
        f"{three['curves']['final_correct']:.4f}": "the correct reduction's final loss",
        f"{two['best_matching_digits']:.1f}": "the gradient check's matching decimal digits",
    }
    missing = {value: what for value, what in expected.items() if value not in readme}
    assert not missing, (
        "the README quotes figures that no longer match the run, or has stopped quoting them:\n"
        + "\n".join(f"  {value} — {what}" for value, what in missing.items())
    )

    # MFU is checked to a TOLERANCE, and the reason is worth stating: its numerator is fixed and
    # its denominator is a wall clock on a shared machine, so it moves by a few tenths of a point
    # between identical runs. Asserting it exactly would make this guard fail for a reason that has
    # nothing to do with the document being wrong — and a guard that cries wolf gets deleted.
    quoted = re.findall(r"\*\*(\d+\.\d\d)%\*\*", readme)
    assert quoted, "the README no longer quotes an MFU figure at all"
    assert any(abs(float(q) - five["mfu"] * 100) < 1.0 for q in quoted), (
        f"the README's MFU figures {quoted} are all more than a point away from the run's "
        f"{five['mfu']:.2%}, which is far outside the run-to-run spread"
    )


@requires_results
def test_the_readme_never_quotes_a_figure_from_an_earlier_run() -> None:
    """The twin, and the specific failure that happened.

    A stale figure is not the absence of a correct one — both can sit in the same document. This
    asserts the wrong value is gone rather than that the right one is present.
    """
    readme = (EXERCISE / "README.md").read_text()
    assert "27.89" not in readme, (
        "27.89% is a figure from an earlier run of this exercise. If MFU genuinely measures that "
        "now, this guard should be updated with the reason — not deleted."
    )


@requires_results
def test_a_verdict_word_is_read_from_the_run_and_not_hard_coded() -> None:
    """Byte-equality would pass on a template with the conclusion typed into it.

    So the data is flipped and the document's *conclusion* must flip with it.
    """
    run = _run()
    real = render_results.render(run)
    assert "reads **higher**" in real

    flipped = json.loads(json.dumps(run))
    flipped["item_3_accumulation"]["curves"]["wrong_reads_higher"] = False
    assert "reads **lower**" in render_results.render(flipped), (
        "the verdict word is hard-coded into the template, not read from the run"
    )


@requires_results
def test_the_document_reports_no_qualifying_step_when_the_run_found_none() -> None:
    """Item 4's empty case must be reachable in the document, not only in the search.

    A section that can only render a finding will render one whatever the data says.
    """
    run = _run()
    empty = json.loads(json.dumps(run))
    empty["item_4_grad_norm"]["found"] = []
    empty["item_4_grad_norm"]["count"] = 0
    rendered = render_results.render(empty)

    assert "No step qualified" in rendered
    assert "that is the result" in rendered


def test_the_readme_sends_the_reader_to_the_measured_evidence() -> None:
    """Two documents, two jobs. The README argues; `RESULTS.md` is the evidence it points to."""
    readme = (EXERCISE / "README.md").read_text()
    assert "RESULTS.md" in readme
