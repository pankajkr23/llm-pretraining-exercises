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
        f"{memory['spread']:.2f}": "the memory ratio's measured spread",
    }
    missing = {value: what for value, what in expected.items() if value not in readme}
    assert not missing, (
        "the README quotes figures that no longer match the runs, or has stopped quoting them:\n"
        + "\n".join(f"  {value} — {what}" for value, what in missing.items())
    )


@requires_results
def test_no_ratio_the_readme_states_is_absent_from_the_run() -> None:
    """Presence is not agreement, and this is the half the test above cannot do.

    **The hole was live and cost a wrong published number.** The check above asks whether the right
    value appears *somewhere*. A README can satisfy that and still carry a second, wrong copy of the
    same quantity — which is exactly what happened: the noise floor was stated correctly in one
    paragraph and as `0.69` forty-eight lines later, against a measured `0.177`. Both sentences were
    in the same file, one of them was fiction, and every test was green.

    So this asks the opposite question: of every `N.NNx` the README states, is each one a ratio this
    run actually produced? A wrong figure fails whether or not the right one is also present.
    """
    import re

    _, _, sensitivity = _load()
    readme = (EXERCISE / "README.md").read_text()
    memory = sensitivity["memory"]

    known = {round(r, 2) for r in memory["ratios"]}
    known |= {round(memory["min"], 2), round(memory["max"], 2)}
    known |= {round((memory["min"] + memory["max"]) / 2, 2)}

    stated = {float(m) for m in re.findall(r"(\d+\.\d\d)x", readme)}
    unknown = sorted(v for v in stated if v not in known)
    assert not unknown, (
        f"the README states {', '.join(f'{v}x' for v in unknown)}, which no measurement in "
        f"results/sensitivity.json produced. Measured: {sorted(known)}. A ratio quoted to two "
        "decimals is a claim about a run, so it has to come from one."
    )


@requires_results
def test_every_spread_the_readme_states_is_the_measured_spread() -> None:
    """Every number the README calls a spread must be the spread. Scoped, not global.

    The property is *"a figure introduced by a word is the figure that word names"*, and it is
    checked here for the one quantity this README has already got wrong. Scoping to a keyword is
    what lets it be exact: a global "every decimal must be real" check would fire on epoch counts,
    percentages and section numbers, and a guard that noisy gets weakened rather than obeyed.
    """
    import re

    _, _, sensitivity = _load()
    readme = (EXERCISE / "README.md").read_text()
    measured = round(sensitivity["memory"]["spread"], 2)

    # "spread OF n", not "spread from a to b" -- the second states a bound, not a spread, and a
    # regex greedy enough to catch both cannot tell which quantity it has. The preposition is the
    # whole distinction, so the guard uses it rather than pretending to a generality it lacks.
    stated = [float(m) for m in re.findall(r"spread(?:\s+\w+){0,2}?\s+of\s+(\d+\.\d+)", readme)]
    wrong = [v for v in stated if round(v, 2) != measured]
    assert stated, (
        "the README no longer states the memory ratio's spread. It is the evidence that 'about 9x' "
        "is the honest precision, so dropping it removes the reason for the hedge."
    )
    assert not wrong, (
        f"the README calls {wrong} a spread; the run measured {measured}. This is the exact "
        "failure that shipped once — a correct figure in one paragraph and a fictional one in "
        "another, with the presence check above satisfied by the correct copy."
    )


@pytest.mark.parametrize("document", ["README.md", "CLAUDE.md"])
def test_every_module_is_named_in_the_documents_that_list_modules(document: str) -> None:
    """A new module is not done until every list that names modules includes it.

    Copied from exercise 06, which `AGENTS.md` asks any exercise past a handful of modules to copy.
    This one has eleven. `provenance.py` was the twelfth thing shipped here and the first thing that
    would have gone unlisted — the README's layout table and this exercise's `CLAUDE.md` both
    enumerate modules, and neither is generated.

    **Its limit is worth stating, because the guard reads stronger than it is.** It checks the
    *document*, not the *list*: a module named once anywhere in the prose satisfies it while the
    table a reader actually follows stays wrong. That gap has already cost exercise 05 a published
    page whose figures contradicted its own tool.
    """
    text = (EXERCISE / document).read_text()
    modules = sorted(
        p.name for p in (EXERCISE / "src" / "lossheads").glob("*.py") if p.name != "__init__.py"
    )
    missing = [name for name in modules if name not in text]
    assert not missing, (
        f"{document} does not mention {missing}. Add it to the table a reader follows, not "
        "merely to a sentence somewhere — this test cannot tell the difference."
    )
