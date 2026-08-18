"""The session notebook, held to the repo's notebook rules.

`AGENTS.md` requires every session to ship `notebooks/SNN-slug.ipynb` that imports the exercise's
package rather than re-implementing it, and that carries no committed outputs. Both are checkable,
so they are checked here rather than remembered.

The rule that matters most on this exercise is the first. A notebook that copied the arithmetic
would keep printing the old numbers after the inventory changed, and it would look right while
doing it — which is the same drift `test_mixture_spec_render.py` guards `SPEC.md` against.
"""

import json
import subprocess
from pathlib import Path

import pytest
from mixture import export

REPO_ROOT = export.EXERCISE_ROOT.parents[2]
NOTEBOOKS = REPO_ROOT / "notebooks"
NOTEBOOK = NOTEBOOKS / "S05-datamixtures-and-curriculum.ipynb"
# Tracked, stdlib-only, and the only notebook a fresh clone or CI ever sees.
SAMPLE = NOTEBOOKS / "hello.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    """The session notebook, when there is one.

    Skips rather than fails when absent. Session notebooks are gitignored, so a fresh clone and CI
    genuinely do not have one, and a hard failure there would be reporting the design as a defect.
    The rules below therefore protect the author's checkout, not the pipeline — which is why
    `notebooks/hello.ipynb` is tracked and executed instead: see `test_the_sample_notebook_runs`.
    """
    if not NOTEBOOK.exists():
        pytest.skip(f"no session notebook at {NOTEBOOK}; they are local-only and gitignored")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(nb: dict) -> list[str]:
    """Every code cell's source."""
    return ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]


def _text(nb: dict) -> str:
    """Every cell's source, of any type."""
    return "\n".join("".join(cell["source"]) for cell in nb["cells"])


def test_the_notebook_is_valid_and_substantial(nb):
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 20
    assert len(_code(nb)) > 10


def test_the_notebook_has_no_committed_outputs(nb):
    """Committed outputs bloat diffs and go stale silently."""
    offenders = [
        index
        for index, cell in enumerate(nb["cells"])
        if cell["cell_type"] == "code"
        and (cell.get("outputs") or cell.get("execution_count") is not None)
    ]
    assert not offenders, f"cells {offenders} carry committed outputs; clear them before committing"


def test_the_output_check_can_actually_fail():
    """Without this, the guard above would pass against a notebook with no code cells at all."""
    fake = {
        "cells": [
            {"cell_type": "code", "source": ["print(1)"], "outputs": [{"text": "1"}]},
            {"cell_type": "code", "source": ["print(2)"], "outputs": []},
        ]
    }
    offenders = [
        index
        for index, cell in enumerate(fake["cells"])
        if cell["cell_type"] == "code"
        and (cell.get("outputs") or cell.get("execution_count") is not None)
    ]
    assert offenders == [0], "a notebook with a committed output was not detected"


def test_the_notebook_imports_the_package_rather_than_reimplementing_it(nb):
    """The rule that stops the notebook drifting from what ships."""
    source = "\n".join(_code(nb))
    assert "from mixture import" in source
    for module in ("lanes", "supply", "inventory", "curriculum", "proxy", "checks"):
        assert module in source, f"the notebook never touches {module}"

    # A notebook that redefined the arithmetic would carry its own definitions of it.
    for forbidden in (
        "def worth_tokens",
        "def evaluate_lane",
        "class Config",
        "WORTH_CEILING_MULTIPLE =",
    ):
        assert forbidden not in source, "the notebook must call the package, not redefine it"


def test_the_notebook_can_detect_colab_without_crashing_off_colab(nb):
    """`importlib.util.find_spec('google.colab')` raises off-Colab instead of returning None.

    That bug failed on cell 1 for every local reader of the Session 4 notebook. Comments are
    stripped first, because this notebook *explains* the bug in a comment and matching the
    explanation instead of the code would fail against the very cell that fixes it.
    """
    lines = [line for line in "\n".join(_code(nb)).splitlines() if not line.strip().startswith("#")]
    source = "\n".join(lines)
    assert "find_spec('google.colab')" not in source
    assert "except ImportError" in source


def test_the_notebook_carries_a_colab_badge_pointing_at_itself(nb):
    text = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "markdown"
    )
    assert "colab.research.google.com" in text
    assert NOTEBOOK.name in text, "the badge must point at this notebook, not another"


def test_the_notebook_is_named_with_its_session_id():
    """`AGENTS.md`: zero-padded session id first, so lexical sort equals session order."""
    assert NOTEBOOK.name.startswith("S05-")
    assert NOTEBOOK.suffix == ".ipynb"


def test_the_notebook_covers_all_seven_assignment_items(nb):
    """A notebook that skipped an item would teach an incomplete version of the deliverable."""
    text = _text(nb).lower()
    for topic in (
        "indic",
        "agentic",
        "reasoning",
        "long-context",
        "protected floor",
        "anneal reserve",
        "difficulty band",
        "proxy",
    ):
        assert topic in text, f"the notebook never covers: {topic}"


def test_the_notebook_shows_a_guard_failing(nb):
    """The notebook's payoff, and the repo's rule made visible to a reader.

    A reader who only ever sees "0 errors" has no way to tell a working checker from a broken one.
    """
    source = "\n".join(_code(nb))
    assert "check_floor" in source or "check_within_supply" in source
    assert "checks.main()" in source


def test_the_notebook_declares_the_stand_ins_it_uses(nb):
    """Exercise 04's rule: declare a stand-in, never publish an accuracy for it."""
    text = _text(nb)
    assert "authored illustrations" in text, "the difficulty-band examples must be declared"
    assert "UNMEASURED" in text or "unmeasured" in text, "the throughput refusal must be visible"


def test_every_session_notebook_follows_the_naming_rule():
    """The rule is repo-wide, so it is checked repo-wide rather than for this session alone.

    `hello.ipynb` is exempt by name. It is not a session notebook — it is the tracked sample, and
    the `SNN-` rule exists so that lexical sort equals session order, which a sample has no part in.
    """
    for path in NOTEBOOKS.glob("*.ipynb"):
        if path.name == SAMPLE.name:
            continue
        assert path.name[0] == "S" and path.name[1:3].isdigit(), (
            f"{path.name} does not start with a zero-padded session id (SNN-)"
        )


def test_the_sample_notebook_is_present_and_tracked():
    """The sample is the only notebook a fresh clone gets, so its absence must be loud.

    Session notebooks are gitignored, which means every rule in this file that reads one skips in
    CI. The sample is what stops that from adding up to no coverage at all — so it is checked for
    existence, for being tracked, and (below) for actually running.
    """
    assert SAMPLE.exists(), f"the tracked sample notebook is missing at {SAMPLE}"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(SAMPLE)],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    assert tracked.returncode == 0, f"{SAMPLE.name} exists but is not tracked; CI will not get it"

    book = json.loads(SAMPLE.read_text(encoding="utf-8"))
    dirty = [i for i, c in enumerate(book["cells"]) if c.get("outputs") or c.get("execution_count")]
    assert not dirty, f"the sample carries committed outputs in cells {dirty}"


# ---- it actually runs -------------------------------------------------------------------------

# The structural tests above read the notebook. None of them can tell you a cell raises, which is
# the one failure a reader meets first and the exercise's own CLAUDE.md names as the gap. Executing
# it is the only check that closes it.


def _execute(path: Path) -> list[tuple[int, str, str]]:
    """Run every code cell in `path` and return `(index, exception, message)` for those that raise.

    `allow_errors=True` so one bad cell does not hide the state of the cells after it: a reader
    fixing a notebook wants the whole list, not the first entry. Outputs are never written back —
    the committed file stays output-free, which `test_the_notebook_carries_no_outputs` enforces.
    """
    import nbformat
    from nbclient import NotebookClient

    book = nbformat.read(path, as_version=4)
    NotebookClient(
        book,
        timeout=600,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()

    unrun = [
        i
        for i, cell in enumerate(book.cells)
        if cell.cell_type == "code" and cell.source.strip() and not cell.get("execution_count")
    ]
    assert not unrun, f"cells {unrun} never ran; a notebook that no-ops would pass silently"

    return [
        (i, out.ename, out.evalue)
        for i, cell in enumerate(book.cells)
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.output_type == "error"
    ]


@pytest.mark.integration
def test_the_sample_notebook_runs() -> None:
    """The tracked sample executes top to bottom.

    This is the one execution guard that runs in CI, because it is the one notebook CI has. It
    proves the harness — that a notebook in this repo opens, runs every cell, and finishes — which
    is exactly what the session notebooks stop proving once they are untracked.
    """
    pytest.importorskip("nbclient", reason="nbclient is not installed")
    pytest.importorskip("ipykernel", reason="no kernel to run the notebook in")
    failures = _execute(SAMPLE)
    assert not failures, "the sample notebook raised: " + "; ".join(
        f"cell {i}: {name}: {value}" for i, name, value in failures
    )


@pytest.mark.integration
def test_the_session_notebook_runs_end_to_end() -> None:
    """Every code cell executes without raising.

    AGENTS.md: "a session's work is not done until its notebook runs the shipped code end to end."
    This proves only that — no cell raises. It does not check that any printed number is right;
    that is what the module tests and `test_mixture_spec_render.py` are for.
    """
    if not NOTEBOOK.exists():
        pytest.skip("no session notebook; they are local-only and gitignored")
    pytest.importorskip("nbclient", reason="nbclient is not installed")
    pytest.importorskip("ipykernel", reason="no kernel to run the notebook in")
    failures = _execute(NOTEBOOK)
    assert not failures, "cells raised: " + "; ".join(
        f"cell {i}: {name}: {value}" for i, name, value in failures
    )


@pytest.mark.integration
def test_a_raising_cell_is_actually_caught(tmp_path: Path) -> None:
    """The twin. A runner that reported success on a broken notebook would be worse than none.

    Appending a cell that raises is the cheapest way to be sure the check above is watching, and
    the repo's rule is that no guard is trusted until it has been seen to fail.
    """
    pytest.importorskip("nbclient", reason="nbclient is not installed")
    pytest.importorskip("ipykernel", reason="no kernel to run the notebook in")

    book = json.loads(SAMPLE.read_text(encoding="utf-8"))
    book["cells"].append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ["raise RuntimeError('deliberate')"],
        }
    )
    broken = tmp_path / SAMPLE.name
    broken.write_text(json.dumps(book), encoding="utf-8")

    failures = _execute(broken)
    assert [(name, value) for _, name, value in failures] == [("RuntimeError", "deliberate")]
