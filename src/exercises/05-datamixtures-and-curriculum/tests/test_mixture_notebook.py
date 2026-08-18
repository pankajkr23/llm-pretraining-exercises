"""The session notebook, held to the repo's notebook rules.

`AGENTS.md` requires every session to ship `notebooks/SNN-slug.ipynb` that imports the exercise's
package rather than re-implementing it, and that carries no committed outputs. Both are checkable,
so they are checked here rather than remembered.

The rule that matters most on this exercise is the first. A notebook that copied the arithmetic
would keep printing the old numbers after the inventory changed, and it would look right while
doing it — which is the same drift `test_mixture_spec_render.py` guards `SPEC.md` against.
"""

import json
from pathlib import Path

import pytest
from mixture import export

REPO_ROOT = export.EXERCISE_ROOT.parents[2]
NOTEBOOK = REPO_ROOT / "notebooks" / "S05-datamixtures-and-curriculum.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    """The committed notebook."""
    if not NOTEBOOK.exists():
        pytest.fail(f"the session notebook is missing at {NOTEBOOK}; AGENTS.md requires one")
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


def test_every_notebook_in_the_repo_follows_the_naming_rule():
    """The rule is repo-wide, so it is checked repo-wide rather than for this session alone."""
    for path in Path(NOTEBOOK).parent.glob("*.ipynb"):
        assert path.name[0] == "S" and path.name[1:3].isdigit(), (
            f"{path.name} does not start with a zero-padded session id (SNN-)"
        )


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
def test_the_notebook_runs_end_to_end() -> None:
    """Every code cell executes without raising.

    AGENTS.md: "a session's work is not done until its notebook runs the shipped code end to end."
    This proves only that — no cell raises. It does not check that any printed number is right;
    that is what the module tests and `test_mixture_spec_render.py` are for.
    """
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

    book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    book["cells"].append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ["raise RuntimeError('deliberate')"],
        }
    )
    broken = tmp_path / NOTEBOOK.name
    broken.write_text(json.dumps(book), encoding="utf-8")

    failures = _execute(broken)
    assert [(name, value) for _, name, value in failures] == [("RuntimeError", "deliberate")]
