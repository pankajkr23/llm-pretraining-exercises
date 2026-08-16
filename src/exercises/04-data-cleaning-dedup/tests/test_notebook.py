"""The session notebook, held to the repo's notebook rules.

`AGENTS.md` requires every session to ship `notebooks/SNN-slug.ipynb` that imports the exercise's
package rather than re-implementing it, and that carries no committed outputs. Both rules are
checkable, so they are checked here rather than remembered.

The outputs rule is not cosmetic on this exercise. Later stages scrub PII from real Stack Exchange
text; executing those cells and committing the result would bake real email addresses into a
tracked file. `test_the_notebook_has_no_committed_outputs` is the guard that stops it, and its twin
proves the guard can fire.
"""

import json
from pathlib import Path

import pytest
from datacleaning import pipeline
from datacleaning.config import EXERCISE_ROOT

# EXERCISE_ROOT is …/src/exercises/04-data-cleaning-dedup, so the repo root is three levels up.
REPO_ROOT = EXERCISE_ROOT.parents[2]
NOTEBOOK = REPO_ROOT / "notebooks" / "S04-data-cleaning-dedup.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    if not NOTEBOOK.exists():
        pytest.fail(f"the session notebook is missing at {NOTEBOOK}; AGENTS.md requires one")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_the_notebook_is_valid_json_and_has_cells(nb):
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 10
    assert any(c["cell_type"] == "code" for c in nb["cells"])


def test_the_notebook_has_no_committed_outputs(nb):
    """Committed outputs bloat diffs — and on this exercise can carry real PII into git."""
    offenders = [
        i
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code" and (c.get("outputs") or c.get("execution_count") is not None)
    ]
    assert not offenders, (
        f"cells {offenders} carry committed outputs. Clear them before committing — "
        "later stages print scrubbed corpus text, and a stale output is untested content in git."
    )


def test_the_output_check_can_actually_fail():
    """Without this, the guard above would pass against a notebook with no code cells at all."""
    fake = {
        "cells": [
            {"cell_type": "code", "source": ["print(1)"], "outputs": [{"text": "1"}]},
            {"cell_type": "code", "source": ["print(2)"], "outputs": []},
        ]
    }
    offenders = [
        i
        for i, c in enumerate(fake["cells"])
        if c["cell_type"] == "code" and (c.get("outputs") or c.get("execution_count") is not None)
    ]
    assert offenders == [0], "a notebook with a committed output was not detected"


def test_the_notebook_imports_the_package_rather_than_reimplementing_it(nb):
    """The rule that stops the notebook drifting from what ships."""
    source = "\n".join(_code(nb))
    assert "import datacleaning" in source
    for module in ("pipeline", "tokens", "corpus", "export"):
        assert "datacleaning import" in source or f"datacleaning.{module}" in source

    # A notebook that redefined the pipeline would carry its own function and class definitions.
    assert "def clean_text" not in source, "the notebook must call the package, not redefine it"
    assert "class Config" not in source


def test_the_notebook_offers_a_lite_profile_by_default(nb):
    """A notebook nobody waits for is a notebook nobody runs."""
    source = "\n".join(_code(nb))
    assert "PROFILE = 'lite'" in source


def test_the_notebook_can_detect_colab_without_crashing_off_colab(nb):
    """`importlib.util.find_spec('google.colab')` raises off-Colab instead of returning None.

    That bug failed on cell 1 for every local reader before it was caught by running the notebook,
    which is exactly why this assertion is here rather than in a reviewer's head.
    """
    # Comments are stripped first: the notebook *explains* this bug in a comment, and matching the
    # explanation instead of the code would fail against the very cell that fixes it.
    lines = [ln for ln in "\n".join(_code(nb)).splitlines() if not ln.strip().startswith("#")]
    source = "\n".join(lines)
    assert "find_spec('google.colab')" not in source
    assert "except ImportError" in source


def test_the_notebook_covers_every_stage_the_pipeline_defines(nb):
    """A new stage that no notebook section mentions is a stage nobody will learn."""
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    missing = [name for _, _, name, _ in pipeline.STAGES if name.lower() not in text.lower()]
    assert not missing, f"the notebook never mentions: {missing}"


def test_the_notebook_carries_a_colab_badge(nb):
    text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "colab.research.google.com" in text
    assert NOTEBOOK.name in text, "the badge must point at this notebook, not another"


def test_the_notebook_is_named_with_its_session_id():
    """`AGENTS.md`: zero-padded session id first, so lexical sort equals session order."""
    assert NOTEBOOK.name.startswith("S04-")
    assert NOTEBOOK.suffix == ".ipynb"


def test_every_notebook_in_the_repo_follows_the_naming_rule():
    """The rule is repo-wide, so it is checked repo-wide rather than for this session alone."""
    folder = Path(NOTEBOOK).parent
    for path in folder.glob("*.ipynb"):
        assert path.name[0] == "S" and path.name[1:3].isdigit(), (
            f"{path.name} does not start with a zero-padded session id (SNN-)"
        )
