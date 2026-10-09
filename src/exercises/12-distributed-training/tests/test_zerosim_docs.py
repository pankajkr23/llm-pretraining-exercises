"""The documents name every module, promise no file that does not exist, and state their limits.

Copied in shape from exercise 06's `test_every_module_is_named_in_the_documents_that_list_modules`,
for the reason it gives: a new module is not done until every list that names modules includes it.
"""

import re
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
README = EXERCISE / "README.md"
MODULES = EXERCISE / "src" / "zerosim"

#: Python files the documents may name that no clone receives. `build_notebook.py` is gitignored by
#: repository policy, so a filesystem scan of a clone would call a correct mention a lie.
LOCAL_ONLY: set[str] = {"build_notebook.py"}


#: The section of each document that IS its module list. Exercise 06's guard checks the whole
#: document, and `AGENTS.md` names the hole that leaves: a module mentioned once in passing
#: satisfies it while the list a reader follows is wrong. That hole was watched here — deleting
#: `timing.py` from CLAUDE.md's list left the whole-document version green, because a later
#: paragraph named it.
MODULE_LISTS = {"README.md": "How the pieces fit", "CLAUDE.md": "Modules"}


@pytest.mark.parametrize("document", sorted(MODULE_LISTS), ids=["claude", "readme"])
def test_every_module_is_named_in_the_documents_that_list_modules(document: str) -> None:
    text = _section((EXERCISE / document).read_text(encoding="utf-8"), MODULE_LISTS[document])
    assert text, f"{document} has no '## {MODULE_LISTS[document]}' section to check"
    modules = sorted(p.name for p in MODULES.glob("*.py") if p.name != "__init__.py")
    assert modules, "found no modules — this guard is checking nothing"
    missing = [name for name in modules if f"`{name}`" not in text]
    assert not missing, f"{document}'s module list does not name {missing}"


@pytest.mark.parametrize("document", ["README.md", "CLAUDE.md", "DECISIONS.md", "PROGRESS.md"])
def test_no_python_file_is_named_that_does_not_exist(document: str) -> None:
    text = (EXERCISE / document).read_text(encoding="utf-8")
    repo = EXERCISE.parents[2]
    present = {
        p.name
        for p in repo.rglob("*.py")
        if ".venv" not in p.parts and "node_modules" not in p.parts
    }
    named = set(re.findall(r"\b([a-z_][a-z0-9_]*\.py)\b", text))
    phantom = sorted(named - present - LOCAL_ONLY)
    assert not phantom, f"{document} names Python files that do not exist: {phantom}"


def _section(text: str, heading: str) -> str:
    match = re.search(rf"^## {re.escape(heading)}$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return match.group(1) if match else ""


def test_the_limits_section_is_substantial() -> None:
    """A limits section of one sentence is a section in name only."""
    words = len(_section(README.read_text(encoding="utf-8"), "What this cannot establish").split())
    assert words >= 150, f"'What this cannot establish' has {words} words"


def test_the_readme_corrects_both_misconceptions() -> None:
    """Batch size multiplies activations, not weights; ZeRO-3 is not pipeline parallelism."""
    section = _section(README.read_text(encoding="utf-8"), "Two things people get wrong")
    assert "activations" in section
    assert "pipeline parallelism" in section
    assert "1/N of every layer" in section


def test_the_time_model_is_called_assumed_wherever_it_appears() -> None:
    """A time from assumed bandwidths must never read as a measurement."""
    limits = _section(README.read_text(encoding="utf-8"), "What this cannot establish")
    assert "assumed" in limits.lower()
    results = (EXERCISE / "RESULTS.md").read_text(encoding="utf-8")
    assert "ASSUMED" in results and "Not a measurement" in results
