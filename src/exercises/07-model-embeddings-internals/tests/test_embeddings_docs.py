"""Every module is named in the documents that list modules.

`AGENTS.md` records this failure with a cost attached: exercise 06 shipped `explainer.py` and left
it missing from three lists, so a reader regenerating that site ran one tool without another and
published a page whose figures contradicted its own tool. The guard there is what caught `replay.py`
having shipped without the README ever learning about it, and `AGENTS.md` asks for it wherever an
exercise grows past a handful of modules. This one just went from six to eight.

**Its limit is worth stating, because it is the same limit exercise 06's has.** It checks the
*document*, not the *list*: a module named once anywhere in the prose satisfies it, while the layout
block a reader actually follows can stay wrong. It catches the module nobody wrote down at all,
which is the failure that has actually happened here.
"""

import re
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
PACKAGE = EXERCISE / "src" / "embeddings"
DOCUMENTS = ("README.md", "CLAUDE.md")

TOOL_SCRIPTS = frozenset(
    {"build_notebook.py", "build_web_data.py", "measure_lock_samples.py", "run_experiment.py"}
)
"""Scripts under `tools/`. Named in the documents, and correctly so - they are not package
modules, and the reverse check below must not read them as stale entries."""

MODULES = sorted(path.name for path in PACKAGE.glob("*.py") if not path.name.startswith("__"))


def test_the_package_has_the_modules_this_guard_thinks_it_has() -> None:
    """A guard that globs an empty directory passes for every input.

    If the package moved, every assertion below would hold vacuously and read as coverage.
    """
    assert len(MODULES) >= 6, f"only found {MODULES} - has src/embeddings/ moved?"


@pytest.mark.parametrize("document", DOCUMENTS)
@pytest.mark.parametrize("module", MODULES)
def test_every_module_is_named_in_the_documents_that_list_modules(
    document: str, module: str
) -> None:
    """Both documents carry a module table, so both must know about every module."""
    text = (EXERCISE / document).read_text(encoding="utf-8")
    assert module in text, (
        f"{module} is not named anywhere in {document}. A module nobody wrote down is one a "
        "reader will not run, and both of this exercise's documents carry a module table."
    )


@pytest.mark.parametrize("document", DOCUMENTS)
def test_the_documents_do_not_name_a_module_that_no_longer_exists(document: str) -> None:
    """The other direction, which the exercise-06 version does not check.

    A table that still lists a deleted module sends a reader to a file that is not there, and the
    forward check above cannot see it.

    **Only bare names count.** A negative lookbehind for `/` is doing real work here: this
    exercise's documents quote the paths of the missing experiment scripts (`k2/scale_cost.py` and
    friends), and a split-on-whitespace version of this check reported those as deleted modules of
    ours. They are neither ours nor deleted, and flagging them would have been a guard demanding
    that a true sentence be reworded.
    """
    text = (EXERCISE / document).read_text(encoding="utf-8")
    known = set(MODULES) | {"__init__.py"} | TOOL_SCRIPTS
    for candidate in set(re.findall(r"(?<![\w/])([a-z][a-z0-9_]*\.py)\b", text)):
        assert candidate in known, (
            f"{document} names {candidate}, which is not in src/embeddings/ and is not one of "
            "this exercise's tool scripts. Either the module was deleted and the document was "
            "not updated, or the name is a typo."
        )
