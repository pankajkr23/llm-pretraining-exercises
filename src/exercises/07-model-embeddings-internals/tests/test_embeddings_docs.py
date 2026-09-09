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
    path.name for path in [*(EXERCISE / "tools").glob("*.py"), *EXERCISE.glob("*.py")]
)
"""Scripts under `tools/` and at the exercise root. Named in the documents, and correctly so - they
are not package modules, and the reverse check below must not read them as stale entries.

**Read from the filesystem, not written out.** This was a hardcoded list of four names, and it went
stale the moment the exercise grew `verify.py`, `evidence.py` and four measurement tools - the guard
then reported a correctly-documented script as a deleted module. A list of files beside the files is
a second copy, and the second copy is the one that drifts."""

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


def _collected(path: Path) -> int:
    """How many cases pytest would collect from one test file, counted with `ast`.

    **`ast` rather than a regex, and the difference is the whole guard.** A regex over
    `@pytest.mark.parametrize` misses a list of bare values, misses stacked decorators (which
    MULTIPLY rather than add) and misses anything indented unusually — it counted 19 in the browser
    file where pytest collects 33, and a bound loose enough to accept that gap also accepted the
    stale number this guard exists to catch.

    Stacked `parametrize` decorators multiply, which is pytest's own rule and the reason the browser
    file's count is not simply its function count.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    total = 0
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        cases = 1
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            name = ast.unparse(decorator.func)
            if not name.endswith("parametrize"):
                continue
            if len(decorator.args) >= 2 and isinstance(decorator.args[1], (ast.List, ast.Tuple)):
                cases *= max(len(decorator.args[1].elts), 1)
        total += cases
    return total


def test_the_readmes_test_counts_are_derived_from_the_test_files() -> None:
    """The README said *54 tests* and a *20-test browser suite*. Both were far out.

    Counted from the files with `ast` rather than by running pytest. A count from a live collection
    would be a count of whatever happened to be installed — this exercise has two files behind an
    `importorskip` — so the number would move with the environment rather than with the code. And a
    guard that must run the suite to check a sentence about the suite cannot run in the fast job.

    **What it cannot see**, so that the band below is a decision rather than a fudge: a
    `parametrize` whose argument is a NAME rather than a literal — `parametrize("arm", ARMS)` is
    ten cases and reads as one — plus fixture-generated cases and `pytest_generate_tests`. Measured:
    `ast` counts 155 where pytest collects 205, and 25 where pytest collects 33. So the count is a
    LOWER BOUND and the assertion is `floor <= stated <= 1.5 * floor`.

    That band still catches the failure that happened: the README said **54** against a floor of 155
    and **20** against a floor of 25, and both are refused. A number cannot be stale by a factor of
    three and survive.
    """
    counted = {p.name: _collected(p) for p in sorted((EXERCISE / "tests").glob("test_*.py"))}
    total = sum(counted.values())
    browser = counted["test_embeddings_render.py"]
    readme = (EXERCISE / "README.md").read_text(encoding="utf-8")

    claimed = re.search(r"\*\*(\d+) tests\*\*", readme)
    claimed_browser = re.search(r"\*\*(\d+)-test\*\* browser suite", readme)
    assert claimed and claimed_browser, "the README no longer states both counts in the pinned form"

    for label, stated, floor in (
        ("tests", int(claimed.group(1)), total),
        ("browser tests", int(claimed_browser.group(1)), browser),
    ):
        assert floor <= stated <= floor * 1.5, (
            f"the README claims {stated} {label}; the files hold {floor} countable cases, so "
            f"anything outside {floor}-{int(floor * 1.5)} is stale rather than imprecise "
            f"({counted})"
        )
