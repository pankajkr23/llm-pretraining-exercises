"""The documents must not disagree with the code, or with each other.

This repo's most expensive recurring defect is a hand-written sentence stating a number sitting
next to a correct generated table. The table looks maintained, so a reader believes the sentence.
`AGENTS.md` names four shipped examples; this exercise's README carried a fifth — a header reading
*stage 1 of 8* directly above a table marking five stages done.

Where prose must stay hand-written, a test asserts the number in it. That is what this file is.
"""

import re
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
README = EXERCISE / "README.md"
MODULES = EXERCISE / "src" / "trainingdata"


def _stage_rows() -> list[tuple[int, str]]:
    """Every row of the README's stage table.

    Returns:
        `(stage number, status cell)` pairs.
    """
    rows = re.findall(r"^\|\s*(\d+)\s*\|[^|]*\|\s*([^|]*?)\s*\|\s*$", README.read_text(), re.M)
    return [(int(number), status) for number, status in rows]


def test_the_status_line_matches_the_stage_table() -> None:
    """**The sentence-versus-table failure, caught rather than shipped again.**

    The header claimed stage 1 while the table below it marked five stages done. Nothing failed,
    and a reader who trusted the sentence would have believed the exercise had barely started.
    """
    rows = _stage_rows()
    assert rows, "the stage table was not found; this guard is now inert"

    done = [number for number, status in rows if "done" in status.lower()]
    claimed = re.search(r"\*\*Status: stage (\d+) of (\d+)\.\*\*", README.read_text())
    assert claimed, "the README lost its status line"

    assert int(claimed.group(2)) == len(rows), (
        f"the status line says there are {claimed.group(2)} stages; the table has {len(rows)}"
    )
    assert int(claimed.group(1)) == max(done, default=0), (
        f"the status line claims stage {claimed.group(1)} but the table marks "
        f"{max(done, default=0)} as the furthest one done"
    )


def test_the_stages_are_completed_in_order() -> None:
    """A gap would mean a stage was skipped, which the staged build exists to prevent."""
    rows = _stage_rows()
    done = [number for number, status in rows if "done" in status.lower()]
    assert done == list(range(1, len(done) + 1)), f"stages completed out of order: {done}"


@pytest.mark.parametrize("document", ["README.md", "CLAUDE.md"], ids=["readme", "claude"])
def test_every_module_is_named_in_the_documents_that_list_modules(document: str) -> None:
    """**A new module is not done until every list that names modules includes it.**

    `explainer.py` shipped in exercise 05 and stayed missing from three such lists, none of which
    any test checked. The consequence was not cosmetic: a reader following the README would have
    regenerated a site whose figures contradicted its own tool.
    """
    text = (EXERCISE / document).read_text()
    modules = sorted(p.name for p in MODULES.glob("*.py") if p.name != "__init__.py")
    missing = [name for name in modules if name not in text]
    assert not missing, f"{document} does not mention {missing}"


def test_no_python_file_is_named_that_does_not_exist() -> None:
    """The other direction: a document promising a file that was renamed or never written.

    A reader following it gets an `ImportError` or an empty `ls`, which is a worse first impression
    than an omission.

    Searched over the whole REPO, not just this exercise, and twice widened for the same reason:
    first when the README began naming the three torch-gated **test** files to state the size of
    CI's blind spot, then when it began naming `tests/test_ci_shards_cover_everything.py`, the
    repo-level guard that closed it. Both times the document had become *more* precise and the test
    called it a lie. A guard that fires on a document improving is measuring the wrong thing — the
    real defect is a name with no file behind it anywhere.
    """
    text = README.read_text()
    repo = EXERCISE.parents[2]
    present = {
        p.name
        for p in repo.rglob("*.py")
        if ".venv" not in p.parts and "node_modules" not in p.parts
    }
    named = set(re.findall(r"\b([a-z_][a-z0-9_]*\.py)\b", text))

    # TWO different reasons a named file may be absent, kept apart on purpose. Rolling them into
    # one set is what broke this: when the `planned` reason expired the whole allowlist was
    # deleted, and the permanent exemption went with it — green locally, red in CI.
    #
    # Temporary: named but not yet written. The README's stage table says stage 8 is unfinished,
    # and the producer/auditor section has to name the two commands the work will be graded on.
    # These entries expire when the files land.
    planned = {"run_demo.py", "verify.py"}

    # Permanent: written, present on every working checkout, and deliberately never shipped.
    # `tools/build_notebook.py` is gitignored by repo policy, so it exists here and on no clone —
    # which is precisely why a filesystem scan disagrees with CI about whether the README is honest.
    local_only = {"build_notebook.py"}

    phantom = sorted(named - present - planned - local_only)
    assert not phantom, f"the README names Python files that do not exist: {phantom}"
