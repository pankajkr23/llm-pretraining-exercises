"""The documents must not disagree with the code, or with each other.

This repo's most expensive recurring defect is a hand-written sentence stating a number sitting
next to a correct generated table. The table looks maintained, so a reader believes the sentence.
`AGENTS.md` names four shipped examples; this exercise's README carried a fifth — a header reading
*stage 1 of 8* directly above a table marking five stages done.

Where prose must stay hand-written, a test asserts the number in it. That is what this file is.
"""

import re
import subprocess
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
README = EXERCISE / "README.md"
MODULES = EXERCISE / "src" / "trainingdata"

#: Python files the documents may name that **no clone receives**, with the reason.
#:
#: `tools/build_notebook.py` is gitignored by repo policy — every exercise has one, none is pushed,
#: and `AGENTS.md` explains why. It therefore exists on a working checkout and not in CI, so a
#: filesystem scan disagrees with a fresh clone about whether the README is honest.
#:
#: **Kept separate from a "not built yet" allowlist, and that separation is the lesson.** This
#: guard also once exempted `run_demo.py` and `verify.py` because they were named before they were
#: written. When that reason expired the whole allowlist was deleted — and took this permanent
#: exemption with it. One expires, the other never does; sharing a set means retiring the first
#: silently retires the second.
LOCAL_ONLY: set[str] = {"build_notebook.py"}


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

    phantom = sorted(named - present - LOCAL_ONLY)
    assert not phantom, f"the README names Python files that do not exist: {phantom}"


def test_the_not_shipped_paragraph_names_nothing_that_exists() -> None:
    """**The stale sentence, caught rather than shipped again — in the agent instructions.**

    `CLAUDE.md` carries a paragraph naming what the exercise does *not* have, and a paragraph
    immediately after it warning that no test reads the header so it goes stale silently. It did:
    it denied `fork`, `verify.py`, `run_demo.py`, the metrics module, the evidence writer, the
    corpus fetcher and a tracked `results/` while all seven were on disk. An agent reading it would
    have rebuilt work that was already done, or reported a finished deliverable as missing.

    So the sentence is now derived from the filesystem, the same way the shipped list is.

    **It matches directories as well as modules**, and that is the second hole rather than a
    flourish. The first version matched only `*.py` names, so when the paragraph went stale a second
    time — denying *"any `web/` bundle"* while that bundle was live in production — the guard read
    the sentence, found no Python file in it, and passed. A claim about a directory was invisible to
    the check written to catch stale claims.

    When nothing is outstanding the paragraph is removed entirely, and this test skips rather than
    failing on a missing marker: a repo with nothing to deny should not be forced to keep an empty
    denial around to satisfy a guard.
    """
    text = (EXERCISE / "CLAUDE.md").read_text()
    marker = "**Not shipped, and do not describe the exercise as having them:**"
    if marker not in text:
        pytest.skip("nothing is currently denied, so there is no claim to check")

    paragraph = text.split(marker, 1)[1].split("\n\n", 1)[0]
    denied = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_./-]*)`", paragraph))

    exists = set()
    for name in denied:
        candidate = name.rstrip("/")
        for base in (MODULES, EXERCISE):
            target = base / candidate
            if target.is_file() or target.is_dir():
                exists.add(name)
    assert not exists, (
        f"CLAUDE.md says {sorted(exists)} are not shipped; they are on disk. A reader would "
        f"rebuild work that is already done, or report a delivered artefact as missing."
    )


def test_every_python_file_the_readme_names_is_either_tracked_or_known_to_be_local_only() -> None:
    """**The half of the check above that a working checkout cannot see.**

    `test_no_python_file_is_named_that_does_not_exist` scans the filesystem, so it passes on a
    machine that has the gitignored builders and fails on a fresh clone — and CI is the fresh
    clone. That asymmetry is not hypothetical: removing an allowlist entry passed here and failed
    in CI, which is the slowest possible way to learn it.

    So this asks the question a clone would ask: is every Python file the README names actually
    *shipped*? A file that is deliberately not shipped is named in `LOCAL_ONLY` with the reason;
    anything else naming a file no clone receives is a broken promise to a reader.

    A newly written file must be `git add`ed before a document may name it. That is the intended
    order anyway — CI decides, and CI only sees the index.
    """
    tracked = {
        Path(line).name
        for line in subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=EXERCISE.parents[2],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    }
    named = set(re.findall(r"\b([a-z_][a-z0-9_]*\.py)\b", README.read_text()))

    unshipped = sorted(named - tracked - LOCAL_ONLY)
    assert not unshipped, (
        f"the README names {unshipped}, which no clone receives. Either commit them, or add them "
        f"to LOCAL_ONLY with the reason they are deliberately not shipped."
    )
