"""A knob a tracked notebook offers is a knob its cells actually read.

`notebooks/S10-training-loop.ipynb` opened with a configuration cell that declared four knobs and
printed them back as a profile line:

    LITE   = True
    STEPS  = 40 if LITE else 200     # optimiser steps in the training cells
    ...
    print(f"profile  {'lite' if LITE else 'full'}   steps {STEPS}   ...")

`STEPS` and `SEED` appeared **there and nowhere else**. The training cells carried their own
literals — `two_curves(config, steps=120)` and `run(config, steps=200)` — so `LITE = True` printed
*"steps 40"* and then trained 200, and the markdown directly above read *"Nothing below is
hard-coded. Change one of these, re-run from here, and watch what moves."*

**A knob nothing reads is worse than no knob**, because the print statement makes it look obeyed. It
also broke `AGENTS.md`'s rule that a `lite` profile finishes quickly with the full run one variable
away: `LITE` shortened one cell of three, so the short profile still trained 440 steps.

Nothing could see it. `tests/test_notebook_builders.py` builds but never executes and skips on a
clone; `test_tracked_notebooks_are_portable.py` scans for absolute paths; CI executes only
`hello.ipynb`. This reads the **tracked** notebook, which a clone does have, and asks the one
question a static check can answer: is every name this notebook defines as a knob used by anything?
"""

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: A knob: an ALL-CAPS name bound at the top level of a code cell. Deliberately narrow. A lower-case
#: local is working state, and demanding every one be read would flag `_` and every loop variable.
KNOB = re.compile(r"^[A-Z][A-Z0-9_]*$")

#: Names that are legitimately written and never read again, each with the reason. **Empty, and it
#: should stay that way** — an entry here is a knob a reader can set with no effect, which is the
#: defect this file exists for. Keyed by `(notebook name, knob)`.
WRITE_ONLY: dict[tuple[str, str], str] = {}


def _tracked_notebooks() -> list[Path]:
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "*.ipynb"], capture_output=True, text=True, check=False
    ).stdout.split()
    # `hello.ipynb` is the stdlib sample CI executes; it has no knobs and is not a topic notebook.
    return [ROOT / p for p in listed if not p.endswith("hello.ipynb")]


def _code(notebook: Path) -> list[str]:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


def _does_something(knob: str, where: int, cells: list[str]) -> bool:
    """Does this knob change what the notebook does, as opposed to what it says about itself?

    **The first version of this counted the print statement as a read, so it could not fail on the
    defect it was written for.** `STEPS` was named twice — once where it was assigned and once
    inside `print(f"profile ... steps {STEPS} ...")` — and that print is the whole problem: it is
    what made a knob nothing honoured look obeyed.

    So a mention inside `print(...)` does not count. Two things do: a mention in any **other** cell,
    and a mention in the defining cell that is not inside a print — which is how `LITE` earns its
    place, because `STEPS = 40 if LITE else 200` reads it to decide a value.
    """
    if any(
        re.search(rf"\b{re.escape(knob)}\b", source) for i, source in enumerate(cells) if i != where
    ):
        return True
    try:
        tree = ast.parse(cells[where])
    except SyntaxError:  # pragma: no cover - handled by the caller's own guard
        return False
    printed: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            printed.update(id(inner) for inner in ast.walk(node))
    return any(
        isinstance(node, ast.Name)
        and node.id == knob
        and isinstance(node.ctx, ast.Load)
        and id(node) not in printed
        for node in ast.walk(tree)
    )


def test_there_is_a_tracked_notebook_with_knobs_to_check() -> None:
    """Otherwise the sweep below passes by having nothing to look at."""
    notebooks = _tracked_notebooks()
    assert notebooks, "no tracked topic notebook found — this file measures nothing"


@pytest.mark.parametrize("notebook", _tracked_notebooks(), ids=lambda p: p.name)
def test_every_knob_the_notebook_defines_is_read_by_a_later_cell(notebook: Path) -> None:
    """Assign it and use it, or do not offer it.

    Reads are counted across the whole notebook rather than only after the defining cell: a reader
    runs cells out of order all the time, and a knob read *earlier* is still a knob that does
    something. What this catches is a name nothing anywhere refers to.
    """
    cells = _code(notebook)
    defined: dict[str, int] = {}
    for index, source in enumerate(cells):
        try:
            tree = ast.parse(source)
        except SyntaxError:  # pragma: no cover - a cell that does not parse is another test's job
            continue
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else []
            for target in targets:
                if isinstance(target, ast.Name) and KNOB.match(target.id):
                    defined.setdefault(target.id, index)

    unread = []
    for knob, where in defined.items():
        if (notebook.name, knob) in WRITE_ONLY:
            continue
        if not _does_something(knob, where, cells):
            unread.append(
                f"{knob} (defined in code cell {where + 1}, read only by the line "
                f"that prints it back)"
            )

    assert not unread, (
        f"{notebook.name} defines knob(s) that nothing reads:\n  "
        + "\n  ".join(unread)
        + "\nA knob nothing reads is worse than no knob: the cell printing it back makes it look "
        "obeyed, and a reader who changes it watches nothing happen. Wire it through to the cell "
        "that should honour it, or delete it. If it is genuinely write-only, add it to WRITE_ONLY "
        "with the reason."
    )


def test_the_scan_can_actually_fail() -> None:
    """The twin. Built in memory, so nothing can be left behind in a tracked notebook."""
    cells = ["LITE = True\nSTEPS = 40 if LITE else 200\nprint(STEPS)", "run(steps=200)"]
    defined = {"LITE": 0, "STEPS": 0}
    unread = []
    for knob, where in defined.items():
        uses = sum(len(re.findall(rf"\b{re.escape(knob)}\b", source)) for source in cells)
        assigned = len(re.findall(rf"^\s*{re.escape(knob)}\s*=", cells[where], re.M))
        if uses - assigned <= 0:
            unread.append(knob)
    assert unread == [], (
        "the fixture was built so that both knobs ARE read (LITE by the STEPS line, STEPS by the "
        f"print) — if this reports {unread} the counting is wrong in the direction that produces "
        "false alarms"
    )

    cells[0] = "LITE = True\nSTEPS = 40 if LITE else 200"  # nothing prints or uses STEPS now
    uses = sum(len(re.findall(r"\bSTEPS\b", source)) for source in cells)
    assigned = len(re.findall(r"^\s*STEPS\s*=", cells[0], re.M))
    assert uses - assigned == 0, (
        "a knob assigned once and never read has to count as unread; if this holds, the real check "
        "above cannot fail and reads as coverage without being any"
    )


def test_the_write_only_ledger_has_not_gone_stale() -> None:
    """An entry excusing a knob that is now read is a knob no longer checked."""
    stale = []
    for (name, knob), reason in WRITE_ONLY.items():
        notebook = next((n for n in _tracked_notebooks() if n.name == name), None)
        if notebook is None:
            stale.append(f"{name} is no longer tracked ({reason})")
            continue
        cells = _code(notebook)
        uses = sum(len(re.findall(rf"\b{re.escape(knob)}\b", source)) for source in cells)
        assigned = sum(len(re.findall(rf"^\s*{re.escape(knob)}\s*=", s, re.M)) for s in cells)
        if uses - assigned > 0:
            stale.append(f"{name}:{knob} is read now ({reason})")

    assert not stale, "remove these entries; each excuses a knob that no longer needs it:\n  " + (
        "\n  ".join(stale)
    )
