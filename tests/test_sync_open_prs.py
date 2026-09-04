"""The log-resolution logic in `tools/sync_open_prs.py`, which is where a bug would be silent.

**The plumbing is not the risk; the resolution is.** Merging `main` into a branch and pushing is
ordinary git — if it goes wrong it goes wrong loudly. What can fail quietly is the rule that puts
`docs/agents/QUEUE.md` and `CHANGELOG.md` back together: get it wrong in one direction and a
branch's own log entry disappears, get it wrong in the other and fifteen branches each land another
copy of the byte-identical `#103` line. Both produce a green build and a wrong file.

So this tests the two pure functions and nothing else. They take strings and return strings, which
is the whole reason they were written that way.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from sync_open_prs import _reapply  # noqa: E402

BASE = "alpha\nbravo\ncharlie\n"


def _blocks(base: str, branch: str):
    """`_own_additions` without git: the same difflib call over two strings."""
    import difflib

    before = base.splitlines(keepends=True)
    after = branch.splitlines(keepends=True)
    out = []
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(
        None, before, after, autojunk=False
    ).get_opcodes():
        if tag in ("insert", "replace"):
            out.append((after[j2] if j2 < len(after) else "", after[j1:j2]))
    return out


def test_a_branch_s_own_entry_survives_a_main_that_moved() -> None:
    """The ordinary case: main gained a line, the branch gained a different one. Keep both."""
    branch = "alpha\nMINE\nbravo\ncharlie\n"
    main = "alpha\nbravo\nTHEIRS\ncharlie\n"
    out, _ = _reapply(main, _blocks(BASE, branch))
    assert "MINE" in out, "the branch's own entry was dropped"
    assert "THEIRS" in out, "main's entry was dropped"
    assert out.count("MINE") == 1, f"the branch's entry was duplicated:\n{out}"


def test_a_line_main_already_has_is_not_added_twice() -> None:
    """The failure this tool exists for, and the reason `merge=union` was refused.

    Fifteen branches carry a byte-identical `#103` line. A union driver keeps both sides of every
    conflict, so the first merge lands one copy and the fifteenth lands fifteen.
    """
    shared = "alpha\nSHARED-103\nbravo\ncharlie\n"
    main = "alpha\nSHARED-103\nbravo\ncharlie\n"
    out, notes = _reapply(main, _blocks(BASE, shared))
    assert out.count("SHARED-103") == 1, f"the shared line was duplicated:\n{out}"
    assert any("already has" in n for n in notes), f"the skip was not reported: {notes}"


def test_a_block_whose_anchor_main_deleted_is_kept_not_lost() -> None:
    """Losing a log entry silently is the worse of the two failure directions.

    If the line a block was inserted before no longer exists on main, the block still has to land
    somewhere and the tool has to say so — an entry that vanishes leaves no trace to notice.
    """
    branch = "alpha\nMINE\nbravo\ncharlie\n"
    main = "alpha\ncharlie\n"  # `bravo`, the anchor, is gone
    out, notes = _reapply(main, _blocks(BASE, branch))
    assert "MINE" in out, "the entry was lost when its anchor disappeared"
    assert any("anchor" in n for n in notes), f"the fallback was not reported: {notes}"


def test_a_multi_line_block_stays_contiguous_and_in_order() -> None:
    """A queue entry is several indented lines; splitting one is unreadable, not merely wrong."""
    branch = "alpha\nONE\nTWO\nTHREE\nbravo\ncharlie\n"
    main = "alpha\nbravo\nTHEIRS\ncharlie\n"
    out, _ = _reapply(main, _blocks(BASE, branch))
    assert "ONE\nTWO\nTHREE\n" in out, f"the block was split or reordered:\n{out}"


def test_nothing_added_changes_nothing() -> None:
    """A branch that never touched the file must not have the file rewritten under it."""
    main = "alpha\nbravo\nTHEIRS\ncharlie\n"
    out, notes = _reapply(main, _blocks(BASE, BASE))
    assert out == main, f"an untouched file was modified:\n{out}"
    assert not notes
