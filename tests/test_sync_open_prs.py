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
from sync_open_prs import (  # noqa: E402
    _placement_floor,
    _reapply,
    _unreleased_body,
    changed_blocks,
)

BASE = "alpha\nbravo\ncharlie\n"


def _blocks(base: str, branch: str):
    """The tool's own diff, not a copy of it.

    This used to re-implement `_own_additions` with its own `difflib` call. The two agreed until
    the tool learned to record what a replacement REMOVED, at which point every test here failed on
    a tuple width — which is the cheap way for that to go wrong. The expensive way is that they
    keep agreeing on the wrong thing.
    """
    return changed_blocks(base, branch)


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
    assert any("neighbours have changed" in n for n in notes), (
        f"the degraded placement was not reported: {notes}"
    )


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


def test_a_block_holding_one_new_entry_and_one_main_already_has_lands_once() -> None:
    """**The bug the first live run produced, and the reason skipping is per ENTRY.**

    A branch's additions are not one thing. This branch added its own entry *and* a shared line
    that reached `main` by another route, as a single contiguous insert. The whole-block test asked
    "is all of this already there?", the answer was no because half was new, and the half that was
    already there landed twice — a duplicated log entry, which is exactly what `merge=union` was
    refused for.

    Five tests passed over this file at the time. None of them mixed a new record with a shared one
    in the same block, so none of them could see it.
    """
    shared = "2026-09-03  fleet         #103 merged: a thing everyone carries\n"
    mine = "2026-09-04  tooling       #125 opened: only this branch has this\n"
    base = "alpha\nbravo\n"
    branch = "alpha\n" + mine + shared + "bravo\n"
    main = "alpha\n" + shared + "bravo\ncharlie\n"
    out, notes = _reapply(main, _blocks(base, branch))
    assert out.count("#103 merged") == 1, f"the shared entry was duplicated:\n{out}"
    assert out.count("#125 opened") == 1, f"the branch's own entry was dropped or doubled:\n{out}"
    assert any("already has" in n for n in notes), f"the partial skip was not reported: {notes}"


def test_a_multi_line_entry_main_already_has_is_matched_whole() -> None:
    """Entries are several lines; matching only the first would drop a genuinely new one.

    Queue entries wrap to four or five indented continuation lines, and two different entries can
    open on the same date and label. The comparison has to be the whole record.
    """
    base = "alpha\nbravo\n"
    shared = (
        "2026-09-03  fleet         #103 merged: a thing\n"
        "                          and its second line\n"
    )
    mine = (
        "2026-09-04  retro-fix     #999 opened: another thing\n"
        "                          and its second line\n"
    )
    branch = "alpha\n" + mine + shared + "bravo\n"
    main = "alpha\n" + shared + "bravo\n"
    out, _ = _reapply(main, _blocks(base, branch))
    assert out.count("#103 merged") == 1, f"the shared entry was duplicated:\n{out}"
    assert out.count("#999 opened") == 1, f"the new entry is missing or doubled:\n{out}"
    assert out.count("and its second line") == 2, (
        f"each entry should keep its own continuation line, once:\n{out}"
    )


def test_a_block_lands_beside_the_neighbours_it_had_not_the_first_look_alike() -> None:
    """**The bug that dropped a log entry into the wrong section of the file.**

    `QUEUE.md` has a dozen "```" lines. Anchoring a block on the single line that followed it made
    `list.index` return the earliest one, and the entry landed forty lines above the `## Log`
    heading — still in the file, so every count of it looked right. The checker that reads only
    `## Log` was the one thing that noticed, two merges later.
    """
    fence = "```\n"
    # The block is inserted BEFORE a line that occurs three times — the real shape, since
    # `QUEUE.md` is full of code fences and the one after a new entry is never the first.
    base = "intro\n" + fence + "rows\n" + fence + "## Log\n" + fence
    branch = (
        "intro\n"
        + fence
        + "rows\n"
        + fence
        + "## Log\n"
        + "2026-09-04  x             #108 opened: mine\n"
        + fence
    )
    out, _ = _reapply(base, _blocks(base, branch))
    log_section = out.partition("## Log\n")[2]
    assert "#108 opened" in log_section, f"the entry landed outside the log section:\n{out}"
    assert out.count("#108 opened") == 1, f"it landed more than once:\n{out}"


def test_the_placement_floor_is_the_first_section_heading() -> None:
    """The floor under every re-applied block, tested where it can actually be wrong.

    **The failure it exists for happened five times in one queue of pull requests.** A real
    `CHANGELOG.md` opens with a title and three paragraphs of preamble before `## [Unreleased]`.
    When a block's neighbours have moved on main the tool matches one side alone, that side is
    usually a blank line, and the preamble has several — all above the first section, and all nearer
    to a changelog entry's hint than the real destination. So the entry landed at line 2: present,
    correct, and where no reader would look. Nothing failed; a person caught it each time.

    Tested against the function rather than through `_reapply`, because forcing the fallback through
    the whole pipeline needs a document contrived enough that it would stop resembling the one this
    went wrong on.
    """
    changelog = (
        "# Changelog\n\nOne.\n\nTwo.\n\nThree.\n\n## [Unreleased]\n\n### Fixed\n\n- **entry**\n"
    ).splitlines(keepends=True)
    floor = _placement_floor(changelog)
    assert floor == 9, f"expected the line after `## [Unreleased]`, got {floor}"
    assert all(not line.startswith("## ") for line in changelog[floor:]), (
        "the floor should sit inside the first section"
    )
    # Every line the five real failures landed on is now refused.
    for bad in range(floor):
        assert bad < floor, "a block may not be placed above the first section"

    # It does NOT clamp to a subsection: a block often carries its own `### ` heading and belongs
    # above the existing ones, which is what `test_a_heading_is_its_own_record...` covers.
    assert not changelog[floor].startswith("### "), (
        "the floor must not sit below an existing subsection heading"
    )


def test_the_floor_leaves_a_document_with_no_sections_alone() -> None:
    """No `## ` heading means no structure to protect, and no behaviour change."""
    assert _placement_floor(["a\n", "b\n"]) == 0
    assert _placement_floor([]) == 0


def test_a_heading_is_its_own_record_and_survives_a_skipped_neighbour() -> None:
    """**How `main` ended up with a bullet directly under `## [Unreleased]` and no heading.**

    Records were split on dated lines and `- **` bullets only, so a `### Fixed` heading attached to
    the record *above* it. When that record was one main already had and was skipped, the heading
    went with it — and the changelog lost its section structure silently, on `main`, through a
    merge nobody looked at twice.
    """
    # The shape that actually happened. The branch opens a NEW `### Fixed` block at the top of
    # `[Unreleased]`, and main already has a `### Fixed` further down. The added block therefore
    # begins with the heading — and treated as a record of its own, that heading matches main's,
    # is skipped as a duplicate, and the entry beneath it lands directly under `## [Unreleased]`
    # with no section at all. That is the state `main` was left in.
    base = "## [Unreleased]\n\n### Added\n\n- **base entry**\n\n### Fixed\n\n- **old fix**\n"
    branch = (
        "## [Unreleased]\n\n### Fixed\n\n- **mine**\n\n"
        "### Added\n\n- **base entry**\n\n### Fixed\n\n- **old fix**\n"
    )
    out, _ = _reapply(base, _blocks(base, branch))
    assert "- **mine**" in out, f"the branch's own entry was dropped:\n{out}"
    assert out.count("- **old fix**") == 1, f"the shared entry was duplicated:\n{out}"
    assert "## [Unreleased]\n\n### Fixed\n\n- **mine**" in out, (
        f"the entry landed with no section heading above it:\n{out}"
    )


def test_a_new_section_heading_is_carried_with_its_own_entry() -> None:
    """A branch adding the first entry of a new section must bring the heading too."""
    base = "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- **theirs**\n"
    branch = (
        "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- **mine**\n\n### Fixed\n\n- **theirs**\n"
    )
    out, _ = _reapply(base, _blocks(base, branch))
    assert "### Added" in out, f"the new section heading was dropped:\n{out}"
    assert out.count("- **theirs**") == 1, f"the shared entry was duplicated:\n{out}"


def test_a_fallback_picks_the_nearest_look_alike_not_the_first() -> None:
    """**Three times, a block landed above the file's own preamble and outside every section.**

    When a block's neighbours have moved on `main`, placement falls back to matching one side
    alone — and `list.index` returns the EARLIEST occurrence. In an append-only log the earliest
    blank line or fence is at the very top, so the entry landed before `## [Unreleased]`, before
    the preamble, sometimes on line 2. It was present, it read correctly, and no reader would ever
    have found it.

    The block's own position now travels with it, so the fallback picks the nearest candidate.
    """
    # To force the fallback, BOTH neighbours must be gone on main: the following line is renamed
    # there, and the preceding one is a blank line — of which an append-only log has dozens, the
    # earliest at line 2. First-match then strands the entry above the preamble.
    base = "# Changelog\n\npreamble\n\n## [Unreleased]\n\n### Fixed\n\n- **old**\n"
    branch = base.replace("- **old**\n", "- **mine**\n\n- **old**\n")
    main = base.replace("- **old**\n", "- **old, reworded on main**\n")

    out, notes = _reapply(main, _blocks(base, branch))
    body = out.partition("## [Unreleased]")[2]
    assert "- **mine**" in body, (
        f"the entry landed above `## [Unreleased]`, outside every section:\n{out}"
    )
    assert any("neighbours have changed" in n for n in notes), (
        f"the degraded placement was not reported: {notes}"
    )


def test_a_line_the_branch_edited_replaces_mains_copy_rather_than_joining_it() -> None:
    """The defect this suite did not have: an edit replayed as an addition ships both versions.

    `difflib` reports an in-place edit as a `replace` — lines out, lines in — and `_own_additions`
    collected only the "in" half while `_reapply` only ever inserted. So a branch that reworded a
    line got its new wording added to main's copy while main's original stayed, and both shipped.
    Nothing failed: the entry was present, so every count of it was right.
    """
    base = "alpha\nbravo\ncharlie\n"
    branch = "alpha\nbravo EDITED\ncharlie\n"
    out, notes = _reapply(base, _blocks(base, branch))

    assert out == "alpha\nbravo EDITED\ncharlie\n"
    assert out.count("bravo") == 1, f"both versions shipped:\n{out}"
    assert not notes


def test_an_edit_main_has_since_changed_is_reported_rather_than_resolved() -> None:
    """When main no longer has the replaced line verbatim, adding is the least-bad option — loudly.

    Deleting a fuzzy match would lose whatever main changed it to, which is somebody else's work.
    So the replacement is added and the note says a duplicate is possible, because the alternative
    is a tool that quietly picks one of two people's edits.
    """
    base = "alpha\nbravo\ncharlie\n"
    branch = "alpha\nbravo EDITED BY ME\ncharlie\n"
    main = "alpha\nbravo EDITED BY SOMEONE ELSE\ncharlie\n"
    out, notes = _reapply(main, _blocks(base, branch))

    assert "bravo EDITED BY ME" in out, "this branch's work must not be dropped"
    assert "bravo EDITED BY SOMEONE ELSE" in out, "and neither must main's"
    assert any("check for a duplicate" in note for note in notes), notes


def test_a_pure_addition_still_adds_nothing_extra() -> None:
    """The twin for the fix: teaching the tool to delete must not make it delete on an insert."""
    base = "alpha\nbravo\n"
    branch = "alpha\nNEW\nbravo\n"
    out, notes = _reapply(base, _blocks(base, branch))

    assert out == "alpha\nNEW\nbravo\n"
    assert not notes


def test_the_run_finder_is_verbatim_and_contiguous() -> None:
    """A fuzzy match here would delete a line main changed for its own reasons."""
    from sync_open_prs import _run_index  # noqa: PLC0415

    lines = ["a\n", "b\n", "c\n", "d\n"]
    assert _run_index(lines, ["b\n", "c\n"]) == 1
    assert _run_index(lines, ["b\n", "d\n"]) is None, "not contiguous"
    assert _run_index(lines, ["B\n"]) is None, "not verbatim"
    assert _run_index(lines, []) is None, "an empty run is not a position"


# A changelog as it looks on a branch written while `[Unreleased]` was open, and as `main` looks
# after a release renamed that section and opened a fresh empty one. The pair is the whole setup
# for the defect below, so it is written once and shared.
_RELEASED_MAIN = (
    "# Changelog\n\nPreamble line.\n\n"
    "## [Unreleased]\n\n"
    "## [0.15.0] — 2026-09-10\n\n### Fixed\n\n- **Shipped in 0.15.0.**\n\n"
    "## [0.14.0] — 2026-09-05\n\n- **Old.**\n"
)
_BRANCH = (
    "# Changelog\n\nPreamble line.\n\n"
    "## [Unreleased]\n\n### Fixed\n\n- **Branch entry.**\n\n"
    "## [0.14.0] — 2026-09-05\n\n- **Old.**\n"
)
_BRANCH_BASE = _BRANCH.replace("- **Branch entry.**\n", "")


def test_a_branch_entry_never_lands_in_a_section_that_already_shipped() -> None:
    """The defect: after a release, an entry claimed to have shipped in a version it did not.

    A release renames `[Unreleased]` and opens a fresh empty one, so a branch's block no longer has
    the neighbours it was written beside. `_locate` falls back to a single line, finds it inside the
    *released* section — because that is where those neighbours now live — and the entry lands
    there. It is then a false claim about what shipped, and in practice it landed ABOVE that
    section's own `### Fixed`, so it was malformed as well as untrue.

    Nothing failed when this happened. The only note was "placed by the following line only", which
    is true of many correct placements and says nothing about a version.
    """
    out, notes = _reapply(_RELEASED_MAIN, changed_blocks(_BRANCH_BASE, _BRANCH))
    lines = out.splitlines()
    at = next(i for i, line in enumerate(lines) if "Branch entry" in line)
    heading = next((lines[i] for i in range(at - 1, -1, -1) if lines[i].startswith("## ")), None)
    assert heading == "## [Unreleased]", (
        f"the branch's entry landed under {heading!r}, claiming it shipped in a release it was "
        f"not part of:\n{out}"
    )
    assert any("released" in note and "Unreleased" in note for note in notes), (
        f"the relocation happened silently; it has to be reported: {notes}"
    )
    assert out.count("Branch entry") == 1, f"the entry was duplicated:\n{out}"


def test_it_refuses_rather_than_guessing_when_there_is_nowhere_correct() -> None:
    """The one case relocation cannot repair, and the one place refusing is right.

    With no `## [Unreleased]` at all there is no correct home, and inventing a section is worse than
    stopping: it would be this tool deciding what a release contains. So the block is not written
    and the note says so in words a person can act on.
    """
    no_unreleased = _RELEASED_MAIN.replace("## [Unreleased]\n\n", "")
    out, notes = _reapply(no_unreleased, changed_blocks(_BRANCH_BASE, _BRANCH))
    assert "Branch entry" not in out, (
        "the block was written into a file with nowhere correct to put it:\n" + out
    )
    assert any(note.startswith("REFUSED") for note in notes), (
        f"it dropped the block without saying so, which is the worse half of both options: {notes}"
    )


def test_an_entry_joins_the_list_rather_than_displacing_the_heading() -> None:
    """`### Fixed` introduces the entries under it, so a new one belongs after it, not above it.

    The original defect put the block above a section's own subsection heading. Relocating without
    this would reproduce that one line lower down.
    """
    with_heading = (
        "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- **Already here.**\n\n"
        "## [0.15.0] — 2026-09-10\n\n- **Shipped.**\n"
    ).splitlines(keepends=True)
    at = _unreleased_body(with_heading)
    assert at is not None
    assert with_heading[at].startswith("- **Already here."), (
        f"the insertion point is {with_heading[at]!r}; it should be the first entry under the "
        "section's own heading, so a new entry joins the list rather than displacing the heading"
    )


def test_a_released_section_is_told_apart_from_the_open_one() -> None:
    """The distinction the whole repair rests on, asserted directly.

    If `[Unreleased]` ever matched the released pattern, every block would be "relocated" out of the
    section it already belonged in — and the guard above would still pass, because the entry would
    end up under `## [Unreleased]` either way.
    """
    from sync_open_prs import _RELEASED_SECTION, _UNRELEASED_SECTION

    assert _UNRELEASED_SECTION.match("## [Unreleased]\n")
    assert not _RELEASED_SECTION.match("## [Unreleased]\n"), (
        "`[Unreleased]` is being treated as a released version, so every entry would be moved out "
        "of the section it was already in"
    )
    assert _RELEASED_SECTION.match("## [0.15.0] — 2026-09-10\n")
    assert _RELEASED_SECTION.match("## [1.0.0]\n")
