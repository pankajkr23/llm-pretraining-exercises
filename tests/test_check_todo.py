"""`tools/check_todo.py` — the reconciler, and every way it could read as working while not.

**Why this file exists, in one sentence.** A tool whose whole job is to catch a stale list is worth
nothing unless it is itself checked, and the three properties that matter are all invisible to a
casual run: that an unannotated item is NOT reported as fine, that a typo in a predicate is NOT
treated as satisfied, and that every marker on a line is seen rather than only the first.

The tool reconciles a **gitignored** file, so this suite builds its own checklists in `tmp_path`
rather than reading `TODO.md`. That is the only shape that runs in CI, and it is also the honest
one: the tool is tracked and tested here, the file it reconciles is local, and no test can pretend
otherwise.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_todo  # noqa: E402
from check_todo import ITEM, LIST_LINE, evaluate, main, reconcile  # noqa: E402


@pytest.fixture
def checklist(tmp_path, monkeypatch):
    """A writable checklist whose predicates resolve against `tmp_path`, not the real repository."""
    monkeypatch.setattr(check_todo, "REPO", tmp_path)

    def write(body: str) -> Path:
        path = tmp_path / "TODO.md"
        path.write_text(body, encoding="utf-8")
        return path

    return write


# --- the predicates -------------------------------------------------------------------------


def test_exists_answers_both_ways(tmp_path, monkeypatch):
    """`exists` is true for a file that is there and false for one that is not."""
    monkeypatch.setattr(check_todo, "REPO", tmp_path)
    (tmp_path / "there.md").write_text("x", encoding="utf-8")
    assert evaluate("exists there.md")[0] is True
    assert evaluate("exists missing.md")[0] is False


def test_present_and_absent_are_opposites(tmp_path, monkeypatch):
    """The two text predicates read one file and disagree, which is why both exist."""
    monkeypatch.setattr(check_todo, "REPO", tmp_path)
    (tmp_path / "doc.md").write_text("the saving is 91%", encoding="utf-8")
    assert evaluate('present "91%" in doc.md')[0] is True
    assert evaluate('absent "91%" in doc.md')[0] is False
    assert evaluate('absent "83.7%" in doc.md')[0] is True


def test_an_unknown_predicate_raises_rather_than_passing():
    """A typo must not read as done.

    This is the failure the tool exists to prevent, one level up: a mistyped predicate that
    silently evaluated to "satisfied" would tick an item nobody had checked.
    """
    with pytest.raises(ValueError, match="unknown predicate"):
        evaluate("exsits README.md")
    with pytest.raises(ValueError, match="unknown predicate"):
        evaluate("contains 'x' in README.md")


def test_a_text_predicate_on_a_missing_file_raises(tmp_path, monkeypatch):
    """`absent "x" in nowhere.md` is vacuously true and meaningless; refuse it instead."""
    monkeypatch.setattr(check_todo, "REPO", tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        evaluate('absent "anything" in nowhere.md')


# --- the parser -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "- [ ] plain",
        "- [x] plain ticked",
        "- `[ ]` backticked",
        "- `[~]` in progress",
        "- `[!]` blocked",
        "  - [X] indented and capitalised",
    ],
)
def test_every_marker_form_this_repository_uses_is_recognised(line):
    """The file the tool reconciles uses all six; a parser that knew two would see a third of it."""
    assert LIST_LINE.match(line) and ITEM.search(line), f"{line!r} was not read as a checklist item"


def test_the_status_legend_is_not_four_items():
    """It carries four markers and is prose. The bullet anchor excludes it, with no exemption."""
    legend = "Status legend: `[ ]` open · `[~]` in progress · `[x]` done · `[!]` blocked"
    assert len(ITEM.findall(legend)) == 4, "the markers are there — that is exactly the trap"
    assert not LIST_LINE.match(legend), "and the line is not a list, which is what excludes it"


def test_every_marker_on_a_line_is_counted(checklist):
    """The retro-fix row packs seven items onto one line, and each is a real item.

    A first-marker-only parser reported one item there and counted the other six as absent — not as
    unverifiable, as *not present at all*, which is the worse of the two errors because it makes the
    file look shorter and better annotated than it is.
    """
    path = checklist("- `[ ]` **07** · `[ ]` **06** · `[ ]` **05** · `[x]` **04**\n")
    _, checked, unverifiable = reconcile(path)
    assert checked + unverifiable == 4


def test_an_annotation_binds_to_its_own_marker(checklist):
    """Two items on one line, one annotated: the predicate must not leak onto its neighbour."""
    path = checklist("- `[x]` **07** <!-- check: exists there.md --> · `[ ]` **06**\n")
    (path.parent / "there.md").write_text("x", encoding="utf-8")
    disagreements, checked, unverifiable = reconcile(path)
    assert (checked, unverifiable) == (1, 1)
    assert not disagreements


# --- what the tool reports --------------------------------------------------------------------


def test_an_unannotated_item_is_unverifiable_and_never_a_pass(checklist):
    """A checklist of thirty unannotated items must not report as thirty items in agreement."""
    path = checklist("- [ ] a\n- [x] b\n- `[~]` c\n")
    disagreements, checked, unverifiable = reconcile(path)
    assert (checked, unverifiable) == (0, 3)
    assert not disagreements, "an unverifiable item is not a disagreement either — it is unknown"


def test_a_done_but_unticked_item_is_reported(checklist):
    """The case that made the tool: work finished, the list still saying it is open."""
    path = checklist("- [ ] write it <!-- check: exists done.md -->\n")
    (path.parent / "done.md").write_text("x", encoding="utf-8")
    disagreements, _, _ = reconcile(path)
    assert len(disagreements) == 1
    assert "is DONE but unticked" in disagreements[0]


def test_a_ticked_but_undone_item_is_reported(checklist):
    """The other direction, which is the one that misleads a reader rather than merely nagging."""
    path = checklist("- [x] write it <!-- check: exists never.md -->\n")
    disagreements, _, _ = reconcile(path)
    assert len(disagreements) == 1
    assert "is ticked but NOT done" in disagreements[0]


def test_a_blocked_item_whose_predicate_passes_is_reported(checklist):
    """`[!]` is not an exemption. An unblocked item still marked blocked is a stale entry."""
    path = checklist("- `[!]` waiting <!-- check: exists arrived.md -->\n")
    (path.parent / "arrived.md").write_text("x", encoding="utf-8")
    disagreements, _, _ = reconcile(path)
    assert len(disagreements) == 1


def test_a_bad_predicate_is_a_disagreement_rather_than_a_crash(checklist):
    """One typo must fail that item and let the rest of the file be reconciled."""
    path = checklist(
        "- [ ] typo <!-- check: exsits x.md -->\n- [x] fine <!-- check: exists fine.md -->\n"
    )
    (path.parent / "fine.md").write_text("x", encoding="utf-8")
    disagreements, checked, _ = reconcile(path)
    assert checked == 2
    assert len(disagreements) == 1 and "unknown predicate" in disagreements[0]


# --- the command line -------------------------------------------------------------------------


def test_the_exit_code_is_non_zero_only_when_something_disagrees(checklist, capsys):
    """A reconciler nobody can put in a script is a reconciler nobody runs twice."""
    path = checklist("- [x] done <!-- check: exists yes.md -->\n")
    (path.parent / "yes.md").write_text("x", encoding="utf-8")
    assert main([str(path)]) == 0

    checklist("- [ ] done <!-- check: exists yes.md -->\n")
    assert main([str(path)]) == 1
    assert "is DONE but unticked" in capsys.readouterr().out


def test_a_missing_checklist_is_reported_rather_than_treated_as_empty(tmp_path, capsys):
    """An empty file and an absent one both have zero disagreements; only one of them is fine."""
    assert main([str(tmp_path / "nothing.md")]) == 2
    assert "no checklist" in capsys.readouterr().out


def test_the_stamp_records_the_counts_and_replaces_itself(checklist, capsys):
    """Stamping twice must leave one stamp, or the file grows a stamp per run."""
    path = checklist("- [x] done <!-- check: exists yes.md -->\n- [ ] unannotated\n")
    (path.parent / "yes.md").write_text("x", encoding="utf-8")

    main([str(path), "--stamp"])
    first = path.read_text(encoding="utf-8")
    assert first.count(check_todo.STAMP) == 1
    assert "1 of 2 items checkable" in first

    main([str(path), "--stamp"])
    assert path.read_text(encoding="utf-8").count(check_todo.STAMP) == 1
    capsys.readouterr()


def test_it_runs_as_a_script(tmp_path):
    """The documented invocation is `uv run python tools/check_todo.py <path>`; run it that way."""
    checklist = tmp_path / "TODO.md"
    checklist.write_text("- [x] the tool itself <!-- check: exists tools/check_todo.py -->\n")
    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "check_todo.py"), str(checklist)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1 checkable" in done.stdout


def test_a_needle_may_be_single_quoted(tmp_path, monkeypatch):
    """Because a needle containing a double quote has to be writable at all.

    The first attempt wrote `absent "id=\\"rail\\"" in index.html`. The regex read the
    backslashes as
    part of the needle, so the predicate asked whether a string nothing contains was absent — true
    everywhere — and reported an unfinished item as DONE. A quoting scheme with no escape is
    fine; an
    escape that silently changes the question is not, so there is no escape character.
    """
    monkeypatch.setattr(check_todo, "REPO", tmp_path)
    (tmp_path / "page.html").write_text('<aside id="rail"></aside>', encoding="utf-8")
    assert evaluate("""present 'id="rail"' in page.html""")[0] is True
    assert evaluate("""absent 'id="rail"' in page.html""")[0] is False
    trap = r"""absent "id=\"rail\"" in page.html"""
    assert evaluate(trap)[0] is True, (
        "the backslashes are part of the needle, so this asks whether a string nothing contains is "
        "absent — true of every file, and the reason an unfinished item read as DONE"
    )
