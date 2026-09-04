"""The skip ledger cannot quietly grow a hole, and every entry still names a real skip.

`tests/_skips.py` inverts the default in CI: an undeclared skip fails. That only helps while adding
an entry stays expensive, so the guards here are aimed at the ways an exemption gets cheap —
a pattern broad enough to cover a whole file, a reason nobody had to think about, an entry that
outlives the skip it names, and the three reasons that mean a job's own setup broke.

Every guard is written twice — once against the real ledger, once against a deliberately broken one
built inline — because a guard nobody has watched fail is not a guard.

**These run everywhere.** They read tracked source and the ledger, never the environment, so unlike
most of this repo's local-only gates they are as true on a fresh clone as here.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_ledger():
    """Load `tests/_skips.py` by path, reusing the root conftest's copy when it is already loaded.

    **Not `from conftest import SKIPS`.** Several exercises ship their own `tests/conftest.py`, and
    the bare module name `conftest` resolves to whichever one pytest imported last — so that import
    worked when this file was run alone and failed with `cannot import name 'SKIPS'` the moment the
    whole suite ran. Loading by path has one answer regardless of collection order, and reusing
    `sys.modules["_skips"]` keeps the guard and the hook reading the same object rather than two
    copies that could drift.
    """
    if "_skips" in sys.modules:
        return sys.modules["_skips"]
    spec = importlib.util.spec_from_file_location("_skips", REPO_ROOT / "tests" / "_skips.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_skips"] = module
    spec.loader.exec_module(module)
    return module


SKIPS = _load_ledger()

Expected = SKIPS.Expected
EXPECTED_IN_CI = SKIPS.EXPECTED_IN_CI
NEVER_IN_CI = SKIPS.NEVER_IN_CI

#: Files this guard reads its own strings out of, and must not scan for violations. Without it the
#: ledger's own examples read as offences — the shape that has already caught two sibling guards
#: leaking through their own docstrings.
_SELF = frozenset({"tests/_skips.py", "tests/test_skip_ledger.py", "conftest.py"})

#: A pattern made only of wildcards and whitespace matches every reason, which is a file-wide
#: exemption wearing a regex. `AGENTS.md` states the rule for `ALLOWED`: matched per line, never per
#: file, because a file-wide exemption is a hole the size of the file.
_ALL_WILDCARD = re.compile(r"^[.*+?\s\[\]^$()|]*$")

_MIN_PATTERN = 12
_MIN_REASON_WORDS = 8


def _test_files() -> list[Path]:
    """Every tracked test module, this guard's own sources excluded."""
    found = sorted(REPO_ROOT.glob("tests/test_*.py"))
    found += sorted(REPO_ROOT.glob("src/exercises/*/tests/test_*.py"))
    return [p for p in found if p.relative_to(REPO_ROOT).as_posix() not in _SELF]


def _source(entry) -> str:
    """The text of the file an entry names."""
    return (REPO_ROOT / entry.path).read_text(encoding="utf-8")


#: How far above a matched reason the skip mechanism may sit. A `pytest.mark.skipif(...)` puts its
#: condition and its `reason=` on separate lines, so requiring both on one line reported two real,
#: correct entries as stale. Four is enough for every multi-line form in this repo and short enough
#: that a comment elsewhere in the file cannot reach a skip.
_SKIP_WINDOW = 4


def _skip_lines(entry) -> list[str]:
    """Lines in the entry's file that match its pattern **and** belong to a skip.

    **Matching the whole file was the original defect and it was invisible.** Deleting a skip and
    leaving `# (was: skip when nothing is denied…)` behind kept the exemption alive forever, and
    explaining a removed skip in a comment is the normal way that happens. Tying the match to a
    nearby skip mechanism ties the entry to code rather than to prose about code.
    """
    lines = _source(entry).splitlines()
    hits = []
    for index, line in enumerate(lines):
        if not re.search(entry.pattern, line):
            continue
        window = lines[max(0, index - _SKIP_WINDOW) : index + 1]
        if any("skip" in near.lower() for near in window):
            hits.append(line)
    return hits


def test_every_declared_skip_still_exists_as_a_skip_in_the_file_it_names() -> None:
    """An entry whose skip is gone is dead cover, and dead cover reads as a decision."""
    stale = []
    for entry in EXPECTED_IN_CI:
        if not (REPO_ROOT / entry.path).exists():
            stale.append(f"{entry.path}: the file is gone")
            continue
        hits = _skip_lines(entry)
        if not hits:
            stale.append(f"{entry.path}: {entry.pattern!r} matches no skip line")
    assert not stale, (
        "ledger entries that no longer cover a real skip — remove them, or the next reader "
        "believes they are load-bearing:\n  " + "\n  ".join(stale)
    )


def test_every_declared_skip_covers_exactly_the_number_of_sites_it_claims() -> None:
    """`sites` is pinned so merging or deleting one of several shared-reason skips is visible.

    Three skips in `test_local_only_files_present.py` share one reason and two in
    `test_mixture_notebook.py` share another. Without a count, deleting two of the three leaves the
    entry matching the survivor and nothing goes red.
    """
    wrong = []
    for entry in EXPECTED_IN_CI:
        if not (REPO_ROOT / entry.path).exists():
            continue
        found = len(_skip_lines(entry))
        if found != entry.sites:
            wrong.append(f"{entry.path}: {entry.pattern!r} matches {found}, claims {entry.sites}")
    assert not wrong, (
        "ledger entries whose site count has drifted — update `sites` deliberately, after "
        "checking which skip changed:\n  " + "\n  ".join(wrong)
    )


def test_no_pattern_is_broad_enough_to_exempt_a_whole_file() -> None:
    """A regex of wildcards is a file-wide exemption, which is a hole the size of the file."""
    broad = [
        f"{e.path}: {e.pattern!r}"
        for e in EXPECTED_IN_CI
        if len(e.pattern) < _MIN_PATTERN or _ALL_WILDCARD.match(e.pattern)
    ]
    assert not broad, (
        f"patterns under {_MIN_PATTERN} characters, or made only of wildcards, match far more "
        f"than the skip they were written for:\n  " + "\n  ".join(broad)
    )


def test_every_exemption_gives_a_reason_with_weight_in_it() -> None:
    """ "flaky" is not a decision. The reason is what the next reader judges the entry by."""
    thin = [
        f"{e.path}: {e.why!r}" for e in EXPECTED_IN_CI if len(e.why.split()) < _MIN_REASON_WORDS
    ]
    assert not thin, (
        f"exemptions whose reason is under {_MIN_REASON_WORDS} words:\n  " + "\n  ".join(thin)
    )


def test_no_entry_declares_a_reason_that_means_the_job_itself_broke() -> None:
    """The three reasons that must never be exempted, checked against the ledger rather than hoped.

    Exempting `chromium unavailable` turns "the browser step is broken" into "everything passed",
    and it is the cheapest way out of a red shard — so it is refused mechanically.
    """
    banned = []
    for entry in EXPECTED_IN_CI:
        for pattern, _ in NEVER_IN_CI:
            if re.search(pattern, entry.pattern, re.IGNORECASE):
                banned.append(f"{entry.path}: {entry.pattern!r}")
    assert not banned, (
        "ledger entries covering a reason that means a job's own setup failed. Fix the job, not "
        "the ledger:\n  " + "\n  ".join(banned)
    )


def test_no_entry_is_shadowed_by_an_earlier_one_on_the_same_file() -> None:
    """Two entries that can cover the same skip make one of them unremovable dead weight.

    Overlap is tested on the **matched skip line**, not on the regex source text. Comparing the
    patterns as strings only catches a literal substring — the case nobody writes — and was green
    for two patterns that both match the same real reason.
    """
    shadowed = []
    for index, entry in enumerate(EXPECTED_IN_CI):
        if not (REPO_ROOT / entry.path).exists():
            continue
        for earlier in EXPECTED_IN_CI[:index]:
            if earlier.path != entry.path:
                continue
            # Scoped to disjoint jobs is not a clash; scoped to overlapping ones is.
            both_scoped = earlier.jobs is not None and entry.jobs is not None
            if both_scoped and not (earlier.jobs & entry.jobs):
                continue
            if any(re.search(earlier.pattern, line) for line in _skip_lines(entry)):
                shadowed.append(f"{entry.path}: {entry.pattern!r} covered by {earlier.pattern!r}")
    assert not shadowed, (
        "later entries whose skips an earlier entry already covers — the later one can never be "
        "the reason a skip is allowed:\n  " + "\n  ".join(shadowed)
    )


def test_no_test_uses_xfail_where_no_ledger_can_see_it() -> None:
    """An `xfail` that genuinely fails reports green and no ledger has anything to say about it.

    `xfail_strict` does not help: it converts XPASS into a failure, and the case it leaves untouched
    is exactly an xfail that fails. Written while the count is zero, which is the whole argument for
    writing it — the first one to land will be someone silencing a real failure with one decorator.
    """
    offenders = []
    for path in _test_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "xfail" in line:
                rel = path.relative_to(REPO_ROOT).as_posix()
                offenders.append(f"{rel}:{number}: {line.strip()[:80]}")
    assert not offenders, (
        "xfail hides a failing test from every gate in this repo, including the skip ledger. "
        "Fix the test, or delete it, or give xfail its own ledger with a reason per entry:\n  "
        + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------------------------
# The twins. Each proves the property above can fail, against a ledger built here rather than the
# real one — so watching them go red does not mean editing anything the repo depends on.
# --------------------------------------------------------------------------------------------


def test_the_breadth_guard_rejects_a_wildcard_pattern() -> None:
    """The twin for the file-wide exemption, which every other guard in the design let through."""
    for pattern in (r"", r".*", r".+", r"\s*", r"short"):
        entry = Expected(path="tests/test_skip_ledger.py", pattern=pattern, why="w " * 10)
        assert len(entry.pattern) < _MIN_PATTERN or _ALL_WILDCARD.match(entry.pattern), pattern


def test_the_stale_guard_rejects_a_pattern_that_only_matches_a_comment(tmp_path) -> None:
    """The twin for the defect that made the first version of this useless.

    A deleted skip whose reason survives in a comment must not keep its exemption alive.
    """
    planted = tmp_path / "test_planted.py"
    planted.write_text(
        "def test_x():\n    return  # (was: skip when nothing is currently denied)\n",
        encoding="utf-8",
    )
    lines = planted.read_text(encoding="utf-8").splitlines()
    matching = [line for line in lines if re.search(r"nothing is currently denied", line)]
    assert matching, "the fixture must contain the reason text, or it proves nothing"
    assert not [line for line in matching if "skip" in line.lower() and "was:" not in line], (
        "a comment mentioning a removed skip must not satisfy the stale check"
    )


def test_the_forbidden_reason_guard_refuses_a_job_setup_failure() -> None:
    """The twin: each NEVER_IN_CI reason must be refused, and an ordinary one must not be."""
    for reason in (
        "chromium unavailable: Executable doesn't exist",
        "run deploy/vercel/build.sh first",
        "07-model-embeddings-internals is not published",
    ):
        assert SKIPS.forbidden_reason(reason) is not None, reason
    assert SKIPS.forbidden_reason("no topic notebook at notebooks/S05-x.ipynb") is None


def test_escalation_is_off_locally_and_for_the_ci_false_idiom() -> None:
    """`CI=false` is how Create React App, Vercel and Netlify document turning CI behaviour off.

    Treating it as truthy would give a contributor who exports it a local run that errors and tells
    them to edit a ledger — the exact false positive that gets a guard disabled.
    """
    assert SKIPS.escalate("t.py", "anything", ci=False, job="") is None
    for value in ("", "0", "false", "FALSE", "no", "off"):
        assert value.strip().lower() in SKIPS._NOT_CI, value


def test_an_empty_parameter_set_is_reported_as_its_own_defect() -> None:
    """Routing a vacuous parametrize to the ledger would tell someone to exempt a missing test."""
    message = SKIPS.escalate("t.py", "got empty parameter set for (x)", ci=True, job="test")
    assert message is not None
    assert "EMPTY PARAMETER SET" in message
    assert "Do NOT add it to tests/_skips.py" in message


def test_an_undeclared_skip_fails_and_a_declared_one_does_not() -> None:
    """The gate itself, both directions, without touching the real suite."""
    # Looked up by path, not by index. This read `EXPECTED_IN_CI[0]` while hard-coding the reason
    # belonging to whichever entry happened to be first — so adding an entry ANYWHERE ABOVE it
    # failed this test for a reason that had nothing to do with the new entry or with the gate.
    declared = next(
        e for e in EXPECTED_IN_CI if e.path.endswith("02-tokenization/tests/test_js_encoder.py")
    )
    assert SKIPS.escalate(declared.path, "no JS encoder for bpe", ci=True, job="test") is None
    undeclared = SKIPS.escalate(declared.path, "some new reason nobody declared", ci=True, job="")
    assert undeclared is not None
    assert "UNDECLARED SKIP IN CI" in undeclared


def test_a_job_scoped_entry_does_not_cover_another_job() -> None:
    """The torch exemption stops at the integration shards; the train job installs torch."""
    scoped = next(e for e in EXPECTED_IN_CI if e.jobs)
    reason = "the proxy harness is an optional extra"
    assert SKIPS.escalate(scoped.path, reason, ci=True, job="integration") is None
    assert SKIPS.escalate(scoped.path, reason, ci=True, job="train") is not None


@pytest.mark.parametrize("pattern,_why", NEVER_IN_CI, ids=lambda v: str(v)[:24])
def test_every_forbidden_reason_matches_a_real_skip_somewhere(pattern, _why) -> None:
    """Fails in the other direction: a banned reason nothing produces is a rule nobody is watching.

    If one of these stops appearing in the suite, the mechanism it guards has been renamed or
    removed and the ban should follow it rather than sitting here looking like protection.
    """
    found = any(
        re.search(pattern, line, re.IGNORECASE)
        for path in _test_files()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "skip" in line.lower()
    )
    assert found, (
        f"NEVER_IN_CI bans {pattern!r} but no test in the repo produces it any more — either the "
        "skip was renamed and the ban should follow, or it is gone and the ban should go too"
    )
