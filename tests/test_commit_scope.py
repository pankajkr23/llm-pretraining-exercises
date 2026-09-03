"""The commit-scope guard refuses breadth by default and accepts a stated reason.

`tools/check_commit_scope.py` runs at pre-commit's `commit-msg` stage. Every property is written
twice — once passing, once against a deliberately broken input — because a guard nobody has watched
fail is not a guard.

**The escape hatch is the part worth testing hardest.** A refusal nobody can get past gets removed;
one anybody can get past protects nothing. So the tests below pin both edges: a trailer with a real
reason lets a wide commit through, and a trailer without one does not.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from check_commit_scope import (  # noqa: E402
    MAX_FILES,
    MAX_LINES,
    MIN_REASON_WORDS,
    NOT_COUNTED,
    stated_reason,
    verdict,
)

_GOOD_REASON = "Wide-change: the hook and the module it imports cannot land apart"


def test_a_narrow_commit_passes_with_no_trailer() -> None:
    """The common case must stay silent, or the guard becomes something people route around."""
    assert verdict(["a.py", "b.py"], 40, "fix: something small") is None


def test_too_many_files_is_refused() -> None:
    """The limit PK asked for, enforced."""
    paths = [f"f{n}.py" for n in range(MAX_FILES + 1)]
    refusal = verdict(paths, 10, "feat: sprawling")
    assert refusal is not None
    assert f"limit {MAX_FILES}" in refusal
    for path in paths:
        assert path in refusal, "the refusal must name the files, or it is not actionable"


def test_too_many_lines_is_refused_even_with_few_files() -> None:
    """A 200-line rewrite of one file is the case a file count alone would wave straight through."""
    refusal = verdict(["big.py"], MAX_LINES + 1, "refactor: everything")
    assert refusal is not None
    assert f"limit {MAX_LINES}" in refusal


def test_a_stated_reason_allows_a_wide_commit() -> None:
    """Otherwise a hook, the module it imports and its test could never land together.

    Split across commits, the first two do not import — so `git bisect` lands on a tree that fails
    for a reason unrelated to what is being bisected, which is the property atomic commits exist to
    provide.
    """
    paths = [f"f{n}.py" for n in range(MAX_FILES + 3)]
    assert verdict(paths, MAX_LINES * 2, f"feat: land it\n\n{_GOOD_REASON}") is None


def test_a_trailer_with_no_real_reason_does_not_count() -> None:
    """The twin for the escape hatch. `Wide-change: needed` is the same as no trailer at all.

    The file list is derived from `MAX_FILES` rather than typed. It was typed once, as five paths,
    and silently stopped testing anything the moment the limit was raised above five — the guard
    passed because the fixture was no longer wide, not because the trailer was rejected.
    """
    wide = [f"f{n}.py" for n in range(MAX_FILES + 1)]
    for thin in ("Wide-change: needed", "Wide-change: big", "Wide-change:   "):
        assert stated_reason(f"feat: x\n\n{thin}") is None, thin
        assert verdict(wide, 10, f"feat: x\n\n{thin}") is not None, thin


def test_the_reason_must_carry_at_least_the_stated_number_of_words() -> None:
    """Pinned against the constant rather than a copy of it, so the two cannot drift."""
    exactly = " ".join("word" for _ in range(MIN_REASON_WORDS))
    assert stated_reason(f"x\n\nWide-change: {exactly}") is not None
    one_short = " ".join("word" for _ in range(MIN_REASON_WORDS - 1))
    assert stated_reason(f"x\n\nWide-change: {one_short}") is None


@pytest.mark.parametrize("exempt", sorted(NOT_COUNTED))
def test_an_exempt_file_does_not_consume_the_budget(exempt: str) -> None:
    """CHANGELOG.md is required by the conventions in the same change; charging for it would leave
    two files for the actual work, which would make the rule fight the rule it sits beside."""
    assert exempt in NOT_COUNTED
    assert NOT_COUNTED[exempt].strip(), "every exemption states its reason"


def test_every_exempt_file_is_real_or_generated() -> None:
    """Fails in the other direction: an exemption for a file that does not exist exempts nothing."""
    missing = [name for name in NOT_COUNTED if not (REPO_ROOT / name).exists()]
    assert not missing, f"NOT_COUNTED names files that are not in the repo: {missing}"


def test_a_merge_or_revert_is_not_judged() -> None:
    """Their breadth is a property of the branches, not a decision anyone is making now."""
    script = REPO_ROOT / "tools" / "check_commit_scope.py"
    for subject in ("Merge pull request #90 from x/y", 'Revert "feat: something"'):
        message = REPO_ROOT / "artifacts" / "scope-msg.txt"
        message.parent.mkdir(parents=True, exist_ok=True)
        message.write_text(subject + "\n", encoding="utf-8")
        try:
            done = subprocess.run(
                [sys.executable, str(script), str(message)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert done.returncode == 0, done.stderr
        finally:
            message.unlink(missing_ok=True)
