"""Mutation testing: disable each guard in turn and prove the suite notices.

The repo's rule is that *"a guard that cannot fail is worse than no guard, because it reads as
coverage"*, and that every invariant is watched to fail before it is trusted. The twins in
`test_mixture_invariants.py` do that one rule at a time, by hand.

This does it mechanically and for all of them at once. Each `check_*` function in `checks.py` is
rewritten to return no findings, the fast suite is run, and the mutant must **die** — some test
must go red. A mutant that survives means the guard it disabled is decorative: nothing in the
suite depends on it doing anything.

Integration-marked because it runs the whole fast suite once per guard. It is the test that makes
the other tests trustworthy, so it is worth its runtime whenever the guards change.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from mixture import checks

CHECKS = Path(checks.__file__)
EXERCISE = CHECKS.parents[2]

# Matches a function definition up to and including the end of its docstring, so the injected
# early return lands after it rather than replacing it — a mutant that also deleted the docstring
# could fail for the wrong reason.
_DEF_THROUGH_DOCSTRING = r'(^def {name}\(.*?\n(?:.*?\n)*?    """.*?"""\n)'


def _check_names(source: str) -> list[str]:
    """Every guard defined in `checks.py`.

    Args:
        source: The module's source.

    Returns:
        Function names, in file order.
    """
    return re.findall(r"^def (check_[a-z_0-9]+)\(", source, re.MULTILINE)


def _run_fast_suite() -> int:
    """Run this exercise's fast tests in a subprocess.

    A subprocess is required: the mutant has to be imported fresh, and this test module has
    already imported the real `checks`.

    Returns:
        The pytest exit code.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(EXERCISE / "tests"),
            "-q",
            "--no-header",
            "-x",
            "-p",
            "no:cacheprovider",
            "-m",
            "not integration",
        ],
        capture_output=True,
        text=True,
        cwd=EXERCISE.parents[2],
    ).returncode


@pytest.mark.integration
def test_every_guard_is_load_bearing():
    """Disabling any single guard must make some test fail.

    Restores the original file in a `finally` even if pytest is interrupted, because leaving a
    neutered `checks.py` on disk would be a far worse outcome than a failing test.
    """
    original = CHECKS.read_text(encoding="utf-8")
    backup = CHECKS.with_suffix(".py.mutation-backup")
    shutil.copy(CHECKS, backup)

    names = _check_names(original)
    assert names, "no guards found to mutate — the pattern or the module has moved"

    survivors: list[str] = []
    unmutated: list[str] = []
    try:
        for name in names:
            pattern = re.compile(_DEF_THROUGH_DOCSTRING.format(name=name), re.MULTILINE | re.DOTALL)
            mutated, count = pattern.subn(r"\1    return []\n", original, count=1)
            if count != 1:
                unmutated.append(name)
                continue
            CHECKS.write_text(mutated, encoding="utf-8")
            if _run_fast_suite() == 0:
                survivors.append(name)
    finally:
        shutil.move(str(backup), str(CHECKS))

    assert CHECKS.read_text(encoding="utf-8") == original, "checks.py was not restored"
    assert not unmutated, f"could not mutate: {unmutated}"
    assert not survivors, (
        f"these guards can be disabled with the suite still green, so nothing depends on them "
        f"doing anything: {survivors}"
    )
