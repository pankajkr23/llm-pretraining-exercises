"""Nothing in the suite may collide when pytest runs tests in parallel.

CI runs `pytest -n auto --dist loadfile`. `loadfile` keeps every test in a file on one worker, and
that is what makes the pattern below safe: several suites write a small JavaScript harness *beside*
the module under test — a fixed path, deleted in a `finally` — because ES module imports resolve
against the importing file's own directory, so a temp directory will not do.

Plain `-n auto` splits a file across workers and one worker deletes the harness another is running:
4 errors, reproducibly. `--dist loadfile` removes that, **provided no two test files write the same
path**. This module pins that proviso, because it is invisible until someone adds a third harness
and reuses a name.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: `name = WEB / "_something.mjs"` — a module-level fixed path a test writes to.
_HARNESS = re.compile(
    r'^\s*(?:harness|script|probe_file|payload_path)\s*=\s*(\w+)\s*/\s*"([^"]+)"',
    re.M,
)


#: Bases that are already per-test private. `tmp_path` is pytest's own per-test directory, so two
#: tests writing `tmp_path / "probes.json"` never touch the same file and are not a collision.
_SAFE_BASES = frozenset({"tmp_path", "tmpdir", "tmp_dir"})


def _declared_harness_paths() -> dict[str, set[str]]:
    """Map each fixed harness path to the set of test files that write it.

    A *set*: the same file writing one path from two places is not a collision, only two different
    files are. Getting that wrong is how this guard first reported `tmp_path/probes.json`, which is
    both per-test private and declared twice inside a single module.
    """
    found: dict[str, set[str]] = {}
    for test_file in sorted(REPO_ROOT.glob("src/exercises/*/tests/test_*.py")):
        for base, name in _HARNESS.findall(test_file.read_text(encoding="utf-8")):
            if base in _SAFE_BASES:
                continue
            # `base` is a module constant (WEB, REPO...). Its value differs per exercise, so the
            # exercise directory plus the literal name is enough to tell collisions apart.
            key = f"{test_file.parents[1].name}/{base}/{name}"
            found.setdefault(key, set()).add(test_file.name)
    return found


def test_no_two_test_files_write_the_same_harness_path() -> None:
    """Two files sharing a path would race even under `--dist loadfile`.

    `loadfile` groups by *file*, so two different files can still run concurrently. A shared path
    is then exactly the bug `loadfile` was adopted to prevent, reintroduced by name reuse.
    """
    collisions = {
        path: sorted(files) for path, files in _declared_harness_paths().items() if len(files) > 1
    }
    assert not collisions, (
        f"these fixed paths are written by more than one test file: {collisions}. "
        f"Under `-n` those files can run concurrently and delete each other's harness. "
        f"Give each file its own filename."
    )


def test_the_collision_check_can_actually_fail() -> None:
    """The twin. Without it, a regex that matches nothing would pass forever."""
    found = _declared_harness_paths()
    assert found, (
        "the harness-path scan found nothing at all — the pattern has drifted from the code it "
        "is meant to watch, so the check above is vacuous"
    )

    # Two DIFFERENT files pointing at one path is the shape the guard must reject.
    fake = {"05/WEB/_agreement_harness.mjs": {"a.py", "b.py"}}
    assert {p: sorted(f) for p, f in fake.items() if len(f) > 1}, (
        "the collision predicate does not flag a genuine duplicate"
    )

    # ...and one file declaring the same path twice must NOT be flagged.
    same_file = {"05/WEB/_agreement_harness.mjs": {"a.py"}}
    assert not {p: sorted(f) for p, f in same_file.items() if len(f) > 1}, (
        "the predicate flags a single file as colliding with itself"
    )


def test_ci_runs_the_suite_with_dist_loadfile() -> None:
    """The safety of every harness above depends on this flag, so assert CI still passes it.

    Dropping `--dist loadfile` for plain `-n auto` reintroduces the 4 errors, and it would look
    like a harmless simplification in a diff.
    """
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    runs = [line for line in ci.splitlines() if "pytest" in line and "-n auto" in line]
    assert runs, "CI no longer runs pytest in parallel; this module's premise has changed"
    missing = [line.strip() for line in runs if "--dist loadfile" not in line]
    assert not missing, (
        f"these CI pytest invocations run in parallel WITHOUT `--dist loadfile`: {missing}. "
        f"That splits a file across workers and races the JS harnesses."
    )
