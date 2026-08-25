"""No two test modules may share a basename.

pytest imports test modules by basename when their directories have no `__init__.py`, which is the
layout every exercise here uses. Two files called `test_config.py` therefore collide, and the
failure is a *collection error*, not a test failure:

    import file mismatch:
    imported module 'test_config' has this __file__ attribute:
      .../03-data-collection-framework/tests/test_config.py
    which is not the same as the test file we want to collect:
      .../06-build-training-dataset/tests/test_config.py

It aborts the run rather than reporting a red test, so it is worth catching by name. Exercise 05
already solves it by convention with a `test_mixture_*` prefix; this makes the convention checkable
instead of remembered.
"""

from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _test_modules() -> dict[str, list[str]]:
    """Map each test module basename to the directories that define it."""
    seen: dict[str, list[str]] = defaultdict(list)
    for path in REPO_ROOT.glob("**/test_*.py"):
        if "__pycache__" in path.parts or ".venv" in path.parts or "public" in path.parts:
            continue
        seen[path.name].append(str(path.parent.relative_to(REPO_ROOT)))
    return dict(seen)


def test_no_two_test_modules_share_a_basename() -> None:
    """A collision aborts collection, so it must be caught before it is committed."""
    clashes = {name: dirs for name, dirs in _test_modules().items() if len(dirs) > 1}
    assert not clashes, (
        f"these test module basenames are defined more than once: {clashes}. pytest imports test "
        f"modules by basename, so this aborts collection with an 'import file mismatch'. Prefix "
        f"the new one with its package, as exercise 05 does with `test_mixture_*`."
    )


def test_the_basename_scan_can_actually_fail() -> None:
    """The twin. A glob that matched nothing would make the check above vacuous forever."""
    found = _test_modules()
    assert len(found) > 30, f"the scan found only {len(found)} test modules — the glob has drifted"
    assert "test_module_names.py" in found, "the scan cannot even see this file"

    fake = {"test_config.py": ["a/tests", "b/tests"], "test_unique.py": ["c/tests"]}
    assert {n: d for n, d in fake.items() if len(d) > 1} == {
        "test_config.py": ["a/tests", "b/tests"]
    }, "the duplicate predicate does not isolate the clash"
