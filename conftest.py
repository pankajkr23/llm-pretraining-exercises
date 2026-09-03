"""In CI, a skip nobody declared is a failure — because pytest reports it as a pass.

**The hole this closes.** `tests/test_ci_shards_cover_everything.py` stops a whole FILE from
vanishing behind a module-level `pytest.importorskip`, and its docstring names the case it does not
cover: an indented skip "skips a single test and shows up in the skip report". Nothing here ever
read that report. The dangerous reasons are environmental — `chromium unavailable`,
`run deploy/vercel/build.sh first`, `{slug} is not published` — and each fires inside a job that has
just installed or built the thing it is checking for. If that step breaks, every browser assertion
in the repo becomes SKIPPED and the job stays green.

**Why a hook rather than another copy of the guard.** Exercises 02–05 each carry an
`if os.environ.get("CI"): pytest.fail(...)` immediately before their chromium skip. Exercises 06, 07
and 08 and both repo-level browser suites do not — the guard was written five times and copied
forward zero. A rule enforced by copying applies to whoever remembers it. One hook inverts the
default everywhere at once, and `tests/_skips.py::EXPECTED_IN_CI` is where an exemption has to be
argued for.

**What it deliberately cannot see.** A skip raised during *collection* — a module-level
`importorskip`, or `pytest.skip(allow_module_level=True)` — produces a collect report and no `item`,
so it never reaches this hook. That is the case the shard ledger already covers lexically, which is
the right tool for it: an `importorskip` line is a fact about the source, true whatever happens to
be installed. This hook covers what that ledger cannot — skips whose condition is the environment
the job just built.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent


def _ledger():
    """The skip ledger, loaded by path rather than by name.

    `tests/` reaches `sys.path` only once pytest imports something out of it, and this file is
    imported before any test module. Loading by path removes the ordering question, and registering
    the result as `_skips` means `tests/test_skip_ledger.py`'s own import gets this same object
    rather than a second copy that could drift.

    Returns:
        The `_skips` module.
    """
    if "_skips" in sys.modules:
        return sys.modules["_skips"]
    spec = importlib.util.spec_from_file_location("_skips", REPO_ROOT / "tests" / "_skips.py")
    assert spec and spec.loader, "tests/_skips.py is missing; the skip ledger cannot be read"
    module = importlib.util.module_from_spec(spec)
    sys.modules["_skips"] = module
    spec.loader.exec_module(module)
    return module


SKIPS = _ledger()


def _relative(item: pytest.Item) -> str:
    """The test file's repo-relative POSIX path.

    Args:
        item: The test item being reported.

    Returns:
        A repo-relative path, falling back to pytest's own location for a file outside the
        repository — which only happens for a test generated into a temporary directory.
    """
    try:
        return item.path.resolve().relative_to(REPO_ROOT).as_posix()
    except (AttributeError, ValueError):
        return str(item.location[0]).replace(os.sep, "/")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Rewrite an undeclared skip as a failure while CI is running.

    Args:
        item: The test item being reported.
        call: The phase being reported.

    Yields:
        Control to the next hook implementation.

    Returns:
        The report, with an undeclared CI skip rewritten as a failure.
    """
    report = yield
    # `wasxfail` marks an xfail. Those are refused outright by `test_skip_ledger.py` rather than
    # handled here, because an xfail that genuinely fails reports green and no ledger sees it.
    if not report.skipped or hasattr(report, "wasxfail"):
        return report
    message = SKIPS.escalate(
        _relative(item),
        SKIPS.reason_of(report.longrepr),
        ci=SKIPS.escalating(),
        job=SKIPS.job_name(),
    )
    if message is not None:
        report.outcome = "failed"
        report.longrepr = message
    return report
