"""A lock between the test that rewrites `checks.py` and the tests that digest it.

`test_mixture_mutation.py` disables each guard in `mixture/checks.py` by rewriting the file on
disk. `mixture.checks` is inside `provenance.code_digest`, and every producer refuses to record a
bundle when that digest moves while it runs. So under `pytest -n auto`, a producer test on another
worker read a different digest before and after its two steps and refused: a correct refusal
caused by the suite, not by the code.

A file lock, because the workers are separate processes. The mutation test holds it exclusively for
its whole run; each producer test holds it shared, so producers still run beside one another. The
lock file lives in the gitignored `artifacts/`.
"""

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

LOCK = Path(__file__).resolve().parents[1] / "artifacts" / ".checks-mutation.lock"


@contextmanager
def _held(mode: int) -> Iterator[None]:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as handle:
        fcntl.flock(handle, mode)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@pytest.fixture
def stable_checks() -> Iterator[None]:
    """For a test whose provenance digests `checks.py`: no mutant is on disk while it runs."""
    with _held(fcntl.LOCK_SH):
        yield


@pytest.fixture
def mutating_checks() -> Iterator[None]:
    """For the mutation test: no producer test is running while `checks.py` is rewritten."""
    with _held(fcntl.LOCK_EX):
        yield
