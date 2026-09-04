"""A pull request that merges without logging itself breaks every other branch.

**This has now cost three rounds of merging, and each time the symptom pointed somewhere else.**
`tools/queue_status.py` refuses when `docs/agents/QUEUE.md` does not record a merged pull request,
and it is wired into the `test` job. So the moment one merges unlogged, `main` fails its own gate —
and **every branch cut from `main` inherits that failure**, reporting a problem that has nothing to
do with the branch. #133 sat red twice for exactly this: once for #108's misplaced entry, once for
#134 not logging itself.

The existing check is `test_queue_status.py`, and it looks *backwards*: it asks whether the log
records what has already merged. By the time it fails, the damage is on `main` and spread across
every open branch. This one looks *forwards* — it asks whether the pull request being tested right
now records itself — so the branch that would cause the problem is the branch that goes red.

**It only runs on a pull-request build**, where GitHub supplies the number in `GITHUB_REF` as
`refs/pull/<n>/merge`. There is nothing to check on a push to `main` or on a local run, and the
skip that produces is declared in `tests/_skips.py`.
"""

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE = REPO_ROOT / "docs" / "agents" / "QUEUE.md"

#: GitHub's ref for a pull-request build. `refs/heads/main` on a push, which is why this is matched
#: rather than assumed.
_PULL_REF = re.compile(r"^refs/pull/(\d+)/")

#: The checker reads this section and nothing above it, and so does this test — a number mentioned
#: in the prose is discussion, not a record. Getting that wrong is what let an entry land forty
#: lines too high and still look present to every `grep`.
_LOG_HEADING = "## Log"


def _pull_request_number() -> str | None:
    """The pull request this build is for, or None when it is not one."""
    for var in ("GITHUB_REF", "GITHUB_HEAD_REF_NUMBER"):
        match = _PULL_REF.match(os.environ.get(var, ""))
        if match:
            return match.group(1)
    return None


def test_this_pull_request_records_itself_in_the_log() -> None:
    """Forwards, not backwards: the branch that would break `main` is the one that fails."""
    number = _pull_request_number()
    if number is None:
        pytest.skip("not a pull-request build, so there is no pull request to look for")

    _, marker, log = QUEUE.read_text(encoding="utf-8").partition(_LOG_HEADING)
    assert marker, f"{QUEUE.name} has no `{_LOG_HEADING}` section to read"
    assert re.search(rf"#{number}\b", log), (
        f"{QUEUE.name}'s log does not mention #{number}.\n\n"
        "A pull request records itself when it OPENS — that convention is in AGENTS.md and in the "
        "log's own header. When one merges unlogged, `tools/queue_status.py` refuses on `main` and "
        "every branch cut from `main` then fails the `test` job for a reason that has nothing to "
        "do with it. Add an entry under `## Log` saying what this change cost, not what it did.\n\n"
        "A mention above the heading does not count: the checker reads the log section only."
    )


def test_the_number_is_read_from_the_ref_github_actually_sets() -> None:
    """The parser is the part that can silently stop working, so it is tested directly.

    If `GITHUB_REF` ever stops matching, `_pull_request_number` returns None, the test above skips,
    and the guard quietly covers nothing — the failure mode this file exists to prevent, one level
    up. The skip is declared in `tests/_skips.py`, so an unexpected one is a CI failure.
    """
    cases = {
        "refs/pull/133/merge": "133",
        "refs/pull/7/head": "7",
        "refs/heads/main": None,
        "refs/tags/v0.13.0": None,
        "": None,
    }
    for ref, expected in cases.items():
        match = _PULL_REF.match(ref)
        got = match.group(1) if match else None
        assert got == expected, f"{ref!r} parsed as {got!r}, expected {expected!r}"
