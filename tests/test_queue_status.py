"""The progress log cannot silently fall behind what merged.

`docs/agents/QUEUE.md` is the single source of truth for progress. It once carried one line reading
*"no unit has run yet"* while nine pull requests merged past it, and nothing noticed — so the point
of these tests is not that the checker runs, but that it **refuses**. Every property is written
twice, once passing and once failing, because a checker that cannot fail would be the most
convincing possible way to prove the log is current.

**These run everywhere.** They drive the functions with synthetic text rather than reading the
repository's real history, so a shallow CI clone changes nothing about them.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from queue_status import (  # noqa: E402
    QUEUE,
    append_stubs,
    logged_numbers,
    stub,
)

_LOG = """# The unit queue

Prose above the log may mention #999 without that counting as a record of it.

## Log

```
2026-09-03  backlog  #88 merged: something
2026-09-03  backlog  #90 merged: something else
```
"""


def test_a_number_in_the_prose_does_not_count_as_a_log_entry() -> None:
    """Otherwise a file that *discusses* merges passes as one that *records* them.

    This is the difference between a guard and a word search. The queue's own prose names pull
    requests constantly — describing what is blocked, what a unit found — and counting those would
    let the log go empty while the check stayed green.
    """
    numbers = logged_numbers(_LOG)
    assert numbers == {"88", "90"}, numbers
    assert "999" not in numbers, "a number above the log heading was counted as an entry"


def test_a_stub_asks_for_the_part_the_tool_cannot_know() -> None:
    """A generated entry that reads as finished is worse than one that reads as unfinished.

    The subject is copied because it is a fact; the rest is a `TODO` because what a change cost —
    what went red first, which guard caught it — is known only to whoever did the work. A stub left
    unfilled is visible in review. A plausible sentence nobody wrote is not.
    """
    text = stub("94", "a receipt so CI can prove the gate ran")
    assert "#94" in text
    assert "a receipt so CI can prove the gate ran" in text
    assert "TODO" in text, "a stub must not read as a finished entry"


def test_appending_puts_stubs_inside_the_fenced_block() -> None:
    """Outside the fence the entry renders as prose and the next check cannot find it."""
    appended = append_stubs(_LOG, [("94", "the receipt"), ("95", "the invariants")])
    assert logged_numbers(appended) >= {"88", "90", "94", "95"}
    body = appended.partition("## Log")[2]
    fenced = body.split("```")[1]
    assert "#94" in fenced and "#95" in fenced, "stubs landed outside the fenced log block"


def test_the_checker_refuses_a_log_that_is_behind_and_says_which() -> None:
    """The twin of the whole tool. Driven end to end, because the exit code is what the hook reads.

    A missing entry must name the pull request. "The log is stale" sends a reader to `git log` to
    work out which one, and that re-derivation is the thing this file exists to remove.
    """
    script = REPO_ROOT / "tools" / "queue_status.py"
    done = subprocess.run(
        [sys.executable, str(script), "--check"], capture_output=True, text=True, check=False
    )
    if "shallow clone" in done.stderr:
        pytest.skip("shallow clone: no history to check against, and the tool says so itself")
    assert done.returncode == 0, (
        f"{QUEUE.name} is behind right now — this test asserts the checker works, and it is "
        f"reporting a real omission:\n{done.stderr}"
    )
    assert "records every merged pull request" in done.stdout


def test_the_hook_runs_where_the_staleness_happens() -> None:
    """`post-merge`, not `pre-commit`. A `git pull` is what brings a merged PR down.

    Read out of the config rather than asserted in prose, because a hook wired to the wrong stage
    is exactly as green as one wired to the right one.
    """
    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    block = config.split("id: queue-is-current")[1].split("- id:")[0]
    assert "post-merge" in block, "the check does not run when a merge lands"
    assert "--check" in block and "--append" not in block, (
        "the hook must refuse rather than write: a generated entry reads as finished"
    )
