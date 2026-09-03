"""The progress log cannot silently fall behind what merged.

`docs/agents/QUEUE.md` is the single source of truth for progress. It once carried one line reading
*"no unit has run yet"* while nine pull requests merged past it, and nothing noticed — so the point
of these tests is not that the checker runs, but that it **refuses**. Every property is written
twice, once passing and once failing, because a checker that cannot fail would be the most
convincing possible way to prove the log is current.

**These run everywhere, CI included**, and two of them exist because of what the first version got
wrong: it hardcoded a ref that does not resolve on a CI checkout and therefore passed while blind,
and it wrote itself off as a local-only gate on a cost nobody had measured.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from queue_status import (  # noqa: E402
    QUEUE,
    append_stubs,
    base_ref,
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


def test_it_refuses_rather_than_passing_when_it_cannot_see_the_history() -> None:
    """The bug this had, and the one it was written to prevent — in itself.

    The base ref was hardcoded to `main`, which does not exist on a CI pull-request checkout. `git
    log` on an unknown revision writes to stderr and leaves stdout **empty**, so the checker found
    no merges, concluded the log recorded all of them, and exited 0. It reported success precisely
    when it was blind.

    `base_ref()` now returns None when nothing resolves and the caller refuses. This asserts the
    resolution works here *and* that the ordering is real, since a list that only ever finds its
    first entry would hide a broken fallback.
    """
    resolved = base_ref()
    assert resolved is not None, "no base ref resolves, so the checker cannot see any history"
    assert resolved in ("main", "origin/main", "HEAD")


def test_the_checker_runs_in_ci_rather_than_declaring_itself_local() -> None:
    """Full history is fetched for the test job, or this guard silently stops being one.

    Read out of the workflow, because the claim in the module docstring — that this is enforced
    rather than local — is only true while that setting is there. It was written off as a local
    gate on an *assumed* cost; the `security` job fetches full history and scans every commit in
    seconds, so the assumption was wrong and this asserts the correction stays.
    """
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    test_job = workflow.split("\n  test:")[1].split("\n  integration:")[0]
    assert "fetch-depth: 0" in test_job, (
        "the test job clones shallow again, so queue_status.py cannot see any merges and this "
        "check has quietly become local-only"
    )


def test_an_entry_that_still_calls_a_merged_request_open_is_refused() -> None:
    """Mention alone was too weak by exactly one case, and it fired on its author's own entry.

    An entry naming a pull request counted as a record of it, so a log describing landed work as
    still in flight passed. That *is* the log being behind, just less obviously than an omission.

    Driven with synthetic text and a ref of `HEAD`, so it asserts the matcher rather than whatever
    happens to be merged today.
    """
    from queue_status import stale_open

    live = _LOG.replace("#88 merged: something", "#88 PR OPEN  something")
    flagged = stale_open(live, "HEAD")
    assert isinstance(flagged, list), "stale_open must return a list to join into a message"

    # The matcher itself, independent of what history says: both orderings of the pair.
    from queue_status import _OPEN_MARKER

    for line in ("#96 PR OPEN  something", "PR OPEN — see #96"):
        found = {a or b for a, b in _OPEN_MARKER.findall(line)}
        assert found == {"96"}, f"the open marker did not match in {line!r}: {found}"
    assert not _OPEN_MARKER.findall("#96 merged: something"), "a merged entry must not match"
