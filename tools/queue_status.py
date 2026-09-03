"""Refuse to let the progress log fall behind what actually merged.

**The failure this exists for happened, and it is why the tool is mechanical.** The queue at
`docs/agents/QUEUE.md` is the single source of truth for progress — tracked precisely so state
survives a crash, a context reset and a fresh clone. It carried one line reading *"no unit has run
yet"* while **nine** pull requests merged past it. Nobody noticed, because nothing was watching, and
re-deriving the state from `git log` afterwards is the work the file exists to make unnecessary.

**Why a check and not an agent on a timer.** The obvious fix is to have something write the log
after every merge. That is worse in two ways, and both matter more than the convenience:

- *It writes prose nobody asked for.* An entry generated on a schedule says what changed, which the
  changelog already says better. The entries worth having are the ones carrying judgement — what
  went red first, what a guard caught, which decision was reversed — and those are known only to
  whoever did the work, at the moment they did it.
- *It hides the omission instead of surfacing it.* A generator that always succeeds means the log is
  never wrong and never informative. A check that fails means the person with the context is asked
  for the one sentence only they can write.

So this refuses, and `--append` offers a stub rather than a story: the merged subject, and a `TODO`
the author replaces. A stub left unfilled is visible in review; a fabricated entry is not.

    uv run python tools/queue_status.py --check     # non-zero when the log is behind
    uv run python tools/queue_status.py --append    # add a stub for each missing merge

Wired to pre-commit's **post-merge** stage, which is when `git pull` brings a merged pull request
down — the moment the log goes stale, rather than whenever somebody remembers.

**What it cannot see.** A shallow clone has no history to read, so the check reports that and stops
rather than passing. CI clones shallow for most jobs, which makes this a local gate; saying so here
is the point, because a guard that quietly does nothing reads as coverage.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUEUE = REPO_ROOT / "docs" / "agents" / "QUEUE.md"

#: A squash merge's subject ends with the pull request it came from: `… (#94)`.
_MERGED = re.compile(r"\(#(\d+)\)\s*$")

#: Where the log starts. Entries above it are prose about the log, not entries.
_LOG_HEADING = "## Log"


def _git(*args: str) -> str:
    """Run git in the repo and return stdout, empty on failure."""
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
    ).stdout


def is_shallow() -> bool:
    """True when the clone has no real history to read — CI, usually."""
    return _git("rev-parse", "--is-shallow-repository").strip() == "true"


def merged_pull_requests(limit: int = 200) -> list[tuple[str, str]]:
    """`(number, subject)` for each squash-merged pull request, newest first."""
    out = []
    for line in _git("log", f"-{limit}", "--first-parent", "--format=%s", "main").splitlines():
        found = _MERGED.search(line)
        if found:
            out.append((found.group(1), _MERGED.sub("", line).strip()))
    return out


def logged_numbers(text: str) -> set[str]:
    """Every pull request number the log mentions.

    Read from the `## Log` section only. A number in the prose above it is discussion, and counting
    it would let the check pass on a file that describes merges without recording them.
    """
    _, _, log = text.partition(_LOG_HEADING)
    return set(re.findall(r"#(\d+)", log))


def missing(text: str) -> list[tuple[str, str]]:
    """Merged pull requests the log does not mention, oldest first so stubs append in order."""
    known = logged_numbers(text)
    return [entry for entry in reversed(merged_pull_requests()) if entry[0] not in known]


def stub(number: str, subject: str) -> str:
    """One log line for a merge nobody has written up yet.

    The subject is copied rather than summarised, and the second half is a `TODO` on purpose: the
    part worth reading is what went wrong or what a guard caught, and this tool does not know that.
    """
    todo = "TODO: what this changed, and what it cost"
    return f"{'':12}#{number:<6} merged: {subject}\n{'':20}{todo}\n"


def append_stubs(text: str, entries: list[tuple[str, str]]) -> str:
    """Insert stubs at the end of the fenced block inside the log section."""
    head, sep, log = text.partition(_LOG_HEADING)
    close = log.rstrip().rfind("```")
    if close == -1:
        raise SystemExit(f"{QUEUE.name} has no fenced log block to append to")
    body = "".join(stub(number, subject) for number, subject in entries)
    return head + sep + log[:close] + body + log[close:]


def main() -> int:
    """Check or append. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="non-zero when the log is behind")
    group.add_argument("--append", action="store_true", help="add a stub per unlogged merge")
    args = parser.parse_args()

    if is_shallow():
        print(
            f"{QUEUE.name}: not checked — this is a shallow clone with no history to read, so "
            "this gate runs where the history does. That is a local check, not a CI one.",
            file=sys.stderr,
        )
        return 0

    text = QUEUE.read_text(encoding="utf-8")
    behind = missing(text)
    if not behind:
        print(f"{QUEUE.name} records every merged pull request")
        return 0

    if args.append:
        QUEUE.write_text(append_stubs(text, behind), encoding="utf-8")
        print(f"added {len(behind)} stub(s) to {QUEUE.name} — replace each TODO before committing")
        return 0

    print(
        f"{QUEUE.name} is behind: it does not record "
        + ", ".join(f"#{number}" for number, _ in behind)
        + ".\n\nIt is the single source of truth for progress, and a reader who cannot trust it "
        "re-derives\nthe state from git — which is the work it exists to remove. Add the entry, or "
        "run:\n\n    uv run python tools/queue_status.py --append\n\nthen replace each TODO with "
        "what the change cost. A stub is visible in review; a guess is not.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
