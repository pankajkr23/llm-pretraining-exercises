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

**Write the entry when the pull request is OPENED, not after it merges.** This is a convention the
check forces rather than one it merely prefers, and the reason is a regress that showed up on the
very first pull after shipping it: a pull request cannot record its own merge, so `--check` failed
on `git pull` after *every* merge, and the only fix — another pull request — needed recording in
turn, for ever. Recording at open time closes it: by the time the merge lands, the log already says
so. `gh pr create` prints the number; add the line, amend, push.

**It runs in CI as well as locally, and the reason is a measurement rather than a preference.** The
first version skipped in CI because the runner clones shallow, and that was written up as "a local
gate" — which this repository's own rule says is barely a gate at all. The number that settles it:
the `security` job already clones **full history** and scans every commit with gitleaks in **7
seconds**, over 519 commits and a 33 MB `.git`. Full history costs nothing here, so the `test` job
fetches it too and this check is enforced rather than declared.

**It fails closed in both blind cases**, and the second was a real bug. A shallow clone now exits
non-zero instead of quietly returning success. And the base ref was hardcoded to `main`, which does
not exist on a CI pull-request checkout: `git log` on an unknown revision writes to stderr and
leaves stdout empty, so the checker found no merges, concluded the log was complete, and passed.
A checker that reports success exactly when it can see nothing is the shape this tool exists to
stop, and it had it.
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


#: Where to look for merges, in order, and **the order is the whole point**.
#:
#: `origin/main` comes first because it is the authority: a local `main` is whatever was last pulled
#: and goes stale the moment it is not. That was the third bug of the same family in this file — the
#: checker being confidently wrong about *what it was looking at* — and it presented identically to
#: the other two, as a clean pass. It read a local `main` that predated a merge and reported the log
#: complete, on a branch where the log was demonstrably behind.
#:
#: `main` is the fallback for a clone with no remote. `HEAD` is last, for a CI pull-request checkout
#: where neither branch exists locally but the merge ref already contains main's history.
_BASE_REFS = ("origin/main", "main", "HEAD")


def base_ref() -> str | None:
    """The first ref that resolves, or None.

    **The ref was hardcoded to `main`, and that is a silent-pass bug rather than a portability
    one.** `git log` on an unknown revision writes to stderr and exits non-zero, leaving stdout
    empty — so the caller found no merges, concluded the log recorded all of them, and passed. A
    checker that reports "everything is fine" precisely when it cannot see anything is the failure
    shape this tool was written to stop, and it had it.
    """
    for ref in _BASE_REFS:
        if _git("rev-parse", "--verify", "--quiet", ref).strip():
            return ref
    return None


def merged_pull_requests(ref: str, limit: int = 200) -> list[tuple[str, str]]:
    """`(number, subject)` per squash-merged pull request reachable from `ref`, newest first."""
    out = []
    for line in _git("log", f"-{limit}", "--first-parent", "--format=%s", ref).splitlines():
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


#: A status that is only true while a pull request is open. Written into the log constantly — a unit
#: in flight is the normal thing to record — and wrong the moment the pull request lands.
_OPEN_MARKER = re.compile(r"#(\d+)[^\n]*\bPR OPEN\b|(?:\bPR OPEN\b)[^\n]*#(\d+)")


def missing(text: str, ref: str) -> list[tuple[str, str]]:
    """Merged pull requests the log does not mention, oldest first so stubs append in order."""
    known = logged_numbers(text)
    return [entry for entry in reversed(merged_pull_requests(ref)) if entry[0] not in known]


def stale_open(text: str, ref: str) -> list[str]:
    """Pull requests the log still calls open that have in fact merged.

    **The check began as mention-or-not, and that was too weak by exactly one case.** An entry
    reading `#96 PR OPEN` counts as a record of #96, so the log passed while describing a merged
    pull request as still in flight — which is the log being behind, just less obviously than an
    omission. Caught the first time the tool ran for real, on its own author's entry.

    Deliberately the only status this understands. Reading further into what an entry *means* would
    make the checker a parser of prose it does not own, and prose is where the judgement lives.

    **The known false positive, which fired immediately:** a log entry that *quotes* the marker
    while explaining something trips this, because a status and a quotation of a status are
    lexically identical and only knowledge separates them. That is the same shape as the
    hand-maintained dotfile list in `tests/test_standards_name_real_code.py`, and it is resolved the
    same way — reword the sentence, rather than teaching the checker to parse quotation. The cost is
    one awkward sentence; the alternative is a matcher that is wrong in ways nobody can predict.
    """
    merged = {number for number, _ in merged_pull_requests(ref)}
    found = {a or b for a, b in _OPEN_MARKER.findall(text.partition(_LOG_HEADING)[2])}
    return sorted(found & merged, key=int)


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

    # **Fails closed, both ways.** A shallow clone and an unresolvable ref both mean "cannot see the
    # history", and the first version returned 0 for one and passed silently on the other. Either
    # answer amounts to a checker reporting success exactly when it is blind, which is what it was
    # written to prevent. CI now fetches full history, so neither branch should fire there — and if
    # one does, that is a change to the workflow worth noticing rather than absorbing.
    if is_shallow():
        print(
            f"{QUEUE.name}: NOT CHECKED — shallow clone, no history to read. The workflow is "
            "supposed to fetch\nfull history for this job; if that changed, this check has stopped "
            "running and is not merely quiet.",
            file=sys.stderr,
        )
        return 1

    ref = base_ref()
    if ref is None:
        print(
            f"{QUEUE.name}: NOT CHECKED — none of {', '.join(_BASE_REFS)} resolves, so there is no "
            "history to compare\nagainst. Refusing rather than reporting success while blind.",
            file=sys.stderr,
        )
        return 1

    text = QUEUE.read_text(encoding="utf-8")
    behind = missing(text, ref)
    stale = stale_open(text, ref)

    if stale and not args.append:
        print(
            f"{QUEUE.name} still calls "
            + ", ".join(f"#{number}" for number in stale)
            + " open, and they have merged.\n\nAn entry that names a pull request counts as a "
            "record of it, so this passed while the log\ndescribed landed work as still in flight. "
            "Update the entry rather than adding a second one.",
            file=sys.stderr,
        )
        return 1

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
