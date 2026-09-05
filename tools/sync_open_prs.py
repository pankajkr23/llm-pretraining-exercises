"""Bring every open pull request up to date with `main`, and say what could not be done.

**Why this exists: merging one pull request in this repo breaks the next one, measurably.**
Every open branch touches three files that no two branches can change independently —

* `docs/agents/QUEUE.md`, because the convention is that a pull request logs itself when it opens,
* `CHANGELOG.md`, for the same reason one directory up,
* `.quote-check-receipt.json`, which is a digest over **all** tracked prose, so any merge that
  changes any prose anywhere invalidates every other branch's copy of it.

Measured before this was written: merging one branch into another produced a content conflict in
`QUEUE.md` **and** left the receipt containing conflict markers, so
`tools/quote_check_receipt.py --verify` died with a `JSONDecodeError` rather than a clean failure.
That is not a merge order anyone can follow by hand.

**What it does, per branch:** merge `origin/main` in, resolve those three files deterministically,
regenerate the receipt, commit, push.

**What it never does.** It does not rebase and it does not force-push — the branches are published,
`AGENTS.md` forbids rewriting published history, and the repository's own settings deny
`git push --force`. A merge commit is uglier in the log and is the only honest option. It never
touches `main`, and it never merges a pull request.

**How the two log files are resolved, and why not `merge=union`.** A union driver keeps both sides
of a conflict, which is right for two branches adding *different* lines and wrong here: fifteen
branches carry a byte-identical `#103` line, so union would land fifteen copies of it. Instead each
branch's OWN additions are computed against the merge base and re-applied to `main`'s version. A
line that main already has is therefore not added twice.

    uv run python tools/sync_open_prs.py --dry-run     # what it would do
    uv run python tools/sync_open_prs.py               # do it
    uv run python tools/sync_open_prs.py --only 121    # one pull request
"""

import argparse
import difflib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The three files every branch touches. Resolved by rule rather than by hand.
LOG_FILES = ("docs/agents/QUEUE.md", "CHANGELOG.md")
RECEIPT = ".quote-check-receipt.json"


def _run(*args: str, check: bool = True, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=check)


def _show(ref: str, path: str) -> str | None:
    """The content of `path` at `ref`, or None when the file does not exist there."""
    done = _run("git", "show", f"{ref}:{path}", check=False)
    return done.stdout if done.returncode == 0 else None


@dataclass
class Result:
    """What happened to one branch."""

    branch: str
    number: str
    state: str = "untouched"
    notes: list[str] = field(default_factory=list)


def open_pull_requests() -> list[tuple[str, str]]:
    """`(number, branch)` for every open pull request, newest first.

    Read from `gh` rather than from a list in this file: a roster written down here is a second
    copy of the truth, and the second copy is the one that goes stale.
    """
    done = _run(
        "gh",
        "pr",
        "list",
        "--state",
        "open",
        "--limit",
        "100",
        "--json",
        "number,headRefName",
        "--jq",
        '.[] | "\\(.number)\\t\\(.headRefName)"',
    )
    out = []
    for line in done.stdout.splitlines():
        if "\t" in line:
            number, branch = line.split("\t", 1)
            out.append((number.strip(), branch.strip()))
    return out


def _own_additions(base: str, branch: str, path: str) -> list[tuple[str, list[str]]]:
    """The blocks this branch ADDED to `path`, each with the line it was inserted before.

    Returned as `(anchor, lines)` pairs, where `anchor` is the first following line that already
    existed. That is what lets the block be re-applied to a `main` whose surrounding text has
    moved — an offset would not survive, a neighbouring line does.
    """
    before = (_show(base, path) or "").splitlines(keepends=True)
    after = (_show(branch, path) or "").splitlines(keepends=True)
    blocks: list[tuple[str, list[str]]] = []
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag not in ("insert", "replace"):
            continue
        added = after[j1:j2]
        # Both neighbours, not just the one after. A single line is not a position: the first
        # block this tool placed was anchored on "```", which occurs a dozen times in QUEUE.md, so
        # `list.index` found the earliest one and dropped a log entry into the prose section forty
        # lines above the log. It was still in the file, so every count of it looked right; the
        # checker that reads only `## Log` was the thing that noticed.
        preceding = after[j1 - 1] if j1 > 0 else ""
        following = after[j2] if j2 < len(after) else ""
        blocks.append(((preceding, following, j1), added))
    return blocks


#: A line that begins a new record in either log: a dated queue entry, a changelog bullet, or a
#: MARKDOWN HEADING. The heading is not decoration here — without it a `### Fixed` line attaches to
#: the record above it, and when that record is skipped as one main already has, the heading is
#: dropped with it. That is how `main` ended up with a bullet sitting directly under
#: `## [Unreleased]` with no section heading at all.
_RECORD_START = re.compile(r"^(?:\d{4}-\d{2}-\d{2}\s|- \*\*|#{2,4}\s)")


#: A markdown section heading. It is not a record of its own — it is CONTEXT for the record that
#: follows it, and that distinction is the fix. Treated as independent, `### Fixed` matches the
#: `### Fixed` main already has, gets skipped as a duplicate, and the entry beneath it lands with
#: no section heading at all. That is exactly how `main` ended up with a bullet sitting directly
#: under `## [Unreleased]`.
_HEADING = re.compile(r"^#{2,4}\s")


def _records(added: list[str]) -> list[list[str]]:
    """Split a block of added lines into the individual entries it contains.

    **Skipping has to be per entry, and this is the bug that proved it.** The first live run of this
    tool re-applied a block that held two entries — the branch's own, and a shared `#103` line that
    had reached `main` by another route. The whole-block test asked "is all of this already there?",
    the answer was no because half of it was new, and the half that was already there landed twice.
    A duplicated log entry is precisely what `merge=union` was refused for.
    """
    out: list[list[str]] = []
    for line in added:
        # A heading opens a record and keeps it open: the entry beneath it belongs to the same
        # record, so the pair is tested against main together and cannot be split apart.
        opens_here = _RECORD_START.match(line) and not _only_a_heading(out[-1] if out else None)
        if opens_here or not out:
            out.append([line])
        else:
            out[-1].append(line)
    return out


def _only_a_heading(record: list[str] | None) -> bool:
    """Is this record so far nothing but a heading and blank lines?"""
    if not record:
        return False
    return all(_HEADING.match(line) or not line.strip() for line in record)


def _placement_floor(lines: list[str]) -> int:
    """The earliest index a re-applied block may occupy.

    **Nearest-to-the-hint was not enough, and this is the fix for the failure it left.** A block's
    hint is where it sat on the branch, and for a changelog entry that is a small number — around
    line 12, just under `[Unreleased]`. So the *nearest* blank line on main is often line 1 or 2,
    and the block lands above the preamble, outside every section: present, correct-looking, and
    somewhere no reader will find it. It happened **five times** across one queue of pull requests,
    every one caught by a person checking placement rather than by anything failing.

    The floor is structural rather than per-file: a document's first `## ` heading is the point
    after which content belongs to a section at all. It is deliberately not the first `### `
    subsection — a block often carries its own subsection heading and belongs above the existing
    ones, and clamping to `### ` broke exactly that case in
    `test_a_heading_is_its_own_record_and_survives_a_skipped_neighbour`. The floor answers "is this
    inside the document's body", which is the question the five real failures got wrong; it does not
    try to choose between one subsection and another.

    Returns 0 when the file has no `## ` heading at all, which leaves behaviour unchanged for
    anything not shaped like these two logs.
    """
    first_section = next((i for i, line in enumerate(lines) if line.startswith("## ")), None)
    return 0 if first_section is None else first_section + 1


def _locate(
    lines: list[str], preceding: str, following: str, hint: int = 0
) -> tuple[int | None, str]:
    """Where a block belongs: the index whose neighbours are the ones it had on the branch.

    The pair is tried first because either line alone can occur many times — a code fence, a blank
    line, a heading. Only if the pair is absent does it fall back to the following line alone, and
    then to the preceding one, and each fallback is reported by the caller rather than assumed.
    """
    for i in range(len(lines)):
        if lines[i] == following and (i == 0 or lines[i - 1] == preceding):
            return i, "pair"
    # **Fall back to the NEAREST look-alike, not the first.** `list.index` returns the earliest
    # occurrence, and in an append-only log the earliest "```" or blank line is at the very top —
    # so a block whose neighbours had moved landed above the file's own preamble, outside every
    # section. That happened three times, on `CHANGELOG.md`, and each time the entry was present
    # and correct-looking while sitting somewhere no reader would find it.
    if following:
        hits = [i for i, line in enumerate(lines) if line == following]
        if hits:
            return min(hits, key=lambda i: abs(i - hint)), "the following line only"
    if preceding:
        hits = [i for i, line in enumerate(lines) if line == preceding]
        if hits:
            return min(hits, key=lambda i: abs(i - hint)) + 1, "the preceding line only"
    return None, "nothing"


def _reapply(
    main_text: str, blocks: list[tuple[tuple[str, str], list[str]]]
) -> tuple[str, list[str]]:
    """Put this branch's own blocks back into main's version. Returns the text and any notes."""
    lines = main_text.splitlines(keepends=True)
    notes: list[str] = []
    for anchor, block in blocks:
        # Drop the entries main already has, keep the rest, and preserve their order.
        keep: list[str] = []
        dropped = 0
        for record in _records(block):
            joined = "".join(record)
            if joined.strip() and joined in "".join(lines):
                dropped += 1
                continue
            keep.extend(record)
        if dropped:
            notes.append(f"skipped {dropped} entr(y/ies) main already has")
        added = keep
        if not added:
            continue
        preceding, following, where = anchor
        at, how = _locate(lines, preceding, following, hint=where)
        floor = _placement_floor(lines)
        if at is not None and at < floor:
            # Above the first section is not a placement, it is a loss that looks like one.
            notes.append(
                f"a {len(added)}-line block resolved to line {at + 1}, above the first section; "
                f"placed at line {floor + 1} instead"
            )
            at, how = floor, "the section floor"
        if how != "pair" and at is not None:
            # Worth saying. A degraded placement is still a placement, but main has moved under
            # this block and somebody should look at where it landed.
            notes.append(
                f"placed a {len(added)}-line block by {how}; its neighbours have changed on main"
            )
        if at is None:
            notes.append(f"could not place a {len(added)}-line block; appended to the end instead")
            lines.extend(added)
            continue
        lines[at:at] = added
    return "".join(lines), notes


def sync(number: str, branch: str, dry_run: bool) -> Result:
    """Merge main into one branch and make its three shared files correct again."""
    result = Result(branch=branch, number=number)

    _run("git", "fetch", "origin", branch, check=False)
    behind = _run(
        "git", "rev-list", "--count", f"{branch}..origin/main", check=False
    ).stdout.strip()
    if behind in ("", "0"):
        result.state = "already current"
        return result
    result.notes.append(f"{behind} commit(s) behind main")
    if dry_run:
        result.state = "would sync"
        return result

    # **Judge the checkout by where HEAD ended up, not by the exit code.** A `post-checkout` hook
    # cannot abort a checkout -- git has already rewritten the working tree by the time it runs --
    # but its failure still sets the exit status, and this refused on that basis. The local-only
    # tripwire runs there, and it goes red on any branch that predates a file becoming tracked:
    # correct-looking, since the class is then partly present, and wrong, since `origin/main` can
    # give the file back. That refused seventeen open pull requests on a healthy checkout.
    #
    # The hook's complaint is still surfaced, because the one thing worse than refusing here is
    # swallowing a real loss. The merge that follows restores anything main holds, and the
    # pre-commit hook re-runs the same tripwire before the merge is allowed to land.
    checkout = _run("git", "checkout", "-q", branch, check=False)
    landed = _run("git", "rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip()
    if landed != branch:
        result.state = "REFUSED"
        result.notes.append(checkout.stderr.strip()[:160] or f"HEAD is {landed!r}, not {branch!r}")
        return result
    if checkout.returncode != 0:
        result.notes.append("a post-checkout hook failed; the checkout itself succeeded")

    base = _run("git", "merge-base", branch, "origin/main").stdout.strip()
    own = {path: _own_additions(base, branch, path) for path in LOG_FILES}

    merged = _run("git", "merge", "--no-edit", "origin/main", check=False)
    conflicted = _run("git", "diff", "--name-only", "--diff-filter=U", check=False).stdout.split()

    # **Every log file is rebuilt, not only the conflicted ones.** A clean auto-merge is not a
    # correct one here: when the branch's entry and a shared entry sit at different offsets, git
    # sees two independent insertions, merges both without complaint, and the shared record lands
    # TWICE. The first duplicate this tool produced came from resolving a conflict too coarsely;
    # the second came from there being no conflict to resolve at all. Rebuilding unconditionally
    # is the only version that covers both.
    for path in LOG_FILES:
        main_text = _show("origin/main", path)
        if main_text is None:
            result.notes.append(f"{path}: not on main, left for a human")
            continue
        text, notes = _reapply(main_text, own[path])
        if text == (REPO_ROOT / path).read_text():
            continue
        (REPO_ROOT / path).write_text(text)
        _run("git", "add", path)
        result.notes.extend(f"{path}: {n}" for n in notes)
        how = "resolved" if path in conflicted else "rebuilt: git merged it cleanly and wrongly"
        result.notes.append(f"{path}: {how} as main + this branch's own {len(own[path])} block(s)")

    if RECEIPT in conflicted:
        # Never merged, always recomputed: the receipt is a digest of the tree it sits in, so a
        # merged one would be a digest of nothing.
        (REPO_ROOT / RECEIPT).write_text(_show("origin/main", RECEIPT) or "{}")
        _run("git", "add", RECEIPT)

    still = _run("git", "diff", "--name-only", "--diff-filter=U", check=False).stdout.split()
    if still:
        result.state = "CONFLICT — needs a person"
        result.notes.append("unresolved: " + ", ".join(still))
        _run("git", "merge", "--abort", check=False)
        return result

    # Regenerate rather than trust: the tree has just changed, so whatever receipt is on disk is
    # describing a tree that no longer exists.
    receipt = _run("uv", "run", "python", "tools/quote_check_receipt.py", "--write", check=False)
    if receipt.returncode != 0:
        result.state = "receipt could not be written"
        result.notes.append((receipt.stdout + receipt.stderr).strip()[-200:])
        return result
    _run("git", "add", RECEIPT)

    if merged.returncode != 0 or conflicted:
        commit = _run("git", "commit", "--no-edit", check=False)
        if commit.returncode != 0:
            result.state = "commit refused"
            result.notes.append((commit.stdout + commit.stderr).strip()[-200:])
            return result

    verify = _run("uv", "run", "python", "tools/quote_check_receipt.py", "--verify", check=False)
    if verify.returncode != 0:
        result.state = "receipt does not vouch for the merged tree"
        return result

    push = _run("git", "push", "origin", branch, check=False)
    if push.returncode != 0:
        result.state = "push refused"
        result.notes.append((push.stdout + push.stderr).strip()[-200:])
        return result

    result.state = "synced"
    return result


def main(argv: list[str] | None = None) -> int:
    """Sync every open pull request, or one, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="say what would happen, change nothing"
    )
    parser.add_argument("--only", metavar="NUMBER", help="one pull request number")
    args = parser.parse_args(argv)

    started_on = _run("git", "branch", "--show-current").stdout.strip()
    dirty = _run("git", "status", "--porcelain").stdout.strip()
    if dirty and not args.dry_run:
        print("the working tree has uncommitted changes; commit or stash them first:")
        print(dirty)
        return 2

    _run("git", "fetch", "origin", "main", check=False)
    pulls = open_pull_requests()
    if args.only:
        pulls = [p for p in pulls if p[0] == args.only]
        if not pulls:
            print(f"no open pull request numbered {args.only}")
            return 2

    results = [sync(number, branch, args.dry_run) for number, branch in pulls]

    if started_on and not args.dry_run:
        _run("git", "checkout", "-q", started_on, check=False)

    width = max((len(r.branch) for r in results), default=10)
    print()
    for r in sorted(results, key=lambda r: int(r.number)):
        print(f"#{r.number:<5} {r.branch:<{width}}  {r.state}")
        for note in r.notes:
            print(f"        {note}")

    bad = [r for r in results if r.state not in ("synced", "already current", "would sync")]
    print(f"\n{len(results) - len(bad)} fine, {len(bad)} needing attention")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
