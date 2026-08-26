"""Snapshot every local-only file into a versioned store outside the repo.

**The problem this exists for.** A handful of files in this repo are gitignored and have **no second
copy anywhere**. Git cannot restore them because git has never seen them. `AGENTS.md` names the
mechanism that destroys them, and it is not carelessness: `git rm --cached` plus a `.gitignore`
entry leaves the working copy in place, but the *next* `checkout` or `pull` that crosses the
untracking commit sees a file that was tracked at the old HEAD and is not at the new one — and
deletes it. Nobody deleted anything. It has already happened twice here: once to all five notebook
builders, once to all four of exercises 01–04's briefs.

**Why a git repo rather than a copy.** A timestamped copy protects against deletion and nothing
else. These files are *regenerated* constantly — a notebook is rebuilt on every session — so the
likelier loss is a bad overwrite, and the second backup would faithfully overwrite the good copy
with the broken one. A git store keeps every version, so `git log` and `git show` reach back past a
mistake. It costs nothing: the whole set is about 12 MB of text.

**What it covers, and why that is wider than the tripwire.** The tripwire
(`tests/test_local_only_files_present.py`) watches notebooks, builders and briefs — the three
classes `AGENTS.md` names. But `docs/sessions/`
holds the entire course corpus (transcripts, assignments, and material for sessions this repo has
not reached yet), and `docs/EXPLAINER_*.md` are the two files any explainer is meant to be built
from. All of it is gitignored, none of it is regenerable, and none of it was in the tripwire. A
backup that only covered the documented cases would have missed the largest exposure.

**What it deliberately does NOT cover.** Anything regenerable (`artifacts/`, `data/`, `public/`,
`.venv/`), and anything secret. `.env` files, keys and credentials are excluded by pattern and the
run refuses outright if one matches — copying a secret into a second git repo is how a secret
outlives the decision to delete it.

Run it::

    uv run python tools/backup_local_only.py                 # snapshot and commit
    uv run python tools/backup_local_only.py --dry-run       # show what would be copied
    uv run python tools/backup_local_only.py --verify        # compare the store against the repo
    uv run python tools/backup_local_only.py --dest <path>   # somewhere other than the default

The default destination is **outside this repository on purpose** — a backup inside the working
tree dies with the working tree.
"""

import argparse
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger("backup_local_only")

#: Where snapshots go. A sibling of the repo, not a child: a backup inside the working tree is lost
#: with the working tree, which is one of the two failure modes this guards against.
DEFAULT_DEST = REPO_ROOT.parent / f".{REPO_ROOT.name}-local-only"

#: What to snapshot, as globs relative to the repo root.
#:
#: Each entry is here because losing it is **permanent**. Nothing regenerable belongs in this list:
#: a backup of derived output teaches the reader that the backup is optional.
PATTERNS: tuple[str, ...] = (
    "notebooks/S[0-9][0-9]-*.ipynb",
    "src/exercises/*/tools/build_notebook.py",
    "src/exercises/*/BRIEF.md",
    "docs/BRIEF.md",
    "docs/SESSIONS.md",
    "docs/EXPLAINER_*.md",
    "docs/sessions/**/*.md",
    "docs/sessions/**/*.svg",
    "TODO.md",
    # Saved reference pages. Not derived output and not reliably re-downloadable — a saved page is
    # a snapshot of something that can change or disappear, which is exactly why someone saved it.
    "src/exercises/*/docs/*.html",
)

#: Never copied, and a match aborts the run rather than skipping quietly.
#:
#: A secret that lands in a backup repo outlives every decision to delete it, and the backup is the
#: place nobody thinks to look. Refusing loudly is the only safe behaviour: silently skipping would
#: leave the operator believing the snapshot is complete.
FORBIDDEN: tuple[str, ...] = (
    "*.env",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "*credentials*",
    "*secret*",
    "*.p12",
    "*.pfx",
)


def _digest(path: Path) -> str:
    """Content hash of one file.

    Args:
        path: The file.

    Returns:
        Hex sha256.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def looks_like_a_credential(path: Path) -> bool:
    """Whether a path matches anything in `FORBIDDEN`.

    A named predicate rather than an inline comprehension, so it can be tested **without creating a
    file called `.env` on disk**. That is not squeamishness: writing real credential-shaped files
    in a test is how a test fixture ends up in a backup, and the sandbox this repo runs under
    refuses to `stat` them anyway — so the on-disk version of the test cannot run at all.

    Args:
        path: A repo-relative path.

    Returns:
        True if it must never be copied.
    """
    return any(path.match(rule) or path.name == rule for rule in FORBIDDEN)


def collect(root: Path) -> list[Path]:
    """Every local-only file present in this checkout, relative to the repo root.

    Args:
        root: The repo root.

    Returns:
        Sorted relative paths.

    Raises:
        SystemExit: If a pattern matched something that looks like a credential.
    """
    found: set[Path] = set()
    for pattern in PATTERNS:
        for path in root.glob(pattern):
            if path.is_file():
                found.add(path.relative_to(root))

    dangerous = sorted(str(p) for p in found if looks_like_a_credential(p))
    if dangerous:
        raise SystemExit(
            f"refusing to back up what looks like a credential: {dangerous}\n"
            f"A secret copied into a backup repo outlives every decision to delete it. Narrow the "
            f"patterns in PATTERNS rather than widening FORBIDDEN."
        )
    return sorted(found)


def _git(dest: Path, *args: str, must_succeed: bool = True) -> subprocess.CompletedProcess:
    """Run git inside the backup store, and refuse to ignore a failure.

    **`check=False` was the original defect and it was invisible.** `git commit` exits non-zero
    when the machine has no `user.email` — a bare CI runner, a fresh container, anyone who has not
    configured git globally. With the failure swallowed, the tool copied every file, printed
    success, and left a directory with **no commits at all**: the version history that is the whole
    reason for using git rather than `cp -r` simply did not exist, and the only symptom was an
    empty `git log` nobody ran. Caught by CI, which is exactly the machine that has no identity.

    Args:
        dest: The store.
        *args: Arguments after `git`.
        must_succeed: Raise if git fails. `False` for probes whose failure is meaningful.

    Returns:
        The finished process.

    Raises:
        SystemExit: If git failed and `must_succeed` is set.
    """
    finished = subprocess.run(
        ["git", "-C", str(dest), *args], capture_output=True, text=True, check=False
    )
    if must_succeed and finished.returncode != 0:
        raise SystemExit(
            f"git {' '.join(args)} failed in the backup store ({finished.returncode}):\n"
            f"{finished.stderr.strip() or finished.stdout.strip()}\n"
            f"The snapshot is NOT safe. Fix this before relying on the store."
        )
    return finished


def verify(root: Path, dest: Path, files: list[Path]) -> tuple[list[str], list[str]]:
    """Compare the store against the working tree.

    Args:
        root: The repo root.
        dest: The store.
        files: What should be in the store.

    Returns:
        `(absent from the store, differing in content)`.
    """
    absent, differing = [], []
    for relative in files:
        backed = dest / relative
        if not backed.is_file():
            absent.append(str(relative))
        elif _digest(backed) != _digest(root / relative):
            differing.append(str(relative))
    return absent, differing


def snapshot(root: Path, dest: Path, files: list[Path], *, message: str) -> int:
    """Copy every file into the store and commit whatever changed.

    Args:
        root: The repo root.
        dest: The store.
        files: What to copy.
        message: Commit subject.

    Returns:
        How many files differed from what the store already held.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").is_dir():
        _git(dest, "init", "--quiet", "--initial-branch=main")
        # A store-local identity, so a snapshot works on a machine with no global git config.
        # Without it `git commit` fails, and the failure used to be swallowed.
        _git(dest, "config", "user.email", "backup@local-only.invalid")
        _git(dest, "config", "user.name", "backup_local_only.py")
        (dest / "README.md").write_text(
            "# local-only backup\n\n"
            "Versioned snapshots of the gitignored, non-regenerable files in "
            f"`{root.name}` — notebooks, notebook builders, briefs and the course corpus under "
            "`docs/sessions/`. Git cannot restore those from the repo itself, because git has "
            "never seen them.\n\n"
            "Written by `tools/backup_local_only.py`. Restore a file with:\n\n"
            "```bash\n"
            "cp <this-repo>/<path> <working-repo>/<path>\n"
            "git -C <this-repo> log --oneline -- <path>   # every earlier version\n"
            "git -C <this-repo> show <sha>:<path>          # one of them\n"
            "```\n",
            encoding="utf-8",
        )

    changed = 0
    for relative in files:
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (root / relative).read_bytes()
        if not target.is_file() or target.read_bytes() != payload:
            changed += 1
        target.write_bytes(payload)

    _git(dest, "add", "-A")
    if _git(dest, "status", "--porcelain").stdout.strip():
        _git(dest, "commit", "--quiet", "-m", message)

    # Assert the commit landed, rather than assuming it. A store with files and no history is a
    # `cp -r` wearing a git directory, and it loses exactly the property the store exists for.
    if not _git(dest, "rev-parse", "--verify", "HEAD", must_succeed=False).stdout.strip():
        raise SystemExit(
            f"the store at {dest} holds files but no commits — every earlier version of every "
            f"file is unreachable. Something is wrong with git in that directory."
        )
    return changed


def main() -> int:
    """Snapshot, verify, or report.

    Returns:
        Process exit status. Non-zero when `--verify` finds the store out of date.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="which checkout to back up (defaults to the one this tool lives in)",
    )
    parser.add_argument("--dry-run", action="store_true", help="list what would be copied")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="compare the store against this checkout and exit non-zero if it is stale",
    )
    parser.add_argument("--message", default="snapshot", help="commit subject")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    files = collect(args.root)
    if not files:
        logger.warning(
            "no local-only files found under %s. On a fresh clone that is correct and expected; "
            "on a working checkout it means they are gone.",
            args.root,
        )
        # Not an error: a clone legitimately has nothing to back up, and failing here would make
        # the tool unusable as a gate. `--verify` still fails below when there IS something to
        # protect and the store does not have it.
        return 0

    total = sum((args.root / f).stat().st_size for f in files)
    logger.info("%d local-only files, %s KB", len(files), f"{total // 1024:,}")

    if args.dry_run:
        for relative in files:
            logger.info("  %s", relative)
        logger.info("would snapshot to %s", args.dest)
        return 0

    if args.verify:
        if not args.dest.is_dir():
            logger.error("no backup store at %s — run without --verify to create one", args.dest)
            return 1
        absent, differing = verify(args.root, args.dest, files)
        for name in absent:
            logger.error("NOT BACKED UP  %s", name)
        for name in differing:
            logger.warning("out of date    %s", name)
        if absent or differing:
            logger.error(
                "%d never backed up, %d out of date -> run tools/backup_local_only.py",
                len(absent),
                len(differing),
            )
            return 1
        logger.info("store at %s is current for all %d files", args.dest, len(files))
        return 0

    changed = snapshot(args.root, args.dest, files, message=args.message)
    head = _git(args.dest, "log", "-1", "--format=%h %s").stdout.strip()
    logger.info("%d changed -> %s%s", changed, args.dest, f"  [{head}]" if head else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
