"""Snapshot every local-only file into a versioned store outside the repo.

**The problem this exists for.** A handful of files in this repo are gitignored and have **no second
copy anywhere**. Git cannot restore them because git has never seen them. `AGENTS.md` names the
mechanism that destroys them, and it is not carelessness: `git rm --cached` plus a `.gitignore`
entry leaves the working copy in place, but the *next* `checkout` or `pull` that crosses the
untracking commit sees a file that was tracked at the old HEAD and is not at the new one — and
deletes it. Nobody deleted anything. It has already happened twice here: once to all five notebook
builders, once to all four of exercises 01–04's requirement documents.

**Why a git repo rather than a copy.** A timestamped copy protects against deletion and nothing
else. These files are *regenerated* constantly — a notebook is rebuilt on every topic — so the
likelier loss is a bad overwrite, and the second backup would faithfully overwrite the good copy
with the broken one. A git store keeps every version, so `git log` and `git show` reach back past a
mistake. It costs nothing: the whole set is about 12 MB of text.

**What it covers, and why that is wider than the tripwire.** The tripwire
(`tests/test_local_only_files_present.py`) watches notebooks, builders and requirement documents —
the three
classes `AGENTS.md` names. The confidential reference material lives **outside the repository**
(see `EXTERNAL_SOURCES`) and is snapshotted here too, and `docs/EXPLAINER_*.md` are the two files
any explainer is meant to be built from. All of it is gitignored, none of it is
regenerable, and none of it was in the tripwire. A
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
import json
import logging
import os
import subprocess
import sys
import textwrap
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
    "src/exercises/*/REQUIREMENTS.md",
    "docs/REQUIREMENTS.md",
    "docs/EXPLAINER_*.md",
    "TODO.md",
    # Hand-written planning and critique notes that live beside an exercise. Only the untracked ones
    # are taken: `collect` drops anything git already has, so this cannot quietly start duplicating
    # tracked documents into the store.
    "src/exercises/*/docs/*.md",
    # What agents in this repo are permitted to run without asking. Never in git, ignored by a
    # directory pattern, and losing it silently changes the permission surface rather than failing.
    ".claude/settings.local.json",
    # Saved reference pages. Not derived output and not reliably re-downloadable — a saved page is
    # a snapshot of something that can change or disappear, which is exactly why someone saved it.
    "src/exercises/*/docs/*.html",
    # **The one class with no recovery path at all.** `src/solution/` holds the course's reference
    # implementation, and unlike everything else here it has NEVER been in git on any branch — so
    # the fallback in AGENTS.md (`git show <untracking-commit>^:<path>`) is inapplicable by
    # construction, because there is no removal commit to reach back past. `.gitignore` excludes it
    # with a **directory** pattern, so a negation could not rescue it either.
    #
    # It also cannot be re-fetched. `corpus/*.raw.html` is the Wikipedia HTML the tracked corpus was
    # derived from, and the script that fetched it pins no revision — re-running it today returns a
    # different article. The derived `.faithful.txt` IS tracked; its input is not, so losing this
    # destroys the provenance of the number the whole exercise is graded on.
    "src/exercises/*/src/solution/**/*",
    # The frozen copies of the standard files. Local-only by decision: they exist to be diffed
    # against on the machine doing the rewriting, and shipping a second copy of AGENTS.md and
    # DESIGN.md to the remote would put the same conventions in the repo twice — the argument that
    # untracked the notebooks.
    #
    # That decision is what makes this entry mandatory rather than tidy. The whole point of the
    # archive is to hold a version nothing else holds, so an archive that is neither tracked nor
    # backed up is the one class of file this repo has already lost twice. `--force-rewrite` is
    # cheap to re-run for a tag that still exists, but a snapshot of a tag that has been deleted,
    # or of a file since rewritten, is gone for good.
    "docs/standards-history/*",
)

#: Directories that live **outside the repository** and are still backed up here.
#:
#: The confidential reference material used to sit inside the working tree, gitignored. That worked
#: for the bytes and failed for everything else: a tracked document could still name its files or
#: quote them, and several did. Moving it out removes the whole class — there is no path inside the
#: repo to leak, nothing for `.gitignore` to name, and no way to commit it by accident.
#:
#: It still needs a backup, because "outside the repo" protects it from git and from nothing else.
#: The location is overridable so a second machine can put it elsewhere; the default is a sibling of
#: the repo, the same shape as the store itself.
EXTERNAL_SOURCES: dict[str, Path] = {
    "notes": Path(
        os.environ.get("LLM_NOTES_DIR", str(REPO_ROOT.parent / f".{REPO_ROOT.name}-notes"))
    ),
}


#: Never copied. A match is **skipped and reported**, and the run exits non-zero.
#:
#: A secret that lands in a backup repo outlives every decision to delete it, and the backup is the
#: place nobody thinks to look. But the first version *aborted the whole snapshot* on a match, which
#: is the wrong failure: one innocuously-named document would take the run from a hundred files
#: protected to zero, and the operator would be left with no backup and a scary message. Skipping
#: the file, naming it, and returning non-zero gives both the protection and the alarm.
#:
#: The patterns are deliberately narrow. `*secret*` and `*credentials*` were here and would match
#: ordinary prose filenames — a topic note about secrets, a document about credentialing — so
#: they are anchored to the shapes real credential files actually take.
FORBIDDEN: tuple[str, ...] = (
    "*.env",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "credentials",
    "credentials.*",
    "*.credentials",
    "*_secret.*",
    "*.secret",
    "secrets.*",
)

#: Paths whose contents are regenerable, and therefore deliberately absent from `PATTERNS`.
#:
#: Used only by `uncovered()`. Nothing here is backed up; the list exists so that "not backed up"
#: can be split into *decided* and *overlooked*, which is the difference between a selection that
#: fails closed and one that fails **silently**.
REGENERABLE: tuple[str, ...] = (
    ".venv",
    ".git",
    "node_modules",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    "artifacts",
    "public",
    "data",
    ".DS_Store",
    ".claude",
    "submission_artifacts",
    "results",
    "uv.lock",
    ".coverage",
    "htmlcov",
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


def _tracked(root: Path) -> set[Path]:
    """Every path git already has, so the store never duplicates one.

    Args:
        root: The repo root.

    Returns:
        Repo-relative paths. Empty when git is unavailable, which fails toward backing up more
        rather than less.
    """
    finished = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True, check=False
    )
    if finished.returncode != 0:
        return set()
    return {Path(name) for name in finished.stdout.split("\0") if name}


def collect(root: Path) -> tuple[list[Path], list[str]]:
    """Every local-only file present in this checkout, relative to the repo root.

    Args:
        root: The repo root.

    Returns:
        `(files to back up, credential-shaped paths that were skipped)`.

        A skipped credential is returned rather than raised. The first version raised, which meant
        one innocuously-named document took the run from a hundred files protected to **zero** — no
        backup at all, plus a frightening message. Skipping it, naming it and exiting non-zero gives
        the protection and the alarm together.
    """
    found: set[Path] = set()
    for pattern in PATTERNS:
        for path in root.glob(pattern):
            # `.DS_Store` is Finder metadata, not content. The directory sweeps below pick it up,
            # and storing it means the store and the checkout disagree the moment Finder touches
            # either — which reads as a loss and is not one.
            if path.is_file() and path.name != ".DS_Store":
                found.add(path.relative_to(root))

    skipped = sorted(str(p) for p in found if looks_like_a_credential(p))
    found = {p for p in found if not looks_like_a_credential(p)}

    # A tracked file has a second copy by definition, and copying it here would make the store's
    # contents ambiguous: a reader could not tell which paths git can restore and which it cannot.
    found -= _tracked(root)
    return sorted(found), skipped


def uncovered(root: Path) -> list[Path]:
    """Ignored files that are neither backed up nor declared regenerable.

    **The check that turns a fail-closed selection into a visible one.** `PATTERNS` is an
    allowlist, so anything new and irreplaceable is missed *silently* — which is exactly how the
    course's reference solution tree sat outside every guard while both of them reported success.
    Enumerating what git ignores and subtracting both the backed-up set and the regenerable set
    leaves precisely the files nobody has decided about.

    Scoped by pathspec rather than filtered afterwards: the repo has ~93,000 ignored files and all
    but a few hundred are inside `.venv` and `artifacts`, so asking git about those directories is
    slow and answers a question nobody has.

    Args:
        root: The repo root.

    Returns:
        Sorted repo-relative paths awaiting a decision.
    """
    finished = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            ".",
            *[f":(exclude){name}" for name in REGENERABLE],
            *[f":(exclude)*/{name}" for name in REGENERABLE],
            *[f":(exclude)**/{name}/**" for name in REGENERABLE],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if finished.returncode != 0:
        return []

    backed_up = set(collect(root)[0])
    out: list[Path] = []
    for name in finished.stdout.split("\0"):
        if not name:
            continue
        path = Path(name)
        if path in backed_up or looks_like_a_credential(path):
            continue
        if any(part in REGENERABLE for part in path.parts):
            continue
        out.append(path)
    return sorted(out)


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


def verify(root: Path, dest: Path, files: list[Path]) -> tuple[list[str], list[str], list[str]]:
    """Compare the store against the working tree, **in both directions**.

    **The one-directional version reported success at the exact moment a file was lost**, and it is
    the command `AGENTS.md` tells you to run first after a checkout, pull, merge or rebase — the
    operation class that has already destroyed these files twice. The reason is structural: it
    checked "is everything the checkout has also in the store?", and a deleted file is not in the
    checkout, so it was not asked about. Deleting all six notebooks made the answer *better*.

    So the store is enumerated too. A path the store holds and the working tree does not is a
    **loss**, not staleness, and it is the only finding here that means "stop and restore".

    Args:
        root: The repo root.
        dest: The store.
        files: What the checkout currently offers.

    Returns:
        `(absent from the store, differing in content, lost from the checkout)`.
    """

    def _source_of(relative: Path) -> Path | None:
        """Where a stored path came from — inside the repo, or an external directory."""
        head = relative.parts[0] if relative.parts else ""
        if head in EXTERNAL_SOURCES:
            source = EXTERNAL_SOURCES[head]
            # An external directory that is simply not mounted on this machine is not a loss, and
            # must not be reported as one. Absent source, absent verdict.
            return source.joinpath(*relative.parts[1:]) if source.is_dir() else None
        return root / relative

    absent, differing = [], []
    for relative in files:
        backed = dest / relative
        if not backed.is_file():
            absent.append(str(relative))
        elif _digest(backed) != _digest(root / relative):
            differing.append(str(relative))

    for name, source in EXTERNAL_SOURCES.items():
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file() or ".git" in path.parts or path.name == ".DS_Store":
                continue
            relative = Path(name) / path.relative_to(source)
            backed = dest / relative
            if not backed.is_file():
                absent.append(str(relative))
            elif _digest(backed) != _digest(path):
                differing.append(str(relative))

    lost = []
    if dest.is_dir():
        for backed in dest.rglob("*"):
            if not backed.is_file() or ".git" in backed.parts:
                continue
            relative = backed.relative_to(dest)
            if relative == Path("README.md"):
                continue  # the store's own note to a reader, with no counterpart in the repo
            origin = _source_of(relative)
            if origin is not None and not origin.is_file():
                lost.append(str(relative))
    return absent, differing, sorted(lost)


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
            f"`{root.name}` — notebooks, builders, requirement files — plus the reference material "
            "that lives outside the repo entirely. Git cannot restore any of it from the repo "
            "itself, because git has never seen it.\n\n"
            "Written by `tools/backup_local_only.py`. Restore a file with:\n\n"
            "```bash\n"
            "cp <this-repo>/<path> <working-repo>/<path>\n"
            "git -C <this-repo> log --oneline -- <path>   # every earlier version\n"
            "git -C <this-repo> show <sha>:<path>          # one of them\n"
            "```\n",
            encoding="utf-8",
        )

    # **The store must not inherit the user's global gitignore, and this is not hypothetical.**
    # The store is a git repository, so `git add` honours `~/.config/git/ignore` and
    # `core.excludesFile` exactly as it would anywhere else. A developer whose global ignore holds
    # `**/.claude/settings.local.json` — a completely reasonable line to have — silently gets a
    # store where that file is **copied but never committed**: on disk, `--verify` satisfied because
    # the bytes match, and no history at all. That is the one property the store exists to provide,
    # and it is the `cp -r` wearing a git directory that the assertion at the end of this function
    # now refuses. It went unnoticed for months.
    #
    # Set on every run rather than only at init, because the stores that need it already exist.
    _git(dest, "config", "core.excludesFile", os.devnull)

    # Everything the repo holds, plus every external directory. `pairs` is (store path, source
    # file); the store does not care that some of these came from outside the working tree.
    pairs: list[tuple[Path, Path]] = [(rel, root / rel) for rel in files]
    for name, source in EXTERNAL_SOURCES.items():
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file() or ".git" in path.parts or path.name == ".DS_Store":
                continue
            pairs.append((Path(name) / path.relative_to(source), path))

    changed = 0
    for relative, origin in pairs:
        target = dest / relative
        payload = origin.read_bytes()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.read_bytes() != payload:
                changed += 1
            target.write_bytes(payload)
        except PermissionError as exc:
            # The store lives OUTSIDE the repo, which is the whole point of it and also the one
            # thing a sandboxed agent is normally refused. Raising the bare PermissionError printed
            # a twelve-line Python traceback from inside a git hook, naming pathlib rather than the
            # cause — indistinguishable, to a reader, from the backup being broken.
            #
            # It still FAILS. It must: a backup that quietly does not run is worse than none, the
            # same reason the secret scan errors rather than skips when gitleaks is absent. What
            # changes is that the message says which of the two it is and what to do about it.
            grant = json.dumps({"sandbox": {"filesystem": {"allowWrite": [str(dest)]}}}, indent=2)
            raise SystemExit(
                f"\ncannot write to the backup store: {exc.filename}\n"
                f"  store: {dest}\n\n"
                "The store is outside the repository by design, so this is usually a sandbox\n"
                "or permissions restriction rather than a broken backup. Two fixes:\n\n"
                "  1. Run it yourself, outside the restriction:\n"
                "       uv run python tools/backup_local_only.py\n\n"
                "  2. Or grant write access to the store once, in the project-local\n"
                "     .claude/settings.local.json:\n\n"
                f"{textwrap.indent(grant, '       ')}\n\n"
                "Do NOT skip this hook to get past it — the files it copies are the ones git\n"
                "cannot restore, and this hook runs on checkout/merge for exactly that reason.\n"
            ) from exc

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

    # **Per file, not just per store.** The check above asks whether *any* commit exists, and it
    # passed for months while one file was copied and never versioned. A file git declines to add —
    # an ignore rule, a nested `.git`, a permission fault — leaves bytes on disk that look like a
    # backup and carry no history, so the "bad overwrite" this store exists to survive would
    # silently destroy the only good copy. `-z` because git octal-quotes non-ASCII paths otherwise,
    # and several stored filenames contain them.
    listed = _git(dest, "ls-files", "-z").stdout
    versioned = {name for name in listed.split("\0") if name}
    unversioned = sorted(str(f) for f in files if str(f) not in versioned)
    if unversioned:
        raise SystemExit(
            f"{len(unversioned)} file(s) were copied into {dest} but git refuses to track them, so "
            "they have no version history and a bad overwrite would be unrecoverable:\n  "
            + "\n  ".join(unversioned)
            + f"\n\nUsually an ignore rule. Check with:\n  git -C {dest} check-ignore -v <path>"
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
        help="compare the store against this checkout in both directions and exit non-zero on a "
        "loss, a gap or staleness",
    )
    parser.add_argument("--message", default="snapshot", help="commit subject")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    files, skipped = collect(args.root)
    for name in skipped:
        logger.error("SKIPPED (looks like a credential)  %s", name)

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

    # Anything ignored, irreplaceable and undecided. Printed on every run rather than behind a
    # flag, because the failure this catches is silence: an allowlist cannot tell you what it
    # missed, and the biggest thing it ever missed sat there for months with both guards green.
    orphans = uncovered(args.root)
    for name in orphans[:20]:
        logger.warning("NOT COVERED    %s", name)
    if len(orphans) > 20:
        logger.warning("NOT COVERED    ... and %d more", len(orphans) - 20)

    if args.dry_run:
        for relative in files:
            logger.info("  %s", relative)
        logger.info("would snapshot to %s", args.dest)
        return 1 if skipped else 0

    if args.verify:
        if not args.dest.is_dir():
            logger.error("no backup store at %s — run without --verify to create one", args.dest)
            return 1
        absent, differing, lost = verify(args.root, args.dest, files)
        # Loss first, and loudest. The other two mean "run the tool"; this one means "stop".
        for name in lost:
            logger.error("LOST FROM THE CHECKOUT  %s", name)
        for name in absent:
            logger.error("NOT BACKED UP  %s", name)
        for name in differing:
            logger.warning("out of date    %s", name)
        if lost:
            logger.error(
                "%d files are in the store and NOT in your checkout. Restore them before doing "
                "anything else:\n  cp %s/<path> <path>",
                len(lost),
                args.dest,
            )
            return 1
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
    # The snapshot happened either way — a skipped credential must not cost the other hundred files
    # their backup — but the exit code carries the warning so a caller cannot miss it.
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
