"""Freeze the repo's standard files at a release, in-tree, so a rewrite can be compared.

Git already holds every version. This exists because *finding* one means knowing a rewrite
happened, then hunting the commit that did it — and the rewrites worth comparing are exactly the
ones nobody remembers making. `docs/DESIGN.md` went 199 -> 488 lines in a single commit: of its 30
rules, 19 survived reworded and **nine were dropped with no replacement anywhere in the repo**. None
of that was visible without a deliberate diff against a version somebody had to go looking for.

So the last two released versions of each standard file sit beside the live one, tracked on the
same branch.

    uv run python tools/snapshot_standards.py            # snapshot the latest tag
    uv run python tools/snapshot_standards.py --ref v0.12.0
    uv run python tools/snapshot_standards.py --check    # non-zero if a snapshot is missing
    uv run python tools/snapshot_standards.py --prune    # retire versions past the retention limit

**Nothing is deleted without `--prune`**, which prints what it would remove and needs confirming.
A snapshot that outlives its usefulness is clutter; one deleted by a tool nobody watched is the
failure this repo keeps paying for.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = REPO_ROOT / "docs" / "standards-history"

#: The files that carry instructions, configuration or conventions — the ones where a bad edit
#: breaks something far from the edit. Deliberately short: a snapshot of everything is a second
#: repo, and the point is that a reader can hold this list in their head.
#:
#: `CLAUDE.md` is not here, and that is not an oversight: it is one line importing `AGENTS.md`, so
#: snapshotting it would archive the pointer and not the thing pointed at.
STANDARDS: tuple[str, ...] = (
    "AGENTS.md",
    "docs/DESIGN.md",
    ".github/workflows/ci.yml",
    ".pre-commit-config.yaml",
    "pyproject.toml",
    ".gitignore",
)

#: How many released versions to keep per file. Two is the smallest number that answers both
#: questions worth asking — "what did this look like before the rewrite?" and "was it already
#: drifting before that?"
RETENTION = 2

_BANNER_MARK = "FROZEN COPY — NOT IN FORCE"

_COMMENT_PREFIX = {".yml": "#", ".yaml": "#", ".toml": "#", "": "#"}


def _banner(source: str, version: str, comment: str | None) -> str:
    """Build the header that stops an agent reading an archived copy as live instructions."""
    lines = [
        f"**{_BANNER_MARK}.** This is `{source}` exactly as it shipped in **{version}**, kept",
        "so a rewrite can be diffed against what it replaced. It is history, not policy: the live",
        f"file at `{source}` is the one in force, and this copy is never edited. Snapshots are",
        "written by `tools/snapshot_standards.py`, checked by `tests/test_standards_history.py`.",
    ]
    if comment is None:  # markdown
        return "\n".join(f"> {line}" for line in lines) + "\n\n---\n\n"
    return "\n".join(f"{comment} {line}" for line in lines) + f"\n{comment}\n\n"


def _git(*args: str) -> str:
    """Run git in the repo and return stdout, raising with stderr on failure."""
    done = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def latest_tag() -> str:
    """The newest release tag, by version order rather than by date."""
    tags = [t for t in _git("tag", "--list", "v*").split() if re.fullmatch(r"v[\d.]+", t)]
    if not tags:
        raise RuntimeError("no v* tags in this repo — pass --ref explicitly")
    return sorted(tags, key=lambda t: [int(p) for p in t[1:].split(".")])[-1]


def archive_name(source: str, version: str) -> str:
    """`docs/DESIGN.md` at `v0.12.0` -> `DESIGN.v0.12.0.md`; dotfiles keep their leading dot."""
    stem = Path(source).name
    if stem.startswith("."):  # .gitignore, .pre-commit-config.yaml
        base, _, ext = stem[1:].partition(".")
        return f"{base}.{version}.{ext}" if ext else f"{base}.{version}"
    return f"{Path(stem).stem}.{version}{Path(stem).suffix}"


def existing(source: str) -> list[Path]:
    """Every archived version of one source file, newest version last."""
    stem = archive_name(source, "@").split("@")[0]
    found = sorted(ARCHIVE.glob(f"{stem}*")) if ARCHIVE.is_dir() else []

    def key(p: Path) -> list[int]:
        m = re.search(r"v(\d+(?:\.\d+)*)", p.name)
        return [int(x) for x in m.group(1).split(".")] if m else [0]

    return sorted(found, key=key)


def snapshot(source: str, version: str, *, force: bool = False) -> tuple[Path, bool]:
    """Write one archived copy. Returns the path and whether it was newly written."""
    out = ARCHIVE / archive_name(source, version)
    if out.exists() and not force:
        return out, False
    try:
        body = _git("show", f"{version}:{source}")
    except RuntimeError:
        return out, False
    suffix = Path(source).suffix
    comment = None if suffix == ".md" else _COMMENT_PREFIX.get(suffix, "#")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    out.write_text(_banner(source, version, comment) + body, encoding="utf-8")
    return out, True


def main() -> int:
    """Snapshot, check or prune. Returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", help="tag to snapshot (default: the newest v* tag)")
    ap.add_argument("--check", action="store_true", help="report gaps, write nothing")
    ap.add_argument("--prune", action="store_true", help="retire versions past the retention limit")
    ap.add_argument("--force", action="store_true", help="rewrite a snapshot that already exists")
    args = ap.parse_args()

    version = args.ref or latest_tag()

    if args.check:
        missing = [s for s in STANDARDS if not (ARCHIVE / archive_name(s, version)).exists()]
        for s in missing:
            print(f"missing: {archive_name(s, version)}")
        print(
            f"{len(STANDARDS) - len(missing)}/{len(STANDARDS)} standards snapshotted at {version}"
        )
        return 1 if missing else 0

    if args.prune:
        stale = [p for s in STANDARDS for p in existing(s)[:-RETENTION]]
        if not stale:
            print(f"nothing past the retention limit of {RETENTION}")
            return 0
        for p in stale:
            print(f"would retire: {p.relative_to(REPO_ROOT)}")
        print(
            f"\n{len(stale)} file(s). Delete them yourself once you have read the list — this tool "
            "does not remove files."
        )
        return 0

    wrote = 0
    for source in STANDARDS:
        if not (REPO_ROOT / source).exists():
            print(f"skipped (no live file): {source}")
            continue
        out, fresh = snapshot(source, version, force=args.force)
        print(f"{'wrote' if fresh else 'kept '} {out.relative_to(REPO_ROOT)}")
        wrote += fresh
    print(f"\n{wrote} new snapshot(s) at {version}; retention is {RETENTION} per file.")
    over = [s for s in STANDARDS if len(existing(s)) > RETENTION]
    if over:
        print(f"{len(over)} file(s) now exceed retention — run --prune to see them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
