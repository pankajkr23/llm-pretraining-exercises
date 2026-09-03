"""Wire the tracked fleet machinery into this clone's gitignored `.claude/` tree.

**Why an installer exists at all.** `.gitignore` excludes `.claude/` in its entirety, so anything
Claude Code reads from there — hook wiring, subagent definitions — is invisible to review, absent
from a fresh clone, and untestable in CI. That is the "reads as coverage" shape this repo keeps
finding, so the machinery is **tracked** (`tools/agent_guard.py`, `tools/agent_fleet/*`,
`docs/agents/reviewers/*`) and only the thin wiring that points at it is local. This copies one to
the other.

    uv run python tools/install_agent_fleet.py            # install or refresh
    uv run python tools/install_agent_fleet.py --check    # non-zero if the clone is behind
    uv run python tools/install_agent_fleet.py --drift    # non-zero ONLY if a copy was edited

**`--check` and `--drift` answer different questions, and conflating them is why the second
exists.** `--check` is "is this clone current?", which is non-zero on a fresh clone where nothing is
installed yet — correct for a human, wrong for a hook, because it would make every new clone's first
`git pull` fail with a message about drift when nothing has drifted.

`--drift` ignores what is merely *absent* and fails only where a deployed copy **exists and differs
from its tracked source**. That is the failure worth catching automatically: `.claude/` is
gitignored, so a hand-edit there is invisible to review, to CI and to a clone, and would otherwise
survive until somebody thought to look. It is wired to `post-merge`, beside the progress-log check.

**It never overwrites hook wiring that already exists.** `.claude/settings.local.json` holds what
agents may run without asking, and losing it silently *shrinks* the permission surface rather than
failing — which is why `AGENTS.md` puts it in the backup set. So settings are merged key by key and
an existing `hooks` key is reported and left alone; only the reviewer definitions are refreshed from
their tracked originals, and those have no local edits worth keeping by construction.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE = REPO_ROOT / ".claude"

#: Tracked source -> the local path Claude Code reads it from.
REVIEWERS = REPO_ROOT / "docs" / "agents" / "reviewers"
AGENTS_OUT = CLAUDE / "agents"

#: The hook wiring, merged into `.claude/settings.local.json` rather than written over it.
#:
#: `PreToolUse` with a `Write|Edit` matcher: the guard is about writes, and matching everything
#: would run a subprocess on every Read in the run for nothing.
HOOK_WIRING: dict = {
    "hooks": {
        "PreToolUse": [
            {
                "matcher": "Write|Edit|NotebookEdit|MultiEdit",
                "hooks": [
                    {
                        "type": "command",
                        "command": "uv run python tools/agent_guard.py",
                    }
                ],
            }
        ]
    }
}


def _display(path: Path) -> str:
    """A repo-relative path for a message, falling back to the absolute one.

    `relative_to` raises for anything outside the repo, which is every path a test drives this with
    — and a crash while *reporting* would take down the install it was reporting on.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def install_reviewers(*, dry_run: bool = False) -> list[str]:
    """Copy the tracked reviewer definitions into `.claude/agents/`. Returns what changed."""
    changed: list[str] = []
    out = Path(AGENTS_OUT)
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
    for source in sorted(REVIEWERS.glob("*.md")):
        target = out / source.name
        if target.is_file() and target.read_bytes() == source.read_bytes():
            continue
        changed.append(_display(target))
        if not dry_run:
            shutil.copy2(source, target)
    return changed


def drifted_reviewers() -> list[str]:
    """Deployed reviewer copies that **exist and differ** from their tracked source.

    Deliberately not "everything that needs updating". A copy that is simply absent is a clone that
    has not run the installer, which is an ordinary state and not a finding — reporting it as one
    would make every fresh clone's first pull red and teach the reader to ignore this check.

    A copy that exists and differs is the real case: `.claude/` is gitignored, so an edit there is
    invisible to review, to CI and to every other clone. Nothing else in the repository would ever
    notice it.
    """
    out: list[str] = []
    for source in sorted(REVIEWERS.glob("*.md")):
        target = Path(AGENTS_OUT) / source.name
        if target.is_file() and target.read_bytes() != source.read_bytes():
            out.append(_display(target))
    return out


def install_hooks(*, dry_run: bool = False) -> list[str]:
    """Merge the hook wiring into `.claude/settings.local.json`, leaving existing keys alone.

    **Merged, never replaced.** That file records what agents in this repo may run without asking;
    overwriting it would quietly change the permission surface rather than failing, which
    `AGENTS.md` names as the reason it is in the backup set at all.
    """
    settings_path = Path(CLAUDE) / "settings.local.json"
    current: dict = {}
    if settings_path.is_file():
        current = json.loads(settings_path.read_text(encoding="utf-8"))
    if current.get("hooks") == HOOK_WIRING["hooks"]:
        return []
    if "hooks" in current:
        return [f"{_display(settings_path)}: has its own `hooks` — left alone"]
    if not dry_run:
        current["hooks"] = HOOK_WIRING["hooks"]
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return [f"{_display(settings_path)}: PreToolUse guard wired"]


def main() -> int:
    """Install or check. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="non-zero if the clone is behind")
    parser.add_argument(
        "--drift",
        action="store_true",
        help="non-zero only if a deployed copy exists and differs from its tracked source",
    )
    args = parser.parse_args()

    if args.drift:
        drift = drifted_reviewers()
        if not drift:
            print("no fleet file has been edited away from its tracked source")
            return 0
        print(
            "a deployed copy has drifted from the tracked source it came from:\n  "
            + "\n  ".join(drift)
            + "\n\n`.claude/` is gitignored, so this edit is invisible to review, to CI and to "
            "every other\nclone. Edit `docs/agents/reviewers/` instead — that is the source — then "
            "re-run:\n\n    uv run python tools/install_agent_fleet.py",
            file=sys.stderr,
        )
        return 1

    pending = install_reviewers(dry_run=args.check) + install_hooks(dry_run=args.check)
    if not pending:
        print("the fleet wiring is current")
        return 0
    verb = "would update" if args.check else "updated"
    for item in pending:
        print(f"{verb}: {item}")
    if args.check:
        print("\nrun `uv run python tools/install_agent_fleet.py` to apply")
        return 1
    print(
        "\nThe guard is wired but INERT for scope until `.claude/UNIT.md` declares one — a guard\n"
        "that fires on every write when no unit is declared is a guard that gets uninstalled.\n"
        "Measured data, guard files and standard files are refused regardless."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
