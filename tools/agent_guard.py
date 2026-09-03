"""Block a tool call an agent must not make, in every permission mode including bypass.

**This is the only layer nothing bypasses.** `permissions.deny` is settings and can be widened;
`AGENTS.md` is prose and can be ignored silently; a pre-commit hook is skippable with `--no-verify`
and absent on a fresh clone. A `PreToolUse` hook that exits **2** blocks the call outright.
Anthropic states the distinction plainly: *"An instruction like 'never edit `.env`' in CLAUDE.md
or a skill is a request, not a guarantee. A `PreToolUse` hook that blocks the edit is
enforcement."*

Wire it from `.claude/settings.local.json` (see `tools/install_agent_fleet.py`); the policy it reads
lives in the **tracked** `tools/agent_fleet/guard_rules.toml`, because the whole `.claude/` tree is
gitignored and a policy nobody can review or test in CI is not a policy.

**Four rules, each traceable to an incident**, all documented in the rules file: writes outside the
unit's declared scope, writes to measured data, writes to a guard the agent could weaken to make its
own work pass, and writes to a standard file mid-unit.

**Two design decisions that matter more than the rules.**

A block returns a **machine-readable instruction** rather than an error — "this is expected; log it
as a finding and continue". An agent that treats a block as a failure stops and waits, and an agent
that stalls overnight has failed differently but just as badly.

It **fails closed**. Malformed stdin, an unreadable rules file, an unparseable payload: all block.
The first version of a guard like this exited 0 on bad JSON, which means the one input an attacker
or a bug controls was also the one that disabled it.
"""

import fnmatch
import json
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES = REPO_ROOT / "tools" / "agent_fleet" / "guard_rules.toml"

#: Tools that write. Everything else is read-only and is never blocked by this guard.
WRITING_TOOLS = frozenset({"Write", "Edit", "NotebookEdit", "MultiEdit"})

#: Appended to every refusal. Without it an agent reads a block as a failure and stops; with it the
#: block is a routing instruction and the run continues.
CONTINUE = (
    "\n\nTHIS IS EXPECTED, NOT AN ERROR — it is a guard doing its job. Do not retry, and do not\n"
    "work around it. Record what you found as a finding in docs/agents/QUEUE.md and carry on with\n"
    "the planned work. If the unit genuinely needs this path, stop and ask for it to be added to\n"
    ".claude/UNIT.md, which is a human decision."
)


def load_rules(path: Path = RULES) -> dict:
    """Read the tracked policy. Raises rather than defaulting — see "fails closed" above."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _matches(rel: str, patterns: list[str]) -> str | None:
    """The first pattern this path matches, or None.

    `fnmatch` treats `*` as matching across separators, which is what the `**` entries want and is
    harmless for the rest: every pattern here is anchored at the repo root.
    """
    for pattern in patterns:
        if fnmatch.fnmatch(rel, pattern):
            return pattern
    return None


def unit_scope(root: Path, rules: dict) -> list[str] | None:
    """Path prefixes the current unit may write, or None when no unit is declared.

    Returns None rather than an empty list deliberately: **no unit file means the scope rule is
    inert**, not that everything is forbidden. A guard that fires on every write when the operator
    simply has not written a UNIT.md is a guard that gets uninstalled within a day.
    """
    unit = root / rules["scope"]["unit_file"]
    if not unit.is_file():
        return None
    scope: list[str] = []
    for line in unit.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- scope:"):
            scope.append(stripped.split("scope:", 1)[1].strip())
    return scope or None


def named_in_unit(root: Path, rules: dict, rel: str) -> bool:
    """True when `.claude/UNIT.md` names this exact path as one the unit may touch.

    This is the escape hatch for the guard and standard sections: editing them is legitimate when
    the unit *is* that work, and illegitimate when it is a way to make failing work pass. Naming the
    file is what separates the two, and it is a decision a human makes when writing the unit.
    """
    unit = root / rules["scope"]["unit_file"]
    return unit.is_file() and rel in unit.read_text(encoding="utf-8")


def decide(payload: dict, root: Path, rules: dict) -> str | None:
    """The refusal for this tool call, or None to allow it.

    Args:
        payload: The `PreToolUse` JSON Claude Code puts on stdin.
        root: The repo root.
        rules: The parsed policy.

    Returns:
        A reason to block with, or None.
    """
    halt = root / rules["halt"]["file"]
    if halt.is_file():
        return (
            f"HALTED — {halt.name} exists, so every tool call is blocked.\n"
            f"Remove it to resume: rm {halt.name}"
        )

    tool = payload.get("tool_name", "")
    if tool not in WRITING_TOOLS:
        return None

    target = payload.get("tool_input", {}).get("file_path")
    if not target:
        return None
    try:
        rel = Path(target).resolve().relative_to(root).as_posix()
    except ValueError:
        return None  # outside the repo entirely; not this guard's business

    for section in ("measured_data", "guards", "standards"):
        pattern = _matches(rel, rules[section]["patterns"])
        if pattern is None:
            continue
        # `measured_data` has no escape hatch on purpose: there is no unit for which rewriting a
        # frozen tokenizer or a recorded result is the work.
        if section != "measured_data" and named_in_unit(root, rules, rel):
            continue
        return f"BLOCKED {rel}\n  matched {section} pattern {pattern!r}\n  {rules[section]['why']}"

    scope = unit_scope(root, rules)
    if scope is not None and not any(rel.startswith(prefix) for prefix in scope):
        return (
            f"BLOCKED {rel}\n  outside this unit's declared scope:\n    "
            + "\n    ".join(scope)
            + "\n  Opportunistic edits are how one unit's change lands in another unit's review."
        )
    return None


def main() -> int:
    """Read the hook payload from stdin and block or allow. Returns the process exit code."""
    try:
        payload = json.load(sys.stdin)
        rules = load_rules()
    except Exception as exc:  # noqa: BLE001 — fail closed, see the module docstring
        print(f"BLOCKED — the guard could not evaluate this call: {exc}{CONTINUE}", file=sys.stderr)
        return 2

    reason = decide(payload, REPO_ROOT, rules)
    if reason is None:
        return 0
    print(reason + CONTINUE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
