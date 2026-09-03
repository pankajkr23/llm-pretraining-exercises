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

**Two bypasses this guard shipped with, both found by auditing it against its own claims.** Neither
was subtle, and both are worth stating because the shape recurs: the guard answered its question
correctly, about a call it never saw.

*It took the repo root from its own `__file__`.* Under `claude --worktree` the branch is checked out
at `.claude/worktrees/<name>/`, so a write to that worktree's `uv.lock` resolved to
`.claude/worktrees/<name>/uv.lock` and matched no pattern — every protected path was unprotected in
the one mode parallel work depends on. The root now comes from the payload's `cwd`.

*`Bash` was not in `WRITING_TOOLS`*, so `echo >`, `sed -i` and `rm` went straight past. The guard
did not prevent the incident it cites as its reason for existing, reproducible in one `sed`.
**Be precise about what the fix buys: a shell is Turing-complete and this raises the cost of a
bypass rather than closing it.** `python -c "open('uv.lock','w')"` builds its path at runtime and no
static reader will see it. What the check does catch is the whole class of *casual* writes — a
redirect, an in-place edit, a copy over the top — which is what an agent taking a shortcut actually
writes, and what the two incidents behind this file actually were.
"""

import fnmatch
import json
import shlex
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES = REPO_ROOT / "tools" / "agent_fleet" / "guard_rules.toml"

#: Tools that write a named file. `Bash` is deliberately **not** here — it carries a command rather
#: than a `file_path`, so it is read by `bash_write_targets` instead.
WRITING_TOOLS = frozenset({"Write", "Edit", "NotebookEdit", "MultiEdit"})

#: Shell operators that end one command and begin another. Each segment is judged on its own, so
#: `grep -rn uv.lock . > /tmp/out` is a read of `uv.lock` and a write to `/tmp/out`, not both.
SEGMENT_SEPARATORS = (";", "&&", "||", "|", "\n")

#: Redirection tokens. The token that follows one is written to.
REDIRECTS = frozenset({">", ">>", "1>", "2>", "&>", ">|"})

#: Commands where **every** path argument is written or destroyed.
WRITES_EVERY_ARGUMENT = frozenset({"rm", "shred", "truncate", "touch", "tee", "unlink"})

#: Commands where only the **last** path argument is the destination. Listing them separately is
#: what keeps `cp uv.lock /tmp/backup` — a perfectly good thing to do — from being refused.
WRITES_LAST_ARGUMENT = frozenset({"cp", "mv", "ln", "install", "rsync"})

#: Prefixes that wrap a real command without being one.
COMMAND_WRAPPERS = frozenset({"sudo", "env", "command", "nohup", "time", "xargs", "exec"})


def _is_in_place_edit(command_word: str, tokens: list[str]) -> bool:
    """True for `sed -i` / `perl -i`, which rewrite their arguments in place.

    Checked by flag rather than by name alone, because `sed` without `-i` writes to stdout and is
    one of the most common read commands there is.
    """
    if command_word not in {"sed", "perl", "ruby", "gsed"}:
        return False
    return any(token == "-i" or token.startswith("-i") for token in tokens)


def _segments(command: str) -> list[list[str]]:
    """Split a shell command into segments and tokenize each one.

    Returns an empty list when the command cannot be tokenized — an unbalanced quote, most often.
    That is the honest answer: it means this reader does not understand the command, and the caller
    treats "no targets found" as "nothing to say", which is why the caveat in the module docstring
    matters.
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return []

    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in SEGMENT_SEPARATORS or set(token) <= {"&", "|", ";"} and token:
            segments.append([])
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def _command_word(tokens: list[str]) -> tuple[str, list[str]]:
    """The real command and its arguments, looking past `VAR=x`, `sudo`, `env` and friends."""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in COMMAND_WRAPPERS or ("=" in token and not token.startswith("-")):
            index += 1
            continue
        return token, tokens[index + 1 :]
    return "", []


def _written_paths(tokens: list[str]) -> list[str]:
    """Every argument in one segment that this segment writes to."""
    targets: list[str] = []

    for position, token in enumerate(tokens):
        if token in REDIRECTS and position + 1 < len(tokens):
            targets.append(tokens[position + 1])
        elif len(token) > 1 and token[0] == ">" and token[1] != ">":
            targets.append(token[1:])  # the attached `>file` form

    command_word, arguments = _command_word(tokens)
    values = [argument for argument in arguments if not argument.startswith("-")]

    if command_word in WRITES_EVERY_ARGUMENT or _is_in_place_edit(command_word, arguments):
        targets.extend(values)
    elif command_word in WRITES_LAST_ARGUMENT and values:
        targets.append(values[-1])
    elif command_word == "git" and values[:1] and values[0] in {"restore", "checkout", "clean"}:
        targets.extend(values[1:])  # these overwrite or delete working-tree files

    return targets


def bash_write_targets(command: str, root: Path) -> list[str]:
    """Repo-relative paths a `Bash` command appears to write, as strings.

    Only *appears to*: see the module docstring. A path this cannot see is a path it cannot guard.
    """
    relative: list[str] = []
    for tokens in _segments(command):
        for target in _written_paths(tokens):
            if not target or target.startswith("-"):
                continue
            try:
                candidate = Path(target)
                absolute = candidate if candidate.is_absolute() else root / candidate
                relative.append(absolute.resolve().relative_to(root).as_posix())
            except ValueError:
                continue  # outside the repo; not this guard's business
    return relative


def resolve_root(payload: dict, fallback: Path) -> Path:
    """The repo root **this call** is happening in, from the payload's `cwd`.

    Taking it from `__file__` instead is the worktree bug described above: it pins the root to
    wherever the script happens to live, which under `claude --worktree` is not where the work is.

    Walks up from `cwd` to the nearest `.git` — a *file* in a worktree, a directory in a checkout —
    so an agent that has changed into a subdirectory still resolves to the root rather than to
    wherever it stood.
    """
    raw = payload.get("cwd")
    if not raw:
        return fallback
    try:
        current = Path(raw).resolve()
    except (OSError, ValueError):
        return fallback
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


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
    tool_input = payload.get("tool_input", {})

    if tool == "Bash":
        targets = bash_write_targets(tool_input.get("command", "") or "", root)
    elif tool in WRITING_TOOLS:
        target = tool_input.get("file_path")
        if not target:
            return None
        try:
            targets = [Path(target).resolve().relative_to(root).as_posix()]
        except ValueError:
            return None  # outside the repo entirely; not this guard's business
    else:
        return None

    for rel in targets:
        refusal = _refuse(rel, root, rules)
        if refusal is not None:
            return refusal
    return None


def _refuse(rel: str, root: Path, rules: dict) -> str | None:
    """The refusal for writing one repo-relative path, or None to allow it."""
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

    reason = decide(payload, resolve_root(payload, REPO_ROOT), rules)
    if reason is None:
        return 0
    print(reason + CONTINUE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
