"""The agent guard blocks what it must, allows what it must, and fails closed.

`tools/agent_guard.py` is the only layer no permission mode bypasses, so it is the one place where a
mistake is invisible: a guard that silently allows looks exactly like a guard that had nothing to
block. Every property below is written twice — the blocking case and the allowing case — because a
guard nobody has watched fail is not a guard, and one that blocks everything gets uninstalled.

**These run everywhere.** They read the tracked policy and synthetic payloads, never the
environment, so they are as true on a fresh clone as here.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from agent_guard import (  # noqa: E402
    CONTINUE,
    RULES,
    WRITING_TOOLS,
    bash_write_targets,  # noqa: F401  — imported so a rename breaks here, not silently
    decide,
    load_rules,
    resolve_root,
)


@pytest.fixture
def rules() -> dict:
    """The real, tracked policy — not a fixture copy, which would drift from it."""
    return load_rules()


def _write(path: str, tool: str = "Write") -> dict:
    """A `PreToolUse` payload for a write to `path`."""
    return {"tool_name": tool, "tool_input": {"file_path": str(REPO_ROOT / path)}}


def test_the_policy_is_tracked_so_ci_and_review_can_see_it() -> None:
    """The whole `.claude/` tree is gitignored, so a policy living there is invisible to everyone.

    This is the reason the rules are a tracked TOML file rather than part of the hook wiring: a
    policy nobody can review in a PR and no CI job can test is the "reads as coverage" shape.
    """
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", str(RULES.relative_to(REPO_ROOT))],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert listed, f"{RULES.relative_to(REPO_ROOT)} is not tracked, so nothing can review it"


def test_measured_data_is_refused(rules) -> None:
    """A sweep rewrote the frozen tokenizer, whose hash every exercise-06 shard manifest pins."""
    for path in (
        "src/exercises/02-tokenization/web/tokenizer.json",
        "src/exercises/02-tokenization/corpus/hi.faithful.txt",
        "src/exercises/05-datamixtures-and-curriculum/results/step0.json",
        "uv.lock",
    ):
        refusal = decide(_write(path), REPO_ROOT, rules)
        assert refusal is not None, path
        assert "measured_data" in refusal, refusal


def test_a_guard_file_is_refused(rules) -> None:
    """The `return []` incident: two invariants returned "no findings" for four commits."""
    refusal = decide(_write("tests/test_forbidden_vocabulary.py"), REPO_ROOT, rules)
    assert refusal is not None
    assert "guards" in refusal


def test_the_guard_refuses_edits_to_itself(rules) -> None:
    """A guard an agent can rewrite is a guard an agent can remove."""
    for path in ("tools/agent_guard.py", "tools/agent_fleet/guard_rules.toml"):
        assert decide(_write(path), REPO_ROOT, rules) is not None, path


def test_ordinary_source_is_allowed_with_no_unit_declared(rules) -> None:
    """**The most important negative case.** A guard that fires constantly gets uninstalled.

    With no `.claude/UNIT.md` the scope rule is inert by design, so routine work is untouched.
    """
    for path in (
        "src/exercises/07-model-embeddings-internals/src/embeddings/codec.py",
        "README.md",
        "docs/agents/QUEUE.md",
    ):
        assert decide(_write(path), REPO_ROOT, rules) is None, path


def test_a_read_only_tool_is_never_blocked(rules) -> None:
    """The guard is about writes. Blocking a Read would make exploration impossible.

    `Bash` is not in this list because it carries a command rather than a `file_path`; it has its
    own blocking/allowing pair below.
    """
    for tool in ("Read", "Grep", "Glob"):
        assert tool not in WRITING_TOOLS
        payload = {"tool_name": tool, "tool_input": {"file_path": str(REPO_ROOT / "uv.lock")}}
        assert decide(payload, REPO_ROOT, rules) is None, tool


def test_a_path_outside_the_repo_is_not_this_guards_business(rules) -> None:
    """Blocking `$TMPDIR` writes would stop the very scratch work the conventions ask for."""
    payload = {"tool_name": "Write", "tool_input": {"file_path": "/tmp/claude/scratch.txt"}}
    assert decide(payload, REPO_ROOT, rules) is None


def test_the_halt_file_stops_everything(rules, tmp_path) -> None:
    """One `touch` halts a run already in flight, which is the point of checking it first."""
    (tmp_path / rules["halt"]["file"]).write_text("", encoding="utf-8")
    refusal = decide(_write("README.md"), tmp_path, rules)
    assert refusal is not None
    assert "HALTED" in refusal


def test_naming_a_file_in_the_unit_allows_a_guard_edit_but_never_measured_data(tmp_path) -> None:
    """The escape hatch, and the one place it deliberately does not exist.

    Editing a guard is legitimate when the unit *is* that work and illegitimate when it is a way to
    make failing work pass; naming the file in `UNIT.md` is what separates them. There is no unit
    for which rewriting a frozen tokenizer is the work, so measured data has no hatch at all.
    """
    rules = load_rules()
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "UNIT.md").write_text(
        "- scope: tests/\nAllowed: tests/test_forbidden_vocabulary.py\n"
        "src/exercises/02-tokenization/web/tokenizer.json\n",
        encoding="utf-8",
    )
    for name in ("tests", "src"):
        (tmp_path / name).mkdir(exist_ok=True)

    guard_edit = {
        "tool_name": "Write",
        "tool_input": {"file_path": str(tmp_path / "tests/test_forbidden_vocabulary.py")},
    }
    assert decide(guard_edit, tmp_path, rules) is None, "naming it in UNIT.md must permit it"

    data_edit = {
        "tool_name": "Write",
        "tool_input": {
            "file_path": str(tmp_path / "src/exercises/02-tokenization/web/tokenizer.json")
        },
    }
    refusal = decide(data_edit, tmp_path, rules)
    assert refusal is not None, "measured data has no escape hatch, even when named"
    assert "measured_data" in refusal


def test_a_write_outside_the_declared_scope_is_refused(tmp_path) -> None:
    """Opportunistic edits are how one unit's change lands in another unit's review."""
    rules = load_rules()
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "UNIT.md").write_text(
        "- scope: src/exercises/07-x/\n", encoding="utf-8"
    )

    inside = {
        "tool_name": "Write",
        "tool_input": {"file_path": str(tmp_path / "src/exercises/07-x/a.py")},
    }
    outside = {
        "tool_name": "Write",
        "tool_input": {"file_path": str(tmp_path / "src/exercises/08-y/b.py")},
    }
    for path in ("src/exercises/07-x", "src/exercises/08-y"):
        (tmp_path / path).mkdir(parents=True, exist_ok=True)

    assert decide(inside, tmp_path, rules) is None
    refusal = decide(outside, tmp_path, rules)
    assert refusal is not None
    assert "outside this unit's declared scope" in refusal


def test_every_refusal_tells_the_agent_to_continue() -> None:
    """An agent that reads a block as a failure stops and waits.

    A run that stalls overnight has failed differently from one that edits the wrong file, but just
    as badly — so the refusal is a routing instruction, not an error.
    """
    assert "THIS IS EXPECTED, NOT AN ERROR" in CONTINUE
    assert "Do not retry" in CONTINUE


def test_the_hook_fails_closed_on_malformed_input() -> None:
    """The one input a bug or an attacker controls must not be the one that disables the guard."""
    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "agent_guard.py")],
        input="not json at all",
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2, "malformed stdin must BLOCK, not allow"
    assert "could not evaluate" in done.stderr


def test_the_hook_exits_two_to_block_and_zero_to_allow() -> None:
    """Exit 2 is the only code that blocks through the hook alone; 1 is a non-blocking error."""
    script = REPO_ROOT / "tools" / "agent_guard.py"
    blocked = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(_write("uv.lock")),
        capture_output=True,
        text=True,
        check=False,
    )
    assert blocked.returncode == 2, blocked.stderr

    allowed = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(_write("README.md")),
        capture_output=True,
        text=True,
        check=False,
    )
    assert allowed.returncode == 0, allowed.stderr


def test_every_section_states_why_it_exists() -> None:
    """A rule with no reason is the first one somebody deletes when it gets in the way."""
    rules = load_rules()
    for section in ("measured_data", "guards", "standards"):
        why = rules[section].get("why", "")
        assert len(why.split()) >= 8, f"{section} has no reason with weight in it: {why!r}"
        assert rules[section]["patterns"], f"{section} has no patterns"


# --------------------------------------------------------------------------------------------
# The installer. It writes into the gitignored `.claude/` tree, so these drive it against
# `tmp_path` rather than the real one — a test that wired a live PreToolUse hook would change what
# every later test in the run is allowed to do.
# --------------------------------------------------------------------------------------------


def test_the_installer_copies_every_tracked_reviewer(tmp_path, monkeypatch) -> None:
    """A reviewer that never reaches `.claude/agents/` is a persona nothing can invoke."""
    import install_agent_fleet as installer

    monkeypatch.setattr(installer, "AGENTS_OUT", tmp_path / "agents")
    changed = installer.install_reviewers()
    copied = {p.name for p in (tmp_path / "agents").glob("*.md")}
    tracked = {p.name for p in installer.REVIEWERS.glob("*.md")}
    assert copied == tracked, f"installed {copied}, tracked {tracked}"
    assert changed, "a first install must report what it wrote"
    assert installer.install_reviewers() == [], "a second run must be a no-op"


def test_the_installer_never_overwrites_existing_hook_wiring(tmp_path, monkeypatch) -> None:
    """`.claude/settings.local.json` records what agents may run without asking.

    Overwriting it would silently *shrink* the permission surface rather than failing — which
    `AGENTS.md` names as the reason the file is in the backup set at all. So an existing `hooks`
    key is reported and left alone.
    """
    import install_agent_fleet as installer

    monkeypatch.setattr(installer, "CLAUDE", tmp_path)
    settings = tmp_path / "settings.local.json"
    settings.write_text(json.dumps({"hooks": {"PreToolUse": ["mine"]}}), encoding="utf-8")

    reported = installer.install_hooks()
    assert reported and "left alone" in reported[0]
    assert json.loads(settings.read_text(encoding="utf-8"))["hooks"]["PreToolUse"] == ["mine"]


def test_the_installer_preserves_unrelated_settings_keys(tmp_path, monkeypatch) -> None:
    """Merged key by key, so permissions and sandbox config survive a wiring refresh."""
    import install_agent_fleet as installer

    monkeypatch.setattr(installer, "CLAUDE", tmp_path)
    settings = tmp_path / "settings.local.json"
    settings.write_text(json.dumps({"permissions": {"allow": ["Bash(ls)"]}}), encoding="utf-8")

    installer.install_hooks()
    written = json.loads(settings.read_text(encoding="utf-8"))
    assert written["permissions"]["allow"] == ["Bash(ls)"], "unrelated keys must survive"
    assert written["hooks"]["PreToolUse"][0]["matcher"].startswith("Write|Edit")


def test_every_reviewer_declares_read_only_tools() -> None:
    """A reviewer that can write is the author grading itself.

    ICLR 2024: without external feedback, self-review *decreased* accuracy — models flip correct
    answers to wrong more often than the reverse. The separation is the whole mechanism, so it is
    asserted from the frontmatter rather than trusted.
    """
    import install_agent_fleet as installer

    for path in sorted(installer.REVIEWERS.glob("*.md")):
        head = path.read_text(encoding="utf-8").split("---")[1]
        tools = next(line for line in head.splitlines() if line.startswith("tools:"))
        allowed = {t.strip() for t in tools.split(":", 1)[1].split(",")}
        assert allowed <= {"Read", "Grep", "Glob"}, f"{path.name} can do more than read: {allowed}"
        for forbidden in ("Write", "Edit", "Bash", "NotebookEdit"):
            assert forbidden not in allowed, f"{path.name} declares {forbidden}"


# --- the two bypasses found by auditing the guard against its own claims -------------------------
#
# Both were live when this file was first written, and both are the same shape: the guard was asked
# a question it answered correctly, about a call it never saw. They are regression tests, so they
# name the bug rather than the fix.


def test_a_write_inside_a_worktree_is_still_guarded(rules, tmp_path) -> None:
    """The guard used to take the repo root from its own `__file__`, and fail open in a worktree.

    `claude --worktree` checks the branch out under `.claude/worktrees/<name>/`. With the root
    pinned to wherever the *script* lives, a write to that worktree's `uv.lock` resolved to
    `.claude/worktrees/<name>/uv.lock`, which matches no pattern in the policy — so every protected
    path was unprotected in the one mode parallel work depends on. Verified before the fix: the same
    payload was BLOCKED from the main checkout and ALLOWED from inside a worktree.

    The root now comes from the payload's `cwd`, which is the root the call is actually running in.
    """
    worktree = tmp_path / ".claude" / "worktrees" / "unit-07"
    (worktree / "tools").mkdir(parents=True)
    payload = {
        "tool_name": "Write",
        "cwd": str(worktree),
        "tool_input": {"file_path": str(worktree / "uv.lock")},
    }
    reason = decide(payload, resolve_root(payload, tmp_path), rules)
    assert reason is not None, "a worktree's uv.lock is measured data exactly as the main one is"
    assert "uv.lock" in reason


def test_a_bash_command_that_writes_a_protected_path_is_refused(rules) -> None:
    """`WRITING_TOOLS` excluded `Bash`, so `echo >` and `sed -i` sailed straight through.

    This is not a hypothetical gap: the incident the `[guards]` section exists for — `return []`
    injected into two invariants — is trivially reproducible with `sed -i`, so the guard did not
    prevent the thing it cites as its reason for existing.
    """
    for command in (
        "echo '{}' > uv.lock",
        "sed -i '' 's/return findings/return []/' "
        "src/exercises/05-datamixtures-and-curriculum/src/mixture/checks.py",
        "cat /dev/null >> tools/agent_guard.py",
        "rm tests/test_forbidden_vocabulary.py",
        "cp /tmp/x src/exercises/02-tokenization/web/tokenizer.json",
    ):
        payload = {"tool_name": "Bash", "cwd": str(REPO_ROOT), "tool_input": {"command": command}}
        assert decide(payload, REPO_ROOT, rules) is not None, command


def test_bash_that_only_reads_a_protected_path_is_allowed(rules) -> None:
    """The twin. A guard that blocked every mention of a protected path would block reading them.

    `grep`, `cat`, `wc` and `git log` over a guard file are exactly what an agent should do before
    reporting a finding about it, and blocking those makes the guard the thing to be worked around.
    """
    for command in (
        "cat uv.lock",
        "grep -n 'return' src/exercises/05-datamixtures-and-curriculum/src/mixture/checks.py",
        "wc -l tools/agent_guard.py",
        "git log --oneline -- tests/test_forbidden_vocabulary.py",
        "python -m pytest tests/test_forbidden_vocabulary.py -q",
    ):
        payload = {"tool_name": "Bash", "cwd": str(REPO_ROOT), "tool_input": {"command": command}}
        assert decide(payload, REPO_ROOT, rules) is None, command


def test_drift_is_detected_only_where_a_copy_exists_and_differs(tmp_path, monkeypatch) -> None:
    """The reviewer definitions live in two places, and nothing noticed when they disagreed.

    `docs/agents/reviewers/` is tracked; `.claude/agents/` is what Claude Code reads and is
    gitignored in its entirety. An edit to the second is invisible to review, to CI and to every
    other clone — so it would survive until somebody thought to look.

    **The distinction this asserts is the whole reason `--drift` exists separately from `--check`.**
    A copy that is merely *absent* is a clone that has not run the installer: ordinary, not a
    finding, and reporting it as one would make every fresh clone's first `git pull` red and teach
    the reader to ignore the check. A copy that exists and *differs* is the real case.

    Driven against a temporary tree rather than the live `.claude/`, which the sandbox refuses to
    write — an earlier attempt to prove this by editing the real file silently did nothing and
    reported "current", which was evidence of exactly nothing.
    """
    import install_agent_fleet as fleet

    source = tmp_path / "reviewers"
    deployed = tmp_path / "agents"
    source.mkdir()
    deployed.mkdir()
    (source / "reader.md").write_text("tools: Read, Grep, Glob\n", encoding="utf-8")
    (source / "absent.md").write_text("tools: Read\n", encoding="utf-8")
    monkeypatch.setattr(fleet, "REVIEWERS", source)
    monkeypatch.setattr(fleet, "AGENTS_OUT", deployed)

    # Absent from the deployed tree: not drift.
    assert fleet.drifted_reviewers() == [], "an uninstalled copy must not read as drift"

    # Present and identical: not drift.
    (deployed / "reader.md").write_text("tools: Read, Grep, Glob\n", encoding="utf-8")
    assert fleet.drifted_reviewers() == []

    # Present and edited: drift, and it names the file.
    (deployed / "reader.md").write_text("tools: Read, Grep, Glob, Write\n", encoding="utf-8")
    found = fleet.drifted_reviewers()
    assert len(found) == 1, found
    assert "reader.md" in found[0]
