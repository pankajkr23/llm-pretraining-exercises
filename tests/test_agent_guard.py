"""The agent guard blocks what it must, allows what it must, and fails closed.

`tools/agent_guard.py` is the only layer no permission mode bypasses, so it is the one place where a
mistake is invisible: a guard that silently allows looks exactly like a guard that had nothing to
block. Every property below is written twice — the blocking case and the allowing case — because a
guard nobody has watched fail is not a guard, and one that blocks everything gets uninstalled.

**These run everywhere.** They read the tracked policy and synthetic payloads, so they are as true
on a fresh clone as here.

**That claim used to be false, and the way it failed is worth keeping.** Four tests drove `decide()`
against the *real* repository root, which means they read whatever `.claude/UNIT.md` the current
unit had written. `UNIT.md` is gitignored, so it exists only on the machine running a unit — and
naming a guard file in it legitimately unlocks that file. The moment a unit declared a scope, four
tests inverted: `test_ordinary_source_is_allowed_with_no_unit_declared` failed although its own name
states the precondition, and `test_the_guard_refuses_edits_to_itself` failed because the guard was
correctly permitting the edit the unit existed to make.

They were green in CI, which has no `UNIT.md`, and red on the working checkout doing the work. That
is this repository's recurring defect — a gate whose result depends on a file only one machine has —
pointing the other way for once. Anything asserting a *default* now runs against an empty root.
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
    destructive_git,
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


def test_the_guard_refuses_edits_to_itself(rules, tmp_path) -> None:
    """A guard an agent can rewrite is a guard an agent can remove.

    Against an empty root, because the real one carries a `UNIT.md` whenever a unit is editing the
    guard — which is legitimate and is exactly what naming a file there means. This test is about
    the default, and the default is what an undeclared agent meets.
    """
    for path in ("tools/agent_guard.py", "tools/agent_fleet/guard_rules.toml"):
        payload = {
            "tool_name": "Write",
            "cwd": str(tmp_path),
            "tool_input": {"file_path": str(tmp_path / path)},
        }
        assert decide(payload, tmp_path, rules) is not None, path


def test_ordinary_source_is_allowed_with_no_unit_declared(rules, tmp_path) -> None:
    """**The most important negative case.** A guard that fires constantly gets uninstalled.

    With no `.claude/UNIT.md` the scope rule is inert by design, so routine work is untouched.

    **Driven against an empty root, because the name of this test is a precondition.** It used to
    run against the real repository, which has a `UNIT.md` whenever a unit is in flight — so the
    one case it exists to prove was decided by a gitignored file that only the machine running the
    unit has. It passed in CI and failed on a working checkout, which is the wrong way round.
    """
    for path in (
        "src/exercises/07-model-embeddings-internals/src/embeddings/codec.py",
        "README.md",
        "docs/agents/QUEUE.md",
    ):
        payload = {
            "tool_name": "Write",
            "cwd": str(tmp_path),
            "tool_input": {"file_path": str(tmp_path / path)},
        }
        assert decide(payload, tmp_path, rules) is None, path


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


def test_the_hook_exits_two_to_block_and_zero_to_allow(tmp_path) -> None:
    """Exit 2 is the only code that blocks through the hook alone; 1 is a non-blocking error.

    **The allowing case runs against an isolated root**, because the real one carries whatever
    `.claude/UNIT.md` the current unit declared, and a path outside that scope is correctly refused.
    Pinning this to the real repository made the result depend on a gitignored file that only one
    machine has: green in CI, red on a working checkout, for a reason unrelated to what it tests.
    """
    script = REPO_ROOT / "tools" / "agent_guard.py"
    blocked = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(_write("uv.lock")),
        capture_output=True,
        text=True,
        check=False,
    )
    assert blocked.returncode == 2, blocked.stderr

    (tmp_path / ".git").mkdir()
    payload = {
        "tool_name": "Write",
        "cwd": str(tmp_path),
        "tool_input": {"file_path": str(tmp_path / "README.md")},
    }
    allowed = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    assert allowed.returncode == 0, allowed.stderr


# --------------------------------------------------------------------------------------------
# Destructive git. These commands name no path, so the path matcher sees nothing to check and the
# call passes a guard that is working exactly as designed. The flag is the whole distinction.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "refused"),
    [
        # `-x` and `-X` delete ignored files: every notebook, builder and requirements document.
        ("git clean -fdx", True),
        ("git clean -x", True),
        ("git clean -X .", True),
        # ...and plain `-fd` does NOT, because those files are ignored rather than untracked.
        ("git clean -fd", False),
        # bundling is how a destructive command travels beside an innocent one
        ("git checkout main && git clean -xfd", True),
        ("git pull; git stash --all", True),
        # `--all` stashes the same set `clean -x` deletes; `-u` does not
        ("git stash -a", True),
        ("git stash -u", False),
        ("git reset --hard HEAD", True),
        ("git reset --soft HEAD~1", False),
        ("git push --force origin feature", True),
        ("git push --force-with-lease origin feature", True),
        ("git push origin feature", False),
        ("git tag -d v1.0.0", True),
        ("git tag v1.0.0", False),
        ("git branch -D feature", True),
        ("git branch -d feature", False),
        # git's own pre-subcommand options must not hide the subcommand
        ("git -C ../store clean -x", True),
        # ...and the store's own documented removal step is not destructive
        ("git -C ../store rm notebooks/x.ipynb && git -C ../store commit -m why", False),
        # not a git invocation at all
        ("echo git clean -x", False),
        ("git status --short", False),
    ],
)
def test_a_destructive_git_command_is_refused_by_its_flag(rules, command, refused) -> None:
    """Every case here is a command this repository's own rulebook names, safe and unsafe.

    The pairs matter more than the refusals. `clean -fdx` and `clean -fd`, `stash -a` and
    `stash -u`, `branch -D` and `branch -d`, `reset --hard` and `reset --soft` — in each the
    command is identical and only the flag decides. A guard written against the command name would
    block the safe half too and be uninstalled within a day.
    """
    assert (destructive_git(command, rules) is not None) is refused


def test_the_destructive_check_sees_a_command_the_path_matcher_cannot(rules) -> None:
    """The reason this check exists at all, asserted rather than described.

    `git clean -fdx` with no path argument writes nothing a path matcher can name, so
    `bash_write_targets` returns an empty list and every pattern in the policy is irrelevant. This
    asserts both halves: the old route is blind, and the new one is not.
    """
    command = "git clean -fdx"
    assert bash_write_targets(command, REPO_ROOT) == [], (
        "if this ever returns a target the premise has changed and the test below is checking "
        "something else"
    )
    assert destructive_git(command, rules) is not None


def test_a_clustered_short_flag_is_read_letter_by_letter(rules) -> None:
    """`-fdx` is the form people type, and a check comparing arguments to `-x` never fires on it."""
    from agent_guard import _short_flag_letters  # noqa: PLC0415

    assert _short_flag_letters("-fdx") == {"f", "d", "x"}
    assert _short_flag_letters("--force") == set()
    assert _short_flag_letters("-") == set()
    assert _short_flag_letters("path") == set()


def test_every_destructive_rule_states_why_and_names_a_flag(rules) -> None:
    """Both directions: a rule with no flag matches nothing, and one with no reason gets deleted."""
    entries = rules["destructive_git"]["rules"]
    assert entries, "the section exists and refuses nothing"
    for entry in entries:
        assert entry["flags"], f"{entry['subcommand']} lists no flag, so it can never fire"
        assert len(entry["why"].split()) >= 8, f"{entry['subcommand']} has no reason with weight"


def test_the_hook_blocks_a_destructive_git_command_end_to_end() -> None:
    """Through the real process, because `decide` returning a string is not the same as exit 2."""
    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "agent_guard.py")],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "git clean -fdx"}}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2, done.stdout + done.stderr
    assert "destructive_git" in done.stderr


def test_every_section_states_why_it_exists() -> None:
    """A rule with no reason is the first one somebody deletes when it gets in the way."""
    from agent_guard import _pattern_sections  # noqa: PLC0415

    rules = load_rules()
    # Derived, so a section added to the policy cannot skip this check by not being listed here.
    for section in _pattern_sections(rules):
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

    # **The property, not one spelling of it.** This asserted `.startswith("Write|Edit")`, which is
    # a guard on the literal string rather than on what the matcher admits — so adding `Bash|` to
    # the front, the fix for a hole that made the guard's whole Bash branch unreachable, turned it
    # red for no reason. `AGENTS.md`: a guard that names one implementation of a property will fail
    # every other implementation, and the pressure is then to reword good work to satisfy the test.
    import re  # noqa: PLC0415

    matcher = written["hooks"]["PreToolUse"][0]["matcher"]
    for tool in ("Bash", *sorted(WRITING_TOOLS)):
        assert re.search(matcher, tool), f"the installed matcher does not admit {tool!r}"


#: Tools that can change the repository, or run something that can. **The property, not a list of
#: the three tools the personas happened to use.**
#:
#: This started as an allowlist — `tools <= {Read, Grep, Glob}` — and the first persona that needed
#: anything else broke it: `research` fetches sources, which takes `WebSearch` and `WebFetch`. Those
#: reach outward and **cannot write**, so they do not touch the property this assertion exists for,
#: and an allowlist would have forced the choice between a persona that cannot do its job and a
#: guard quietly widened to let it. `AGENTS.md`: a guard that names one implementation of a property
#: will fail every other implementation, and the pressure is then to reword good work to satisfy it.
WRITING_TOOLS_IN_A_PERSONA = frozenset(
    {"Write", "Edit", "MultiEdit", "NotebookEdit", "Bash", "Task", "Agent"}
)


def test_every_reviewer_declares_read_only_tools() -> None:
    """A reviewer that can write is the author grading itself.

    ICLR 2024: without external feedback, self-review *decreased* accuracy — models flip correct
    answers to wrong more often than the reverse. The separation is the whole mechanism, so it is
    asserted from the frontmatter rather than trusted.
    """
    import install_agent_fleet as installer

    granted = []
    for path in sorted(installer.REVIEWERS.glob("*.md")):
        head = path.read_text(encoding="utf-8").split("---")[1]
        tools = next(line for line in head.splitlines() if line.startswith("tools:"))
        allowed = {t.strip() for t in tools.split(":", 1)[1].split(",") if t.strip()}
        for tool in sorted(allowed & WRITING_TOOLS_IN_A_PERSONA):
            granted.append(f"{path.name} declares `{tool}`")
    assert not granted, "a read-only persona can write:\n  " + "\n  ".join(granted)


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


def test_a_bash_command_that_writes_a_protected_path_is_refused(rules, tmp_path) -> None:
    """`WRITING_TOOLS` excluded `Bash`, so `echo >` and `sed -i` sailed straight through.

    This is not a hypothetical gap: the incident the `[guards]` section exists for — `return []`
    injected into two invariants — is trivially reproducible with `sed -i`, so the guard did not
    prevent the thing it cites as its reason for existing.

    **Driven against an empty root rather than the real one**, because this test is about the
    *pattern* rules and the real root carries whatever `.claude/UNIT.md` the current unit wrote.
    Naming a guard file in a unit legitimately unlocks it, so with a unit declared this test failed
    on a working checkout and passed in CI — a result that depends on a gitignored file only one
    machine has, which is the shape this repository keeps being bitten by.
    """
    for command in (
        "echo '{}' > uv.lock",
        "sed -i '' 's/return findings/return []/' "
        "src/exercises/05-datamixtures-and-curriculum/src/mixture/checks.py",
        "cat /dev/null >> tools/agent_guard.py",
        "rm tests/test_forbidden_vocabulary.py",
        "cp /tmp/x src/exercises/02-tokenization/web/tokenizer.json",
    ):
        payload = {"tool_name": "Bash", "cwd": str(tmp_path), "tool_input": {"command": command}}
        assert decide(payload, tmp_path, rules) is not None, command


def test_bash_that_only_reads_a_protected_path_is_allowed(rules, tmp_path) -> None:
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
        payload = {"tool_name": "Bash", "cwd": str(tmp_path), "tool_input": {"command": command}}
        assert decide(payload, tmp_path, rules) is None, command


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


# --- The three defects found on 2026-09-05, each with the twin that fails when the fix reverts ---


def _bash(command: str) -> dict:
    """A `PreToolUse` payload for a shell command."""
    return {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(REPO_ROOT)}


def test_mv_is_refused_where_cp_is_allowed(rules) -> None:
    """**`mv` destroys its source and was classified with `cp`, which does not.**

    `WRITES_LAST_ARGUMENT` flagged only the final argument, so `mv uv.lock /tmp/backup` read as a
    copy of a protected file and was allowed. The destination is outside the repository, so
    `bash_write_targets` then discarded it as "not this guard's business" — a protected file could
    be moved out from under the policy with nothing recorded anywhere.

    A logic defect, not a wiring one: it survives every fix to the hook matcher.
    """
    assert decide(_bash("mv uv.lock /tmp/backup"), REPO_ROOT, rules) is not None, (
        "mv moved a protected file out of the repository and the guard allowed it"
    )
    assert decide(_bash("cp uv.lock /tmp/backup"), REPO_ROOT, rules) is None, (
        "copying a protected file is legitimate and must stay allowed — that distinction is the "
        "entire reason the two command sets exist"
    )


def test_creating_a_branch_is_not_a_write_to_a_file_named_after_it(rules) -> None:
    """**Arming `Bash` would have blocked `git checkout -b` on the first scoped unit.**

    Every non-flag token after `checkout` counted as a written path, so `git checkout -b feature/x`
    registered a write to `feature/x`. Harmless while no `UNIT.md` existed — no pattern matches a
    branch name — and a total block on branch creation the moment one did, since anything outside
    the declared scope is refused. That is why this and the matcher fix had to land together.
    """
    for command in ("git checkout -b feature/x", "git switch -c feature/x", "git checkout main"):
        assert decide(_bash(command), REPO_ROOT, rules) is None, (
            f"{command!r} is branch work, not a write to the working tree"
        )


def test_checking_out_over_an_irreplaceable_file_is_still_refused(rules) -> None:
    """The other half: the fix above must not blind the guard to a real working-tree overwrite."""
    assert decide(_bash("git checkout -- notebooks/S10-training-loop.ipynb"), REPO_ROOT, rules), (
        "`git checkout -- <path>` overwrites the working tree and is the exact command AGENTS.md "
        "names as prohibited on these paths"
    )


@pytest.mark.parametrize(
    "command",
    [
        "echo x > notebooks/S10-training-loop.ipynb",
        "rm src/exercises/10-training-loop/tools/build_notebook.py",
        "mv notebooks/S10-training-loop.ipynb /tmp/nb",
    ],
)
def test_the_files_git_cannot_restore_are_refused(rules, tmp_path, command: str) -> None:
    """**The policy protected the backup tool and not one byte of what it exists to protect.**

    `tools/backup_local_only.py` and the tripwire were in `[guards]`; `notebooks/**` and
    `src/exercises/*/tools/build_notebook.py` were in no section at all. These are the files
    `AGENTS.md` calls the only ones in the repository with no second copy — gitignored, so
    `git checkout` cannot bring them back — and the backup store is a high-water mark that never
    removes, so it cannot undo an overwrite that was itself backed up.

    Refusing the write is the only moment at which the content still exists.

    **Run against an empty root, and asserting the REASON.** The first version of this test drove
    the real repository and only checked that *something* refused. It passed with the entire
    `[irreplaceable]` section deleted — because these paths were also outside the current unit's
    declared scope, so the scope rule refused them and the test could not tell the two apart. Green
    for the wrong reason is the failure this whole file is written against.
    """
    payload = {"tool_name": "Bash", "cwd": str(tmp_path), "tool_input": {"command": command}}
    refusal = decide(payload, tmp_path, rules)
    assert refusal is not None, f"{command!r} was allowed"
    assert "irreplaceable" in refusal, (
        f"{command!r} was refused, but not by the irreplaceable rule: {refusal!r}"
    )


def test_every_section_carrying_patterns_is_actually_enforced() -> None:
    """**The section list was hardcoded, so a new section protected nothing.**

    `_refuse` iterated a literal `("measured_data", "guards", "standards")`. Adding
    `[irreplaceable]` to the rules file would have read — in review, and in the file itself — as
    protection while enforcing nothing. This asserts the property rather than the tuple: every
    section that carries patterns is one the refusal path actually consults.
    """
    from agent_guard import _pattern_sections  # noqa: PLC0415

    rules = load_rules()
    declared = {
        name for name, body in rules.items() if isinstance(body, dict) and "patterns" in body
    }
    consulted = set(_pattern_sections(rules))
    assert consulted == declared, (
        f"sections carrying patterns but never consulted: {declared - consulted}"
    )
    assert "irreplaceable" in declared, "the irreplaceable section vanished from the policy"


def test_the_hook_matcher_offers_bash_to_the_guard() -> None:
    """**The Bash branch was written, correct, tested — and unreachable.**

    `re.search` never matched `Bash` against `"Write|Edit|NotebookEdit|MultiEdit"`, so no shell
    command was ever offered to `decide()`. Not theoretical: a pull request modified `uv.lock`,
    which the policy lists as protected, and nothing fired, because the change was made with
    `uv sync`.

    Asserts the property — that the matcher admits Bash — rather than the literal string, so the
    order can change freely.
    """
    import re  # noqa: PLC0415

    from install_agent_fleet import HOOK_WIRING  # noqa: PLC0415

    matcher = HOOK_WIRING["hooks"]["PreToolUse"][0]["matcher"]
    for tool in ("Bash", *sorted(WRITING_TOOLS)):
        assert re.search(matcher, tool), f"the PreToolUse matcher does not admit {tool!r}"


def _root_with_unit(tmp_path, body: str = "") -> Path:
    """A repo root carrying a UNIT.md, resolved so `relative_to` works on macOS.

    `tempfile` hands back `/var/...`, which resolves to `/private/var/...`; without `.resolve()`
    every path falls outside the root and `decide()` returns None for the wrong reason — a test
    that passes because the guard never saw the file.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path.resolve()
    (root / ".claude").mkdir(parents=True, exist_ok=True)
    (root / ".claude" / "UNIT.md").write_text(body, encoding="utf-8")
    return root


@pytest.mark.parametrize("note", ["TODO.md", "HANDOFF.md"])
def test_a_working_note_is_refused_by_default_and_allowed_when_named(tmp_path, rules, note) -> None:
    """**Protected, but openable — the distinction `[irreplaceable]` got wrong.**

    These two were in `[irreplaceable]`, which is a `NO_ESCAPE_HATCH` section. That is right for a
    notebook, whose legitimate rewrite happens through a builder this guard never inspects. It is
    wrong for a working note: rewriting them *is* their function, and freezing them meant the guard
    refused a rewrite that had been explicitly asked for, with no way to name past it.

    They still need protecting — neither is in git, and `HANDOFF.md` is not in the backup set
    either, so it exists in exactly one place on disk. So: refused by default, allowed when a unit
    names it, which makes the clobber deliberate rather than accidental.
    """
    bare = _root_with_unit(tmp_path / "bare")
    (tmp_path / "bare" / ".claude" / "UNIT.md").unlink()
    payload = {
        "tool_name": "Write",
        "cwd": str(bare),
        "tool_input": {"file_path": str(bare / note)},
    }
    refusal = decide(payload, bare, rules)
    assert refusal is not None, f"{note} must be refused when no unit declares it"
    assert "working_notes" in refusal, refusal

    named = _root_with_unit(tmp_path / "named", f"- scope: {note}\n{note}\n")
    allowed = {
        "tool_name": "Write",
        "cwd": str(named),
        "tool_input": {"file_path": str(named / note)},
    }
    assert decide(allowed, named, rules) is None, (
        f"naming {note} in UNIT.md must permit rewriting it"
    )


def test_naming_a_notebook_still_does_not_unlock_it(tmp_path, rules) -> None:
    """The twin, and the reason the two sections are separate.

    If naming a file in `UNIT.md` unlocked `[irreplaceable]` too, this change would have quietly
    removed the protection from the files this repository has actually lost — twice.
    """
    root = _root_with_unit(
        tmp_path,
        "- scope: notebooks/\nnotebooks/S10-training-loop.ipynb\n"
        "src/exercises/10-training-loop/tools/build_notebook.py\n",
    )
    for rel in (
        "notebooks/S10-training-loop.ipynb",
        "src/exercises/10-training-loop/tools/build_notebook.py",
    ):
        payload = {
            "tool_name": "Write",
            "cwd": str(root),
            "tool_input": {"file_path": str(root / rel)},
        }
        refusal = decide(payload, root, rules)
        assert refusal is not None, f"{rel} was unlocked by being named, which must never happen"
        assert "irreplaceable" in refusal, refusal
