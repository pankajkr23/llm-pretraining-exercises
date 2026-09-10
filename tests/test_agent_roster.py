"""Every persona in the fleet is read-only, well-formed, and named where the fleet is described.

`tools/install_agent_fleet.py` copies `docs/agents/reviewers/*.md` into `.claude/agents/` by glob,
so **adding a persona is adding a file** — nothing validates it, and `.claude/` is gitignored, which
means a malformed or over-privileged persona would be invisible to review, to CI and to every other
clone. The installer already has a guard for the copies drifting from their source; it has none for
the source itself.

Two properties here, and the read-only one lives next door. `tests/test_agent_guard.py` has
asserted since the fleet landed that no persona declares a writing tool; duplicating it here would
be a second copy of one rule, and the second copy is the one that drifts. What that file did **not**
have is anything about the frontmatter the installer needs, or about the roster and the document
agreeing.

**A persona the documentation never mentions is a persona nobody invokes**, and
`docs/AGENT_FLEET.md` says the converse itself: *"a document that names a file the machinery
ignores is worse than one that stays silent."* Both directions, therefore — a file with no
mention, and a mention with no file.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEWERS = REPO_ROOT / "docs" / "agents" / "reviewers"
FLEET_DOC = REPO_ROOT / "docs" / "AGENT_FLEET.md"

#: The keys every persona must declare. `model` is included because the default is not stated
#: anywhere, so a persona without it silently inherits whatever the caller happens to be.
REQUIRED_KEYS = ("name", "description", "tools", "model")


def _personas() -> list[Path]:
    return sorted(REVIEWERS.glob("*.md"))


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert match, f"{path.name} has no YAML frontmatter block"
    out: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out


def test_there_are_personas_to_check() -> None:
    """Otherwise every case below passes by having nothing to look at."""
    found = _personas()
    assert len(found) >= 4, f"only {len(found)} persona(s) in {REVIEWERS}; the glob has rotted"


def test_every_persona_declares_what_the_installer_copies() -> None:
    """Name, description, tools, model — and the name matches the file it is in.

    The installer copies by filename and the runtime dispatches by `name`, so a mismatch produces a
    persona that installs under one identity and answers to another.
    """
    problems = []
    for path in _personas():
        front = _frontmatter(path)
        for key in REQUIRED_KEYS:
            if not front.get(key):
                problems.append(f"{path.name}: no `{key}:`")
        if front.get("name") and front["name"] != path.stem:
            problems.append(f"{path.name}: declares `name: {front['name']}`")
    assert not problems, "persona frontmatter is incomplete:\n  " + "\n  ".join(problems)


def test_the_fleet_document_and_the_roster_agree() -> None:
    """Both directions: a persona nobody documented, and a documented persona that does not exist.

    The document is where someone decides which persona to invoke, so a file it never mentions is a
    file nobody runs — and `docs/AGENT_FLEET.md` makes the opposite argument about itself in its own
    words: *"a document that names a file the machinery ignores is worse than one that stays
    silent."*
    """
    doc = FLEET_DOC.read_text(encoding="utf-8")
    names = {path.stem for path in _personas()}

    undocumented = sorted(n for n in names if f"`{n}`" not in doc)
    assert not undocumented, (
        f"these personas exist and {FLEET_DOC.name} never names them: {undocumented}. "
        "Add them where the roster is described, or they are files nobody will invoke."
    )

    # The other direction, restricted to the roster table so ordinary prose is not scanned for
    # every English word that happens to match a persona name.
    claimed = set(re.findall(r"^\| `([a-z][a-z-]*)` \|", doc, re.M))
    missing = sorted(claimed - names)
    assert not missing, (
        f"{FLEET_DOC.name} names these personas in its roster table and no file defines them: "
        f"{missing}. The installer copies by glob, so a documented persona with no file is one "
        "that silently does not exist."
    )
