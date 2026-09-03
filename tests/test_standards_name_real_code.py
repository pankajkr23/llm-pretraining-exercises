"""A standard that names a CSS class must name one that exists.

`docs/DESIGN.md` is the design standard, and it specifies the page by naming the classes that
implement it — `.plate`, `.preamble`, `.rail-inner`. That makes it the one document here whose
prose is partly *code*, and nothing was watching it.

**The defect this exists for.** A mechanical word-substitution sweep rewrote `` `.preamble` `` — a
CSS class name — into `` `.requirements document` ``. The class was renamed correctly in the
builder and the stylesheet, so the page kept working and the suite stayed green; only the standard
was wrong, and it was wrong in the specific way that matters, because the next person to implement a
plate would have gone looking for a class that does not exist. The same sweep turned the
`` `preamble()` `` builder into `` `requirements document()` `` in exercise 08's `CLAUDE.md`.

**Why this is more than one repair.** The standards are what an agent reads before touching a page,
and this repo is about to point a fleet of them at `docs/DESIGN.md`. A rename that updates the code
and forgets the document is the ordinary case, not the exotic one — `AGENTS.md` already records the
same shape twice, in `.rail-link.on` (styled for years, set by one exercise) and in the `_shared/`
rules that select elements no page emits.

**What the guard asserts, and what it deliberately does not.** The property is *"every class this
standard names resolves somewhere in the tracked front-end source"*. It is not *"the standard says
`.preamble`"* — naming one phrasing would fail every other correct one, which `AGENTS.md` calls out
as its own failure mode. It cannot tell you the standard *describes* the class correctly; only that
the class is real. That half stays a reading job.
"""

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The documents whose prose is partly code. Both are read by agents before they touch a page.
STANDARDS: tuple[str, ...] = ("docs/DESIGN.md", "AGENTS.md")

#: A bare CSS class: a dot, then an identifier, and nothing else.
_BARE_CLASS = re.compile(r"^\.[A-Za-z][A-Za-z0-9_-]*$")

#: Backtick spans are code spans; this pairs them left to right so a `foo`. `bar` sequence is read
#: as two spans rather than as one span starting at the closing backtick of the first.
_CODE_SPAN = re.compile(r"`([^`\n]+)`")

#: Dot-prefixed spans that are **not** selectors. Each is a real filename or directory, and they are
#: listed rather than pattern-matched because `.pre-commit-config.yaml` and `.rail-link.on` are
#: lexically the same shape — a dot, a name, a dot, a name — and only knowledge tells them apart.
NOT_SELECTORS: frozenset[str] = frozenset(
    {
        ".gitignore",
        ".gitleaksignore",
        ".pre-commit-config.yaml",
        ".claude",
        ".venv",
        ".env",
        ".mcp.json",
        ".worktreeinclude",
        ".quote-check-receipt.json",
    }
)

#: Classes named in a standard that legitimately resolve nowhere, with the reason each is allowed.
#: Fails in the other direction too (see the test below), so it cannot quietly absorb a real break.
EXEMPT: dict[str, str] = {}

_WEB_SUFFIXES = (".css", ".js", ".html")

#: A class is *declared* by being styled, or by being put on an element. Both halves are needed and
#: neither is optional: `.head` is styled only in an inline `<style>` block on the landing page, and
#: several classes are set by `el()` in a builder and styled in a file the guard reads separately.
_STYLE_BLOCK = re.compile(r"<style[^>]*>(.*?)</style>", re.DOTALL | re.IGNORECASE)
_SELECTOR = re.compile(r"\.([A-Za-z][A-Za-z0-9_-]*)")
_CLASS_SETTERS = (
    re.compile(r"""\bel\(\s*['"][^'"]*['"]\s*,\s*['"]([^'"]*)['"]"""),  # el('div', 'plate wide')
    re.compile(r"""\bclass\s*=\s*["']([^"']*)["']"""),  # class="plate wide"
    re.compile(r"""\bclassName\s*=\s*['"]([^'"]*)['"]"""),
    re.compile(r"""\bclassList\.\w+\(\s*['"]([^'"]*)['"]"""),
)


def declared_classes(sources: dict[str, str]) -> set[str]:
    """Every class the front end actually defines, as `{name}` without the leading dot.

    **Never matched against bare prose.** The first version of this guard asked only whether the
    name appeared inside any quoted string, and JavaScript in this repo is full of narrative strings
    — so `.requirements` "resolved" against a sentence containing the word *requirements*, and the
    guard passed on the exact defect it was written for. It was caught by breaking `DESIGN.md` on
    purpose and watching the guard stay green, which is the only reason it is not still wrong.
    """
    names: set[str] = set()
    for rel, text in sources.items():
        if rel.endswith(".css"):
            names.update(_SELECTOR.findall(text))
        elif rel.endswith(".html"):
            for block in _STYLE_BLOCK.findall(text):
                names.update(_SELECTOR.findall(block))
        for setter in _CLASS_SETTERS:
            for value in setter.findall(text):
                names.update(value.split())
    return names


def _tracked_web_source() -> dict[str, str]:
    """Every tracked stylesheet, script and page, as `{path: text}`.

    Tracked rather than globbed: a class that exists only in an untracked working file is a class a
    fresh clone does not have, and the standard would be describing something nobody else can see.
    """
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"], capture_output=True, text=True, check=False
    ).stdout.split()
    return {
        rel: (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        for rel in listed
        if rel.endswith(_WEB_SUFFIXES)
    }


def classes_named_in(text: str) -> list[tuple[str, str]]:
    """Every CSS class a document names, as `(class, the span it came from)`.

    The span is carried through because it is what makes a failure readable: `.requirements` on its
    own says nothing, while the span `.requirements document` says immediately what happened to it.
    """
    found: list[tuple[str, str]] = []
    for span in _CODE_SPAN.findall(text):
        stripped = span.strip()
        if stripped in NOT_SELECTORS:
            continue
        # A path, a rule body, or a call — none of these is a bare selector reference.
        if "/" in stripped or "{" in stripped or "(" in stripped:
            continue
        for part in re.split(r"[\s>+~,]+", stripped):
            if not part.startswith(".") or part == ".":
                continue
            if part in NOT_SELECTORS:
                continue
            # `.a.b` is two chained classes. A trailing `*` is prose shorthand for a family
            # (`.fig-*`) and is kept as-is, so the family can be checked for members rather than
            # skipped — a family whose last member was deleted is exactly this guard's business.
            for name in part.split("."):
                if name:
                    found.append((f".{name}", span))
    return found


def unresolved_in(text: str, declared: set[str]) -> list[tuple[str, str]]:
    """The classes a document names that the front end never declares."""
    out = []
    for cls, span in classes_named_in(text):
        if cls in EXEMPT:
            continue
        if cls.endswith("*"):  # a family: satisfied by any one member
            prefix = cls[1:-1]
            if prefix and not any(d.startswith(prefix) for d in declared):
                out.append((cls, span))
            continue
        if not _BARE_CLASS.match(cls) or cls[1:] not in declared:
            out.append((cls, span))
    return out


def test_every_class_named_in_a_standard_exists_in_the_tracked_source() -> None:
    """The gate. A standard specifying a class nobody implements specifies nothing."""
    declared = declared_classes(_tracked_web_source())
    broken: list[str] = []
    for rel in STANDARDS:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for cls, span in unresolved_in(text, declared):
            broken.append(f"{rel}: `{span}` names {cls}, which is styled and set nowhere")
    assert not broken, (
        f"{len(broken)} class reference(s) in a standard resolve to nothing. Either the class was "
        "renamed and the standard was not updated, or a word-substitution edit corrupted the "
        "identifier — check the span, not just the name:\n  " + "\n  ".join(broken)
    )


def test_the_guard_catches_a_class_that_no_longer_exists() -> None:
    """The twin, for the ordinary case: a rename that updated the code and forgot the document."""
    planted = "- **`.plate-legend`** carries the key beneath the figure.\n"
    declared = declared_classes({"a.css": ".plate { display: grid }\n.preamble { margin: 0 }\n"})
    assert unresolved_in(planted, declared) == [(".plate-legend", ".plate-legend")]


def test_the_guard_catches_an_identifier_a_word_sweep_corrupted() -> None:
    """The twin for the defect that actually happened, in its actual shape.

    A sweep replaced a banned word inside a code span, producing a two-word "class". Splitting the
    span on whitespace is what catches it: the surviving dotted piece resolves to nothing, and the
    span is reported alongside so the cause is legible rather than inferred.
    """
    corrupted = "- **`.requirements document`** is the orientation before the figure.\n"
    declared = declared_classes(
        {
            "p.css": ".preamble { margin: 0 }\n.preamble-lab { font: monospace }\n",
            # Narrative prose in a real builder — the exact thing that made the first version of
            # this guard pass on this very defect. It must not count as a declaration.
            "c.js": "const t = 'The requirements say the five problems are separate';\n",
        }
    )
    found = unresolved_in(corrupted, declared)
    assert found == [(".requirements", ".requirements document")], found


def test_a_path_or_a_rule_body_is_not_read_as_a_selector() -> None:
    """False positives here would push someone to reword correct prose to satisfy the test.

    `AGENTS.md` names dotfiles and quotes whole CSS declarations constantly; both start with a dot
    and neither is a class reference.
    """
    declared = declared_classes({"a.css": ".wrap { padding-left: 260px }\n"})
    for benign in (
        "See `.pre-commit-config.yaml` for the gates.",
        "Run `uv run ruff format .` before committing.",
        "`_shared/page.css` sets `.wrap { padding-left: 260px }` unconditionally.",
        "Vendored at `./_shared/tokens.css`, which is not the token file.",
    ):
        assert unresolved_in(benign, declared) == [], benign


def test_a_class_styled_only_in_an_inline_style_block_still_resolves() -> None:
    """The landing page styles `.head` inside `<style>`, not in any `.css` file.

    Reading only `*.css` reported it as missing — a false positive on correct work, which is the
    failure mode that gets a guard reworded until it stops catching anything.
    """
    page = "<html><style>.head { max-width: 54ch }</style><div class='head x'></div></html>"
    declared = declared_classes({"index.html": page})
    assert unresolved_in("The prose column is `.head`.", declared) == []
    assert "x" in declared, "a class set via a class attribute is declared too"


def test_every_exemption_names_a_class_a_standard_actually_uses() -> None:
    """Fails in the other direction, so EXEMPT cannot become the easy way to silence a red gate.

    An exemption for a class no standard names is an exemption nobody is watching, and the next
    person reading the list cannot tell which entries are load-bearing.
    """
    named = set()
    for rel in STANDARDS:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        named.update(cls for cls, _ in classes_named_in(text))
    stale = sorted(set(EXEMPT) - named)
    assert not stale, f"EXEMPT names classes no standard mentions, so they exempt nothing: {stale}"


def test_a_family_reference_needs_at_least_one_member() -> None:
    """`.fig-*` is prose shorthand for a family, and is checked rather than skipped.

    Skipping wildcards would leave a hole the width of every family reference: a family whose last
    member was renamed away would keep reading as specified. It resolves on any one member.
    """
    declared = declared_classes({"a.css": ".fig-caption { margin: 0 }\n"})
    assert unresolved_in("Figures are keyed `.fig-*`.", declared) == []
    assert unresolved_in("Bays are keyed `.bay-*`.", declared) == [(".bay-*", ".bay-*")]
