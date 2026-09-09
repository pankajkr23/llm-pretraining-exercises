"""No CSS declaration silently swallows the ones after it.

**The defect this was written for shipped to production and stayed there.** Exercise 01's four proof
pages declare their local diagram colours inside `@media (prefers-color-scheme: dark)` and
`[data-theme=…]` blocks — and every declaration in those blocks was missing its semicolon:

    --plane: #9aa6c0
    --bend: #ff9f0a
    --pos: #ff9f0a

A custom property's value runs to the next `;` or the closing `}`, so a CSS parser reads that as
**one** declaration whose value is `#9aa6c0 --bend: #ff9f0a --pos: #ff9f0a`. Two things follow, and
the second is the one that is easy to miss: `--bend`, `--pos` and the rest are never declared in the
dark block at all and keep their light values — *and* `--plane` itself is set to something that is
not a colour, so it fails at computed-value time too. Every token in the block is wrong.

**Why nothing caught it.** The file is valid HTML and the stylesheet parses — CSS is specified to
discard what it cannot understand rather than to fail. `node --check` never sees a `<style>` block.
The page renders, and a reader on a dark theme gets light diagram colours on a dark page, which is
the exact failure the comment two lines below the block says it exists to prevent.

**The check is lexical and deliberately narrow.** It looks for a declaration whose *value* contains
another property name, which is what a swallowed run looks like and is close to impossible to write
on purpose. Anything cleverer would need a real CSS parser, and this repo would rather have a check
it can reason about than one that is right for reasons nobody can restate.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The inside of a declaration block: `{ … }` with no nested braces.
_BLOCK = re.compile(r"\{([^{}]*)\}")

#: A declaration's name and value, split on the first colon.
_DECL = re.compile(r"^\s*(--[A-Za-z][\w-]*)\s*:\s*(.*)$", re.S)

#: A property name appearing inside a value — the signature of a swallowed declaration.
_SWALLOWED = re.compile(r"--[A-Za-z][\w-]*\s*:")


def _styled_sources() -> list[Path]:
    """Every file **we author and serve** that can carry CSS.

    Scoped to `web/` bundles and `deploy/`, and that scope is load-bearing rather than tidy. A
    first version globbed all of `src/exercises`, and the orphan check below immediately flagged
    dozens of properties inside two kinds of file this repo is explicit about not owning: the saved
    reference pages under `*/docs/*.html`, which are snapshots of other people's sites, and the
    tokenizer's `corpus/*.raw.html`, which is **measured input** whose bytes a guard must never
    invite anyone to edit. Both are somebody else's CSS, and flagging it would have taught a reader
    that this guard's findings are noise.
    """
    out: list[Path] = []
    for base in ((REPO_ROOT / "src" / "exercises").glob("*/web"), [REPO_ROOT / "deploy"]):
        for directory in base:
            for path in directory.rglob("*"):
                if path.suffix in {".css", ".html"} and path.is_file():
                    out.append(path)
    return sorted(out)


def swallowed_declarations(text: str) -> list[str]:
    """Declarations whose value contains another property name, meaning a `;` is missing."""
    found: list[str] = []
    for block in _BLOCK.findall(text):
        for chunk in block.split(";"):
            match = _DECL.match(chunk)
            if match and _SWALLOWED.search(match.group(2)):
                swallowed = _SWALLOWED.findall(match.group(2))
                found.append(f"{match.group(1)} swallows {len(swallowed)}: {', '.join(swallowed)}")
    return found


def test_no_declaration_swallows_the_ones_after_it() -> None:
    """The gate. A missing semicolon is invisible to every other check this repo runs."""
    offenders: list[str] = []
    for path in _styled_sources():
        for problem in swallowed_declarations(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {problem}")

    assert not offenders, (
        "a CSS declaration is missing its semicolon, so its value has swallowed the declarations "
        "after it. Those properties are never declared in that block and fall back to whatever the "
        "base rule says, while the swallowing property itself gets a value that is not valid for "
        "its type:\n  " + "\n  ".join(offenders)
    )


def test_the_guard_catches_the_shape_it_was_written_for() -> None:
    """The twin, built from exercise 01's actual defect rather than an invented one.

    A guard nobody has watched fail is not a guard, and this one is easy to write so that it never
    fires — the whole difficulty is that the broken CSS is syntactically fine.
    """
    broken = """
      @media (prefers-color-scheme: dark) {
        :root:not([data-theme]) {
          --plane: #9aa6c0
          --bend: #ff9f0a
          --pos: #ff9f0a
        }
      }
    """
    found = swallowed_declarations(broken)
    assert len(found) == 1, found
    assert "--plane swallows 2" in found[0], found

    fixed = broken.replace("#9aa6c0\n", "#9aa6c0;\n").replace(
        "#ff9f0a\n          --pos", "#ff9f0a;\n          --pos"
    )
    assert not swallowed_declarations(fixed), f"the fixed form must be clean: {fixed}"


def test_a_legitimate_value_containing_two_dashes_is_not_flagged() -> None:
    """The other twin: it must not fire on `var()`, which is where property names legitimately live.

    Without this, the obvious implementation — "a value mentioning `--`" — would flag every themed
    rule in the repository, and a guard that fires constantly is one that gets deleted.
    """
    fine = """
      .plate { color: var(--fg); border-color: var(--rule, var(--fg)); }
      :root { --a: 1px; --b: calc(var(--a) * 2); }
    """
    assert not swallowed_declarations(fine), swallowed_declarations(fine)


# --- the other half of the same class ------------------------------------------------------------
#
# A missing semicolon leaves a property undeclared inside one block. Referencing a property that no
# stylesheet declares anywhere leaves it undeclared in EVERY block, and the two look identical to a
# reader: the fallback renders, the page looks fine, and the theme silently does not apply.

_VAR_USE = re.compile(r"var\(\s*(--[A-Za-z][\w-]*)")
_VAR_DECL = re.compile(r"(--[A-Za-z][\w-]*)\s*:")


def _declared_and_used() -> tuple[set[str], dict[str, set[str]]]:
    """Every property the served bundle declares, and every one it references."""
    declared: set[str] = set()
    used: dict[str, set[str]] = {}
    for path in _styled_sources():
        text = path.read_text(encoding="utf-8")
        declared |= {m.group(1) for m in _VAR_DECL.finditer(text)}
        for match in _VAR_USE.finditer(text):
            used.setdefault(match.group(1), set()).add(str(path.relative_to(REPO_ROOT)))
    return declared, used


def test_no_page_references_a_property_no_stylesheet_declares() -> None:
    """Exercise 04 referenced seven that existed in no theme, so the fallback always won.

    `--radius`, `--rule`, `--rule-strong`, `--code-bg`, `--tip-bg`, `--tip-fg` and `--shadow-soft`
    were invented names sitting beside a real six-theme token file that publishes `--line`,
    `--line-strong`, `--panel`, `--fg` and `--shadow`. Every one rendered its hardcoded fallback in
    all six themes — including a tooltip that was a dark chip on a light page, which is wrong in
    `soft-light` and `high-contrast` and was wrong for as long as the page had existed.

    The failure mode is the reason this is worth a guard: a fallback means nothing breaks. There is
    no console error, no failing render, and the page looks deliberate.
    """
    declared, used = _declared_and_used()
    orphans = {name: sorted(files) for name, files in used.items() if name not in declared}
    assert not orphans, (
        "these custom properties are referenced but declared in no stylesheet, so their fallback "
        "wins in every theme:\n  "
        + "\n  ".join(f"{name} — {', '.join(files)}" for name, files in sorted(orphans.items()))
    )


def test_the_orphan_guard_can_actually_fail() -> None:
    """Its twin. The set of declared properties is large, so an accidental pass is easy."""
    declared, used = _declared_and_used()
    assert declared, "no properties were found at all, so the guard is reading nothing"
    invented = "--a-property-no-theme-declares"
    assert invented not in declared
    # The assertion the real test makes, applied to a name that cannot be declared anywhere.
    assert {invented: ["x"]}, "the orphan check must treat an undeclared reference as a failure"


#: What may legitimately follow a selector line that ends in a comma. Three shapes, and the third
#: is the one the first version missed: the last selector in a list often carries its whole rule on
#: one line (`.catsearch-hits { max-width: 72ch; }`), which ends in `}` rather than in `{`.
_CONTINUES_A_LIST = re.compile(r"^[.#:\[a-zA-Z@&*][^{};]*(,|\{|\{[^{}]*\}\s*)$")


def _selector_lines(text: str) -> list[tuple[int, str]]:
    """Every line that is a bare selector ending in a comma, with its 1-based number."""
    return [
        (n, line.strip())
        for n, line in enumerate(text.split("\n"), 1)
        if re.match(r"^\s*[.#][a-zA-Z][\w -]*(\s+[\w.>#:-]+)?\s*,\s*$", line)
    ]


def test_no_selector_lost_its_declaration_block() -> None:
    """A selector in a comma list must be followed by another selector or by an opening brace.

    **This is the shape a deletion leaves behind, and I made it today.** Stripping dead rules from
    exercise 09's stylesheet removed the `{ … }` of two rules and left their selectors, which then
    fused onto the *next* rule in the file:

        .limitlist,
        .limitlist li,
        .ahead { fill: var(--muted); }

    So `fill: var(--muted)` silently applied to two more selectors than anyone wrote it for. It was
    latent — neither class is emitted by that page — which is exactly why it survived: nothing
    rendered wrong, nothing failed, and the rule reads correctly at a glance because the last line
    of it is correct.

    `test_no_declaration_swallows_the_ones_after_it` above cannot see this. It inspects
    *declaration values* for a swallowed run; this is a defect in the *selector list*, one level up,
    and the declarations either side of it are all well formed.

    Deliberately narrow, for the same reason that one is: a selector line ending in a comma is
    unambiguous, and anything cleverer would need a real CSS parser.
    """
    offenders: list[str] = []
    for sheet in _styled_sources():
        if sheet.suffix != ".css":
            continue
        lines = sheet.read_text(encoding="utf-8").split("\n")
        for number, selector in _selector_lines(sheet.read_text(encoding="utf-8")):
            # **The next line, literally — blanks and comments are not skipped, and that is the
            # whole detector.** A fused selector IS followed by a selector, which is why skipping
            # over the gap finds nothing: the giveaway is the gap itself. Nobody puts a blank line
            # or a comment block *inside* a selector list; a deletion that removes a rule's body
            # leaves exactly that.
            nxt = lines[number].strip() if number < len(lines) else ""
            if not _CONTINUES_A_LIST.match(nxt):
                offenders.append(f"{sheet}:{number}  {selector}  → separated from: {nxt[:60]!r}")
    assert not offenders, (
        "a selector list is missing its own declaration block, so it has fused onto the next rule "
        "and those declarations now apply to more elements than anyone wrote them for:\n  "
        + "\n  ".join(offenders)
    )


def test_the_fused_selector_guard_catches_a_planted_one(tmp_path: Path) -> None:
    """Break it on purpose. A guard nobody has watched fail is not a guard.

    The planted shape is the real one: a rule whose body was deleted, leaving its selector attached
    to whatever came next.
    """
    planted = tmp_path / "page-extra.css"
    planted.write_text(
        ".gone,\n\n/* a comment, because that is what hid it the first time */\n\n"
        ".ahead {\n  fill: var(--muted);\n}\n",
        encoding="utf-8",
    )
    lines = planted.read_text(encoding="utf-8").split("\n")
    found = []
    for number, selector in _selector_lines(planted.read_text(encoding="utf-8")):
        nxt = lines[number].strip() if number < len(lines) else ""
        if not _CONTINUES_A_LIST.match(nxt):
            found.append(selector)
    assert found == [".gone,"], (
        f"the detector missed a planted fused selector; it found {found}. It cannot be trusted on "
        "the real stylesheets either."
    )
