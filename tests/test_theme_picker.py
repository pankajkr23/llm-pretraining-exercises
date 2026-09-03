"""One theme picker, not eight — and one rule inside it that fails silently when it drifts.

Every deployable page offers the same control over the same six themes. Eight of them had
hand-written the same eighteen lines: the storage key, the read, the `system`-removes-the-attribute
rule, and two `try`/`except` blocks around storage. That is eight places for one rule to drift.

**And the rule that would drift is the dangerous one.** `system` must *remove* `data-theme`, not set
it to the string `"system"`. The token file scopes its `prefers-color-scheme` block to
`:root:not([data-theme="light"])` and friends, so a page that sets the attribute instead stops
following the operating system — on that page only, for readers who use the system setting, in a way
no screenshot at a fixed theme would ever show.

This file is lexical on purpose. It reads the pages themselves rather than a rendered DOM, so it
runs in the ordinary CI job with no browser: a structural rule that only holds when chromium happens
to be installed is one that can silently stop running, and this repository has already lost 46 tests
exactly that way.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED = REPO_ROOT / "deploy" / "vercel" / "_shared" / "theme.js"

#: Every page that ships a theme control, found rather than listed — a hand-maintained list would
#: go stale the first time an exercise was added, which is how this class of guard usually fails.
PAGES = sorted(REPO_ROOT.glob("src/exercises/*/web/*.html")) + [
    REPO_ROOT / "deploy" / "vercel" / "index.html"
]

#: The one storage key for the whole site, so a choice made on the landing page survives into an
#: exercise and back.
THEME_KEY = "era5-theme"


def _pages_with_a_picker() -> list[Path]:
    return [p for p in PAGES if 'id="theme"' in p.read_text(encoding="utf-8")]


def test_the_shared_module_is_the_only_place_that_writes_the_storage_key() -> None:
    """The whole point. One implementation, and every page reaches for it.

    A page may *read* the key in its pre-paint `<head>` script — that has to be inline, because a
    module loads too late and the page would paint system colours for a frame and then repaint,
    which is the flash every theme switcher is judged by. What no page may do is write it.
    """
    offenders: list[str] = []
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        if f"setItem('{THEME_KEY}'" in text or f'setItem("{THEME_KEY}"' in text:
            offenders.append(str(page.relative_to(REPO_ROOT)))

    assert not offenders, (
        "these pages write the theme key themselves instead of using `_shared/theme.js`:\n  "
        + "\n  ".join(offenders)
        + "\n\nEight pages once carried their own copy of this logic. Import `bindThemePicker` "
        "(when the page renders its own <select>) or `mountThemePicker` (when it does not)."
    )


def test_every_page_with_a_picker_wires_it_through_the_shared_module() -> None:
    """A rendered `<select>` nobody wired is a control that silently does nothing.

    **The first version of this test was blind, and watching it fail is how I found out.** It
    checked `"bindThemePicker" not in text` — so deleting the *call* and leaving the `import` line
    satisfied it perfectly. An unused import is exactly what a page has after someone removes the
    one line that mattered.

    So the import lines are stripped before looking, and what is asserted is a **call**.
    """
    unwired: list[str] = []
    for page in _pages_with_a_picker():
        text = page.read_text(encoding="utf-8")
        # Strip the import statements: naming a function is not calling it.
        body = re.sub(r"^\s*import\s+\{[^}]*\}\s+from\s+'[^']*';\s*$", "", text, flags=re.M)
        if "bindThemePicker(" not in body and "mountThemePicker(" not in body:
            unwired.append(str(page.relative_to(REPO_ROOT)))

    assert not unwired, (
        "these pages render a theme control and never CALL anything to wire it:\n  "
        + "\n  ".join(unwired)
        + "\n\nAn import alone does not wire a control, and an unused import is what a page has "
        "after someone deletes the one line that mattered."
    )


def test_the_shared_module_removes_the_attribute_for_system() -> None:
    """The rule that fails silently, asserted where it lives.

    Setting `data-theme="system"` rather than removing the attribute breaks
    `prefers-color-scheme` — and it breaks it only for readers on the system setting, only on that
    page, and never in a screenshot taken at a fixed theme.
    """
    source = SHARED.read_text(encoding="utf-8")
    apply_body = source[source.index("export function applyTheme") :]
    apply_body = apply_body[: apply_body.index("\n}")]

    assert "removeAttribute" in apply_body, (
        "applyTheme no longer removes data-theme for 'system'. Setting it to the string 'system' "
        "leaves the attribute present, and every prefers-color-scheme rule is scoped to its "
        "absence — so the page stops following the operating system, silently."
    )


def test_every_picker_offers_the_same_themes() -> None:
    """Six themes everywhere. A page missing one is a page some readers cannot use.

    `docs/DESIGN.md`: a page styled for two of the six is unreadable in the other four.
    """
    source = SHARED.read_text(encoding="utf-8")
    themes = source[source.index("export const THEMES") :]
    canonical = re.findall(r"\['([a-z-]+)', '[^']+'\]", themes)
    assert len(canonical) >= 5, f"the shared THEMES list looks wrong: {canonical}"

    wrong: list[str] = []
    for page in _pages_with_a_picker():
        text = page.read_text(encoding="utf-8")
        offered = re.findall(r'<option value="([a-z-]+)"', text)
        if not offered:
            continue  # built in JS from THEMES, so it cannot disagree
        if offered != canonical:
            wrong.append(f"{page.relative_to(REPO_ROOT)}: {offered}")

    assert not wrong, (
        f"these pages offer a different set of themes than the shared list {canonical}:\n  "
        + "\n  ".join(wrong)
    )
