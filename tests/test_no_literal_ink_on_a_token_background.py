"""No rule paints a hardcoded white or black on a background that comes from a theme token.

**The literal cannot know what it is sitting on.** A theme token is a different colour in each of
the six themes by definition — that is what makes it a token — so a rule reading
`background: var(--accent); color: #fff` is asserting something true in the themes where `--accent`
is dark and false in the ones where it is bright. It is not a near miss either. Measured on the
shipped palette, white over `--accent` runs:

| theme | `--accent` | white on it | `--on-accent` on it |
| --- | --- | --- | --- |
| light, soft-light, high-contrast | dark blues | 5.38 - 11.22:1 | identical: the token *is* white |
| system dark | `#2997ff` | **3.02:1** | 6.25:1 |
| tinted-dark | `#5aa9ff` | **2.46:1** | 7.77:1 |
| neon | `#00e5ff` | **1.54:1** | 12.19:1 |

At 1.54:1 the text is not hard to read, it is gone. Every theme already ships the paired token that
answers this — `--on-accent`, near-white on the light themes and near-black on the dark ones — so
the fix is never to pick a better literal, it is to stop picking.

**Lexical on purpose, and that is the stronger check here rather than the weaker one.** The rule
this was written for is `.back:hover`, and a hover state is one no static render ever enters: a
browser test would have to know to hover the pill on all six pages in all six themes before it could
see anything. The source says it unconditionally.

**What it cannot see.** A background set from JavaScript — `style="background:${colOf(w)}"` — is not
in any rule, so a literal `color` paired with it is invisible here. Exercise 01's chip palette is
exactly that shape and is checked where it can be: in the browser, against what was painted.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A literal that names white or black outright. `transparent`, `currentColor` and any `var()` are
#: fine by construction — they are all deferrals to something that does know the theme.
_LITERAL_INK = re.compile(
    r"^\s*color\s*:\s*(#fff(?:fff)?|#000(?:000)?|white|black)\s*(?:!important)?\s*$",
    re.I,
)

#: A background whose value is a custom property, i.e. one that changes with the theme.
_TOKEN_BACKGROUND = re.compile(r"^\s*background(?:-color)?\s*:.*\bvar\(\s*--", re.I)


def _stylesheets() -> list[Path]:
    """Every tracked stylesheet and page under a deployable bundle or the shared layer."""
    out = subprocess.run(
        ["git", "ls-files", "src/exercises/*/web/*", "deploy/vercel/*"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in out.stdout.split("\n") if line.endswith((".css", ".html"))]


def _offending_rules(text: str) -> list[str]:
    """Every rule block declaring both a token background and a literal ink.

    Deliberately naive about CSS: it splits on braces and looks at each block's own declarations,
    which is exactly the granularity the property needs. Nesting would make a block's opening line
    wrong; the declarations inside it stay right, and those are what is being read.
    """
    found = []
    for block in re.split(r"\}", text):
        head, _, body = block.rpartition("{")
        if not head:
            continue
        declarations = body.split(";")
        has_token_bg = any(_TOKEN_BACKGROUND.match(d) for d in declarations)
        inks = [d.strip() for d in declarations if _LITERAL_INK.match(d)]
        if has_token_bg and inks:
            selector = " ".join(head.strip().split("\n")[-1].split())
            found.append(f"{selector} {{ … {inks[0]} … }}")
    return found


@pytest.mark.parametrize("path", _stylesheets(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_rule_paints_a_literal_ink_on_a_token_background(path: Path) -> None:
    """One case per file, so a failure names the file rather than the repository."""
    offenders = _offending_rules(path.read_text())
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} paints a hardcoded ink on a background that changes with "
        f"the theme:\n  " + "\n  ".join(offenders) + "\n\n"
        "Use the paired token — `--on-accent` — rather than a better literal. There is no literal "
        "that is correct on both a near-black and a near-white ground, which is what a token "
        "background means."
    )


def test_the_check_can_actually_fail() -> None:
    """Break it on purpose, on a fixture rather than on a real file.

    `AGENTS.md`: a guard nobody has watched fail is not a guard — and the backup for a deliberate
    break must never live in the working tree, because that is a file `git add -A` will commit.
    A string fixture cannot be left behind by an early return at all.
    """
    broken = """
    .back:hover {
      background: var(--accent);
      color: #fff;
    }
    """
    assert _offending_rules(broken), (
        "the exact rule this file was written for was not reported, so the check is decorative"
    )

    # The three shapes that must NOT be reported, or the guard fails correct work.
    for allowed, why in [
        (".x { background: var(--accent); color: var(--on-accent); }", "the paired token"),
        (".x { background: #0068d1; color: #fff; }", "a literal ground, so the ink can be one too"),
        (".x { background: var(--panel); color: currentColor; }", "a deferral, not a literal"),
    ]:
        assert not _offending_rules(allowed), f"reported a rule that is correct: {why}"


def test_more_than_a_handful_of_files_are_actually_read() -> None:
    """A parametrised guard over an empty set is green and worthless."""
    found = _stylesheets()
    assert len(found) >= 20, (
        f"only {len(found)} stylesheet(s) were found, so the check above is passing over almost "
        "nothing. `git ls-files` has probably stopped matching."
    )
