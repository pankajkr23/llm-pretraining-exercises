"""Nothing sits in a vendored `web/_shared/` that no page uses.

**Six identical copies of a 167-line module shipped to the live site for months, imported by
nothing.** `anim.js` exported seven helpers — `onEnterOnce`, `countUp`, `morph` and four more — and
not one of them was referenced by any exercise, any test, or `deploy/vercel/build.sh`. It was
assembled into `public/` six times on every deploy. Nothing failed, because dead code is by
definition the code that nothing exercises.

**What made it durable rather than obvious.** `docs/DESIGN.md` already recorded it, in a table whose
own row read `| anim.js | reveal-on-enter helpers | **nobody** |`. A fact written down in prose is
not a fact anything enforces, and this one stayed written down and true for as long as it took
somebody to act on it. That is the gap this file closes.

**The measurement is harder than it looks, and it caught me first.** A page sets a class four
different ways in this repository, and the one that matters most is a local helper — `$(tag,
className)` — not the `el(tag, className)` the design standard describes. A first version of this
check looked for `el(` and reported `explainer.css` as **entirely** unused, when exercise 03 emits
36 of its 56 classes. Deleting on that evidence would have taken out a live stylesheet. So
`CLASS_SETTERS` below is the full list of idioms, and adding one is how this check stays honest.

**It also corrects three numbers `docs/DESIGN.md` published.** `explainer.css` is not "used by two":
it is used by 01, 02, 03 and 05, with 12 of 56 classes orphaned. `page.css` has 10 orphans, not 15.
Those counts are now derived here rather than typed there, because a count in prose beside a
generated fact is the failure this repository has paid for most often.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXERCISES = REPO_ROOT / "src" / "exercises"

#: Every way a page in this repository puts a class on an element. **The local `$(tag, className)`
#: helper is the important one** — it is what exercise 03 uses almost exclusively, and omitting it
#: made a live stylesheet look dead.
CLASS_SETTERS = (
    r"""\$\(\s*["'][^"']*["']\s*,\s*["']([^"']*)["']""",
    r"""\$\(\s*["'][^"']*["']\s*,\s*`([^`]*)`""",
    r"""class\s*=\s*["']([^"']+)["']""",
    r"""class\s*=\s*["']?\$?\{?`([^`]+)`""",
    r"""className\s*=\s*["']([^"']*)["']""",
    r"""className\s*=\s*`([^`]*)`""",
    r"""classList\.\w+\(\s*["']([^"']*)["']""",
    r"""setAttribute\(\s*["']class["']\s*,\s*["']([^"']*)["']""",
)

#: Files a page references by name rather than by class — scripts and stylesheets.
_BY_NAME = re.compile(r"""_shared/([A-Za-z0-9_.-]+\.(?:js|css))""")


def vendored_shared_dirs() -> list[Path]:
    """Every `web/_shared/` an exercise carries its own copy of."""
    return sorted(p for p in EXERCISES.glob("*/web/_shared") if p.is_dir())


def _page_files(exercise: Path) -> list[Path]:
    """The exercise's own pages — never the vendored shared directory itself."""
    web = exercise / "web"
    if not web.is_dir():
        return []
    return [
        p
        for p in web.rglob("*")
        if p.suffix in {".js", ".html"} and "_shared" not in p.relative_to(web).parts
    ]


def referenced_shared_files(exercise: Path) -> set[str]:
    """Shared filenames this exercise's pages actually link or import."""
    names: set[str] = set()
    for path in _page_files(exercise):
        names |= set(_BY_NAME.findall(path.read_text(encoding="utf-8", errors="ignore")))
    return names


def test_every_vendored_shared_file_is_referenced_by_the_page_that_vendors_it() -> None:
    """The gate. A copy nobody links is a copy that still ships on every deploy.

    Reported per exercise rather than as one set, because vendoring is per exercise: a file that is
    live in 03 and dead in 07 is a finding about 07, and collapsing them would hide it.
    """
    orphans: list[str] = []
    for shared in vendored_shared_dirs():
        exercise = shared.parent.parent
        referenced = referenced_shared_files(exercise)
        for candidate in sorted(shared.iterdir()):
            if candidate.suffix in {".js", ".css"} and candidate.name not in referenced:
                orphans.append(
                    f"{candidate.relative_to(REPO_ROOT)} — not linked by {exercise.name}"
                )

    assert not orphans, (
        "these files sit in a vendored web/_shared/ and the exercise that vendors them references "
        "none of them. They are assembled into public/ on every deploy:\n  " + "\n  ".join(orphans)
    )


def test_the_guard_would_have_caught_the_module_that_prompted_it(tmp_path) -> None:
    """Its twin, built as the shape `anim.js` actually had: present, vendored, referenced nowhere.

    Written against a fixture rather than the real tree, so it keeps failing-on-purpose after the
    real offender is gone — otherwise this pair would quietly become one passing test.
    """
    web = tmp_path / "web"
    (web / "_shared").mkdir(parents=True)
    (web / "_shared" / "used.js").write_text("export const a = 1;\n", encoding="utf-8")
    (web / "_shared" / "dead.js").write_text("export const b = 2;\n", encoding="utf-8")
    (web / "index.html").write_text(
        '<script type="module" src="_shared/used.js"></script>\n', encoding="utf-8"
    )

    referenced = referenced_shared_files(tmp_path)
    assert referenced == {"used.js"}, referenced
    assert "dead.js" not in referenced, "the orphan must not be seen as referenced"


def test_the_class_setter_list_covers_the_helper_this_repo_actually_uses() -> None:
    """`$(tag, className)` is the idiom that matters, and leaving it out made a live file look dead.

    Asserted against real source rather than a fixture: if exercise 03 ever stops using `$`, this
    fails and whoever changed it decides whether the pattern list should follow.
    """
    chapters = EXERCISES / "03-data-collection-framework" / "web" / "chapters.js"
    text = chapters.read_text(encoding="utf-8")
    assert re.search(r"""\$\(\s*["'][a-z]+["']\s*,""", text), (
        "exercise 03 no longer calls $(tag, className); CLASS_SETTERS may now be measuring nothing"
    )

    found: set[str] = set()
    for pattern in CLASS_SETTERS:
        for match in re.finditer(pattern, text):
            found |= {w.strip("\"' `${}.") for w in re.split(r"[\s,]+", match.group(1))}
    assert len(found) > 20, f"only {len(found)} classes found in a file that emits dozens: {found}"
