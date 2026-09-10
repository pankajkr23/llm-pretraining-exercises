"""No deployed page carries a shell command. They live in the exercise's README.

**A page is read far more often than it is executed, and a command block is the part of a page that
nothing tests.** Rename a module, move a script, add a flag, and every guard stays green while the
page goes on inviting a reader to copy something that no longer works. The README's copy sits beside
the code it names, so the same rename breaks something visible.

Exercise 07 has enforced this on itself since its page was rebuilt. Promoting it found what a
single-exercise guard always finds: **exercise 05 carried eight commands and exercise 06 carried
six**, and neither was a duplicate of its README — each list had commands the other did not.

- 05's page listed three follow-on experiments (`mixture.repetition`, `.seam`, `.scale`) that
  appeared in **no** tracked file; its README listed `mixture.bench` and the integration suite,
  which the page omitted.
- 06's page listed `run_demo.py` and `verify.py` — the two commands the whole exercise turns on —
  and the README listed neither, so the exercise's most important instructions lived only in the
  one place nothing checks.

That is the failure mode stated exactly: two copies of a list, and the copy nobody runs from is the
one that rots. Everything missing was moved into the READMEs before the blocks came out, so the
guard cost no content.

**`docs/DESIGN.md` said the opposite for months** — "a `reproduce` section is mostly these" — while
seven of the nine spine-carrying pages had none at all and 07's guard forbade them outright. A
standard that contradicts a guard is a standard that loses; it now describes what the pages do,
which is to answer *can I believe this?* and point at the README for *what do I type?*.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Deliberately the same pattern exercise 07 has used since its rebuild. It is a *prefix* matcher
#: rather than a shell parser: `uv run`, `bash `, `pytest `, `pip install`, `npm run` and
#: `python -m` are how every command in this repository starts, and a page that wants to name one
#: without inviting a copy can talk about the module instead of the invocation.
COMMAND = re.compile(r"\b(?:uv run|bash |pytest |pip install|npm run|python -m)\b")


def _pages() -> list[Path]:
    """Every deployed page's source, from the filesystem.

    `deploy/vercel/build.sh` publishes any `src/exercises/*/web/`, so a new exercise is covered the
    day it lands. A guard holding its own list of pages reports green for every page it was not
    told about, which is the failure this file was promoted to fix.
    """
    return sorted(REPO_ROOT.glob("src/exercises/*/web/chapters.js"))


def _lines(source: Path) -> list[tuple[int, str]]:
    """Every line, with `${...}` substitutions stripped.

    A `${...}` is a value read from the results at render time, which is the correct thing to do;
    only the literal text around it is being judged.
    """
    return [
        (number, re.sub(r"\$\{[^}]*\}", "", line))
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1)
    ]


def test_some_page_exists() -> None:
    """Otherwise the sweep below is vacuous and passes in silence."""
    assert _pages(), "no deployed page source found — this whole file tests nothing"


def test_no_deployed_page_carries_a_shell_command() -> None:
    """One assertion over every page, because the report is a list and not a verdict.

    Reported together rather than one case per exercise on purpose: the value of this guard is
    seeing *which* pages drifted at once. Split per exercise, the first failure hides the rest
    behind it and the shape of the problem — two exercises, fourteen commands, none of them a clean
    duplicate — is only visible on the third run.
    """
    offenders: list[str] = []
    for page in _pages():
        slug = page.parents[1].name
        for number, line in _lines(page):
            if COMMAND.search(line):
                offenders.append(f"{slug}/web/chapters.js:{number}: {line.strip()[:80]!r}")

    assert not offenders, (
        f"{len(offenders)} shell command(s) on deployed pages. They belong in the exercise's "
        "README, beside the code they operate on — a page is read far more often than it is "
        "executed, so a command block on one is the copy that goes stale while every test stays "
        "green. Move it, do not delete it: check the README carries it first.\n  "
        + "\n  ".join(offenders)
    )


def test_the_guard_goes_red_on_a_planted_command(tmp_path: Path) -> None:
    """The twin. A guard nobody has watched fail is not a guard.

    Planted into a **copy** under `tmp_path` rather than into the real tree: `AGENTS.md` records an
    afternoon in which an agent mutated two guards in place to watch them go red, restored one, and
    committed the other — leaving two data-handling invariants returning "no findings" for four
    commits, which is indistinguishable from a clean run. Nothing here touches a tracked file, so
    there is nothing to restore and no `git add -A` can sweep a mutation into a commit.
    """
    planted = tmp_path / "chapters.js"
    planted.write_text(
        "const x = 1;\n"
        "para('run it with uv run pytest src/exercises/00-nothing');\n"
        "const y = `${M.value} is derived and must not trip this`;\n",
        encoding="utf-8",
    )
    hits = [number for number, line in _lines(planted) if COMMAND.search(line)]
    assert hits == [2], (
        f"the planted command was not found on line 2 alone (found {hits}). Either the pattern has "
        "stopped matching a real command, or the `${...}` stripping has started matching one."
    )
