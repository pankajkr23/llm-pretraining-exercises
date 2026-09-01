"""Every published explainer tells the same story in the same order — checked repo-wide.

`AGENTS.md` requires each deployable page to carry a narrative spine, declared as `data-role` on
each section so a test can check the *structure* while the prose stays free:

    thesis · glossary · problem · mechanism · method · expected
    results · negatives · conclusion · limits · next · reproduce

The standard was written after exercise 07 shipped a page that was nine tables, one button and no
diagram — correct, and unreadable. PK caught it by reading it; no test did. Exercise 07's own render
test now checks its spine in the browser, but **that only protects exercise 07**: an exercise added
later would inherit the convention in prose and nothing would fail. This file is the repo-wide half.

**Why this guard is lexical rather than a browser test.** It reads the filesystem and the source of
each `chapters.js`, which are facts about the *repository*, not about whatever happens to be
installed. A browser check needs playwright and an assembled site, so it lives behind an
`importorskip` and an integration marker — and this repo has already lost 46 tests to a gated file
that ran nowhere while CI stayed green. A structural rule that only runs when chromium is present is
a structural rule that can silently stop running. This one runs in the plain `test` job, always.

**What it therefore cannot see, stated plainly.** Source order is not DOM order — a page assembles
its sections in `buildPage`, so a role declared here could still render in the wrong place, or not
at all. That half is the per-exercise browser test's job, and
`test_an_enforced_page_also_checks_its_spine_in_a_browser` asserts every enforced exercise has one,
so the two halves cannot drift apart. Presence is checked here; order and rendering are checked
there.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXERCISES = REPO / "src" / "exercises"

#: The spine, in the order a reader meets it. Keep in step with `AGENTS.md`.
SPINE: tuple[str, ...] = (
    "thesis",
    "glossary",
    "problem",
    "mechanism",
    "method",
    "expected",
    "results",
    "negatives",
    "conclusion",
    "limits",
    "next",
    "reproduce",
)

#: Exercises whose page is held to the spine. Adding an exercise here is the deliberate act of
#: adopting the standard; the ledger below is what stops a new one skipping it by accident.
SPINE_ENFORCED: frozenset[str] = frozenset(
    {
        "05-datamixtures-and-curriculum",
        "06-build-training-dataset",
        "07-model-embeddings-internals",
    }
)

#: Exercises with a deployable `web/` that are deliberately NOT held to the spine, each with the
#: reason. These predate the standard and are not being rewritten to it: the spine describes an
#: exercise that ran an experiment and reports a result, and none of these do.
#:
#: A reason is required rather than a bare name so that "we never got to it" and "this shape does
#: not apply" cannot be confused when someone reads this list in a year.
SPINE_EXEMPT: dict[str, str] = {
    "01-introductions": (
        "four hand-written proof pages, no experiment and no result to report; its notebook embeds "
        "the shipped pages rather than re-implementing them"
    ),
    "02-tokenization": ("a tokenizer widget and a how-it-works explainer, not a write-up of a run"),
    "03-data-collection-framework": (
        "a chaptered framework tour; predates the spine and reports invariants rather than an "
        "experiment"
    ),
    "04-data-cleaning-dedup": (
        "a chaptered pipeline tour; predates the spine and reports invariants rather than an "
        "experiment"
    ),
}

#: How a page is allowed to declare a section's role. Both forms name the role at the point the
#: section is actually constructed, so a role cannot be claimed without building a section for it —
#: which is the property that keeps this guard able to fail.
#:
#:   section('id', 'role', ...)     the helper exercise 07 introduced
#:   node.dataset.role = 'role'     a direct assignment
_ROLE_DECLARATION = re.compile(
    r"""\bsection\(\s*['"][\w-]+['"]\s*,\s*['"]([a-z]+)['"]"""
    r"""|\.dataset\.role\s*=\s*['"]([a-z]+)['"]""",
)


def _roles_declared(source: str) -> set[str]:
    """Every role this page constructs a section for."""
    return {a or b for a, b in _ROLE_DECLARATION.findall(source)}


def _chapters(slug: str) -> Path:
    return EXERCISES / slug / "web" / "chapters.js"


def _deployable() -> set[str]:
    """Every exercise `deploy/vercel/build.sh` would publish — read from the filesystem.

    Derived rather than listed, because the build script globs `src/exercises/*/web/` and an
    exercise becomes deployable the moment that directory exists. A hand-kept copy of this list
    would be exactly the thing that goes stale.
    """
    return {p.parent.name for p in EXERCISES.glob("*/web") if p.is_dir()}


def test_every_deployable_exercise_is_either_enforced_or_exempt():
    """The ledger fails in both directions, which is the point.

    A new exercise ships a `web/` bundle and is in neither set, so this goes red and somebody has to
    decide. Without it the spine is a rule that applies to whoever remembers to apply it.
    """
    deployable = _deployable()
    classified = SPINE_ENFORCED | set(SPINE_EXEMPT)

    unclassified = sorted(deployable - classified)
    assert not unclassified, (
        f"these exercises publish a page but are in neither ledger: {unclassified}. Add each to "
        f"SPINE_ENFORCED (and give it the spine) or to SPINE_EXEMPT with a reason."
    )

    phantom = sorted(classified - deployable)
    assert not phantom, f"these are in a ledger but publish no web/ bundle: {phantom}"

    both = sorted(SPINE_ENFORCED & set(SPINE_EXEMPT))
    assert not both, f"these are both enforced and exempt: {both}"


def test_every_exempt_exercise_gives_a_reason():
    """A bare exemption list decays into "we never got round to it" with no way to tell."""
    for slug, reason in SPINE_EXEMPT.items():
        assert len(reason.split()) >= 6, f"{slug}'s exemption reason is too thin to be useful"


def test_enforced_pages_declare_every_part_of_the_spine():
    """The rule itself: an enforced page constructs a section for each role."""
    for slug in sorted(SPINE_ENFORCED):
        source = _chapters(slug)
        assert source.exists(), f"{slug} is enforced but has no web/chapters.js"
        declared = _roles_declared(source.read_text(encoding="utf-8"))
        missing = [r for r in SPINE if r not in declared]
        assert not missing, (
            f"{slug}'s page is missing these parts of the story: {missing}. Every reader arriving "
            f"cold needs all of them; see AGENTS.md."
        )


def test_enforced_pages_declare_no_unknown_roles():
    """A typo'd role satisfies nothing and silently drops a section out of the checked spine."""
    for slug in sorted(SPINE_ENFORCED):
        declared = _roles_declared(_chapters(slug).read_text(encoding="utf-8"))
        unknown = sorted(declared - set(SPINE))
        assert not unknown, f"{slug} declares roles that are not in the spine: {unknown}"


def test_an_enforced_page_also_checks_its_spine_in_a_browser():
    """This file checks presence; only a rendered check can see order. Both must exist.

    Named explicitly because the lexical guard is the one that always runs, and it would be easy to
    conclude it is sufficient. It is not: it cannot tell `buildPage` assembled the sections in the
    right order, or at all.
    """
    for slug in sorted(SPINE_ENFORCED):
        tests = sorted((EXERCISES / slug / "tests").glob("*render*.py"))
        assert tests, f"{slug} is enforced but has no render test to check the rendered spine"
        checked = [
            t
            for t in tests
            if "dataset.role" in t.read_text(encoding="utf-8")
            or "data-role" in t.read_text(encoding="utf-8")
        ]
        assert checked, (
            f"{slug} has render tests but none reads the sections' roles, so nothing checks that "
            f"the spine actually renders in order: {[t.name for t in tests]}"
        )


def test_the_spine_here_matches_the_one_agents_md_documents():
    """Prose that states a list goes stale while the code beside it stays right.

    This repo's most expensive recurring failure. `AGENTS.md` is where a contributor reads the
    spine, so the two must not be able to disagree.
    """
    doc = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    for role in SPINE:
        assert re.search(rf"\b{role}\b", doc), f"AGENTS.md never mentions the `{role}` section"


def test_the_role_extractor_actually_finds_roles_and_misses_absent_ones():
    """Break it on purpose. A guard nobody has watched fail is not a guard.

    Every invariant in this repo is written twice — once against the real spine, once against a
    deliberately broken fixture. Without this, a regex that silently matched nothing would make
    every assertion above pass vacuously.
    """
    good = """
      section('thesis', 'thesis', 'Eyebrow', 'Title', []);
      const s = el('section'); s.dataset.role = 'glossary';
    """
    assert _roles_declared(good) == {"thesis", "glossary"}

    # The failure this guard exists for: a section built without declaring a role.
    silent = "const s = el('section'); s.id = 'limits';"
    assert _roles_declared(silent) == set()

    # And the vacuous-pass failure: if the regex matched nothing, `missing` would be everything.
    assert [r for r in SPINE if r not in _roles_declared(silent)] == list(SPINE)
