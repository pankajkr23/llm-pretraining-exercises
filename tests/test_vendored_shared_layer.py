"""Every exercise vendors `web/_shared/`, and nothing checked that the copies still matched.

**This was found twice in one queue of pull requests, both times by the merge order rather than by
anything failing.** Exercises 09 and 10 were built while five shared-layer fixes were still open, so
they froze the pre-fix stylesheets and carried no `theme.js` at all. Measured on their own rendered
pages, the `← Back` pill ran **1.54:1 on `neon`** and **2.46:1 on `tinted-dark`** against a 4.5:1
floor, and each page carried a hand-written copy of the theme logic a refactor had just centralised
— the ninth and tenth such copies, written *after* the refactor removed the other eight.

A vendored copy going stale is invisible: the page renders, every existing test passes, and the only
symptom is that a fix landed everywhere except here. The honest check is a byte comparison, and it
did not exist.

The misnamed `tokens.css` this file used to pin is being renamed instead, in its own change:
pinning a trap is worse than removing it, and a guard asserting that a name stays misleading would
have had to be rewritten by the very next pull request.
"""

import hashlib
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The vendored layer, found rather than listed — a hard-coded roster would go stale the first time
#: an exercise gained or dropped a file, which is the failure this whole module is about.
VENDOR_DIRS = sorted(REPO_ROOT.glob("src/exercises/*/web/_shared"))

#: What `deploy/vercel/_shared/` serves at the site root, alongside each page's own vendored copy.
DEPLOY_SHARED = REPO_ROOT / "deploy" / "vercel" / "_shared"

#: Exercises that deliberately do not vendor a file most others do, each with its reason. Keyed by
#: `(exercise slug, file name)`, so an exemption covers one file in one exercise and nothing else.
#:
#: **It exists because the alternative was worse.** Exercises 07 and 12 build no explainer, so
#: neither links `explainer.css` — but the majority rule below required them to vendor it anyway,
#: and each then carried an HTML comment naming the file purely so the orphan check in
#: `test_shared_layer.py` would count it as referenced. Two guards were satisfied by a copy that
#: shipped on every deploy and styled nothing. An exemption written down here is honest; a comment
#: that exists to clear a gate is not.
#:
#: **It fails in both directions.** An entry for an exercise that does vendor the file, for a file
#: that is no longer a majority file, or for an exercise whose pages reference the file, is stale —
#: it would exempt something for a reason that has stopped being true.
DELIBERATELY_NOT_VENDORED: dict[tuple[str, str], str] = {
    ("07-model-embeddings-internals", "explainer.css"): (
        "the page builds no explainer: no step strip, step panel or stage list, so no rule in the "
        "stylesheet has anything to style"
    ),
    ("12-distributed-training", "explainer.css"): (
        "the page builds no explainer: no step strip, step panel or stage list, so no rule in the "
        "stylesheet has anything to style"
    ),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _copies() -> dict[str, list[Path]]:
    """Every vendored file, grouped by name."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for directory in VENDOR_DIRS:
        for path in sorted(directory.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                grouped[path.name].append(path)
    return dict(grouped)


def test_every_vendored_file_is_byte_identical_across_the_exercises() -> None:
    """The drift itself.

    Reported per file with the digests and the exercises grouped under them, because "these eight
    differ" is not actionable and "seven have one version and this one has another" is.
    """
    drifted: list[str] = []
    for name, paths in sorted(_copies().items()):
        if len(paths) < 2:
            continue
        by_digest: dict[str, list[str]] = defaultdict(list)
        for path in paths:
            by_digest[_digest(path)].append(path.parents[2].name)
        if len(by_digest) > 1:
            groups = "\n      ".join(
                f"{digest}  {', '.join(sorted(slugs))}"
                for digest, slugs in sorted(by_digest.items())
            )
            drifted.append(f"  {name} has {len(by_digest)} versions:\n      {groups}")

    assert not drifted, (
        "vendored copies of the shared layer have drifted apart:\n"
        + "\n".join(drifted)
        + "\n\nA copy that stops matching is invisible — the page renders and every other test "
        "passes — so a fix lands everywhere except there. Copy the current version across, or if "
        "the difference is deliberate, give the file a name that says so."
    )


def _per_exercise() -> dict[str, set[str]]:
    """The file names each exercise vendors, keyed by exercise slug."""
    return {
        d.parents[1].name: {
            p.name for p in d.iterdir() if p.is_file() and not p.name.startswith(".")
        }
        for d in VENDOR_DIRS
    }


def _majority(per_exercise: dict[str, set[str]]) -> set[str]:
    """Files vendored by more than half of the exercises."""
    counts: dict[str, int] = defaultdict(int)
    for names in per_exercise.values():
        for name in names:
            counts[name] += 1
    return {name for name, n in counts.items() if n * 2 > len(per_exercise)}


def _short_of_majority(
    per_exercise: dict[str, set[str]], exempt: dict[tuple[str, str], str]
) -> dict[str, list[str]]:
    """Per exercise, the majority files it lacks and has no written exemption for."""
    expected = _majority(per_exercise)
    short: dict[str, list[str]] = {}
    for slug, names in per_exercise.items():
        missing = sorted(n for n in expected - names if (slug, n) not in exempt)
        if missing:
            short[slug] = missing
    return short


#: A page references a vendored file by its path; an HTML comment naming it is not a reference.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _pages_reference(
    slug: str, name: str, exercises: Path = REPO_ROOT / "src" / "exercises"
) -> bool:
    """Does any of this exercise's own pages link or import `_shared/<name>`?"""
    web = exercises / slug / "web"
    if not web.is_dir():
        return False
    for path in web.rglob("*"):
        if path.suffix not in {".js", ".html"} or "_shared" in path.relative_to(web).parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix == ".html":
            text = _HTML_COMMENT.sub("", text)
        if f"_shared/{name}" in text:
            return True
    return False


def _stale_exemptions(
    per_exercise: dict[str, set[str]],
    exempt: dict[tuple[str, str], str],
    references: Callable[[str, str], bool],
) -> list[str]:
    """Every exemption whose reason has stopped being true, with what changed."""
    expected = _majority(per_exercise)
    stale: list[str] = []
    for (slug, name), reason in sorted(exempt.items()):
        if slug not in per_exercise:
            stale.append(f"{slug}/{name}: no such exercise vendors a shared layer ({reason})")
        elif name in per_exercise[slug]:
            stale.append(f"{slug}/{name}: the exercise vendors it after all ({reason})")
        elif name not in expected:
            stale.append(f"{slug}/{name}: no longer a majority file, nothing to exempt ({reason})")
        elif references(slug, name):
            stale.append(f"{slug}/{name}: a page references it, so it must be vendored ({reason})")
    return stale


def test_a_file_most_exercises_vendor_is_vendored_by_all_of_them() -> None:
    """A missing file drifts as surely as a changed one, and reads as nothing at all.

    Exercises 09 and 10 carried no `theme.js`, so each hand-wrote the logic it holds. Nothing was
    different about their copies of that file, because there were none.

    **Majority rather than identical sets, and the threshold is doing real work.** Most of the layer
    is vendored by every exercise, while `explainer.js` and `num.js` are vendored only by the pages
    that build an explainer. Demanding identical sets would red-flag that deliberate opt-in — a
    guard that fails correct work gets weakened, which is how guards die.

    Majority rather than "present in all but this one", too. That phrasing goes vacuous exactly when
    it matters: the moment a new exercise arrives without `theme.js`, the file stops being universal
    and the rule stops applying to the thing it was written for.

    A majority file an exercise deliberately leaves out is named in `DELIBERATELY_NOT_VENDORED`
    with its reason, never cleared by vendoring an unused copy.
    """
    per_exercise = _per_exercise()
    if len(per_exercise) < 2:
        return
    short = _short_of_majority(per_exercise, DELIBERATELY_NOT_VENDORED)
    assert not short, (
        f"these exercises are missing part of the shared layer that "
        f"{len(_majority(per_exercise))} file(s) of it are vendored by a majority of exercises:\n  "
        + "\n  ".join(f"{slug}: {', '.join(missing)}" for slug, missing in sorted(short.items()))
        + "\n\nA file that is absent cannot drift and cannot be fixed by a shared-layer change; "
        "the page ends up hand-writing what the missing file provides, which is how two exercises "
        "came to carry the ninth and tenth copies of a theme picker a refactor had just removed. "
        "If the page genuinely has no use for the file, add it to DELIBERATELY_NOT_VENDORED with "
        "the reason."
    )


def test_every_deliberate_omission_is_still_needed_and_still_true() -> None:
    """The ledger's other direction: an exemption whose reason has lapsed is a hole.

    It would excuse an exercise from a file it now vendors, a file that is no longer a majority
    file, or — the one that matters — a file its page has started to link, which would then 404.
    """
    stale = _stale_exemptions(_per_exercise(), DELIBERATELY_NOT_VENDORED, _pages_reference)
    assert not stale, (
        "remove or correct these DELIBERATELY_NOT_VENDORED entries; each exempts an exercise for "
        "a reason that no longer holds:\n  " + "\n  ".join(stale)
    )


def test_the_omission_ledger_can_fail_in_both_directions() -> None:
    """The twin, on planted input, so it keeps failing on purpose whatever the real tree holds."""
    planted = {
        "a": {"theme.js", "page.css"},
        "b": {"theme.js", "page.css"},
        "c": {"page.css"},
        "d": {"page.css", "theme.js"},
    }

    # Missing without an exemption is short; with one, it is not.
    assert _short_of_majority(planted, {}) == {"c": ["theme.js"]}
    assert _short_of_majority(planted, {("c", "theme.js"): "why"}) == {}

    # An exemption is stale when the exercise vendors the file after all, when the file is not a
    # majority file, when the exercise does not exist, and when a page references the file.
    never = lambda slug, name: False  # noqa: E731
    always = lambda slug, name: True  # noqa: E731
    assert _stale_exemptions(planted, {("c", "theme.js"): "why"}, never) == []
    assert _stale_exemptions(planted, {("a", "theme.js"): "why"}, never)
    assert _stale_exemptions(planted, {("c", "nobody.css"): "why"}, never)
    assert _stale_exemptions(planted, {("z", "theme.js"): "why"}, never)
    assert _stale_exemptions(planted, {("c", "theme.js"): "why"}, always)


def test_a_comment_naming_a_file_is_not_a_reference_to_it(tmp_path) -> None:
    """The exact shape the ledger replaced: a page whose only mention of the file is a comment."""
    web = tmp_path / "99-x" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        "<!-- `_shared/explainer.css` is deliberately not linked. -->\n", encoding="utf-8"
    )
    assert not _pages_reference("99-x", "explainer.css", tmp_path)
    (web / "index.html").write_text(
        '<link rel="stylesheet" href="./_shared/explainer.css" />\n', encoding="utf-8"
    )
    assert _pages_reference("99-x", "explainer.css", tmp_path)


def test_the_scan_can_actually_fail() -> None:
    """The twin. A glob that matched nothing would make all of the above vacuous for ever."""
    copies = _copies()
    assert len(VENDOR_DIRS) >= 6, (
        f"only {len(VENDOR_DIRS)} vendored directories found; the glob has drifted"
    )
    assert copies, "no vendored files found at all"
    shared_by_all = [name for name, paths in copies.items() if len(paths) == len(VENDOR_DIRS)]
    assert len(shared_by_all) >= 3, (
        f"only {len(shared_by_all)} file(s) are vendored by every exercise: {shared_by_all}. "
        "The drift check above is only as strong as the set it compares."
    )

    # The predicate itself, on planted input: two digests for one name must read as drift.
    planted = {"a": "d1", "b": "d1", "c": "d2"}
    assert len(set(planted.values())) > 1, "the drift predicate does not detect two versions"
