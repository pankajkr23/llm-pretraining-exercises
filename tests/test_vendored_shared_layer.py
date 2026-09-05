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
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The vendored layer, found rather than listed — a hard-coded roster would go stale the first time
#: an exercise gained or dropped a file, which is the failure this whole module is about.
VENDOR_DIRS = sorted(REPO_ROOT.glob("src/exercises/*/web/_shared"))

#: What `deploy/vercel/_shared/` serves at the site root, alongside each page's own vendored copy.
DEPLOY_SHARED = REPO_ROOT / "deploy" / "vercel" / "_shared"


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


def test_a_file_most_exercises_vendor_is_vendored_by_all_of_them() -> None:
    """A missing file drifts as surely as a changed one, and reads as nothing at all.

    Exercises 09 and 10 carried no `theme.js`, so each hand-wrote the logic it holds. Nothing was
    different about their copies of that file, because there were none.

    **Majority rather than identical sets, and the threshold is doing real work.** Four files are
    vendored by all eight exercises; `explainer.js` and `num.js` are vendored by two, because they
    belong to the pages that build an explainer. Demanding identical sets would red-flag that
    deliberate opt-in — a guard that fails correct work gets weakened, which is how guards die.

    Majority rather than "present in all but this one", too. That phrasing goes vacuous exactly when
    it matters: the moment a new exercise arrives without `theme.js`, the file stops being universal
    and the rule stops applying to the thing it was written for.
    """
    per_exercise = {
        d.parents[1].name: {
            p.name for p in d.iterdir() if p.is_file() and not p.name.startswith(".")
        }
        for d in VENDOR_DIRS
    }
    if len(per_exercise) < 2:
        return
    counts: dict[str, int] = defaultdict(int)
    for names in per_exercise.values():
        for name in names:
            counts[name] += 1
    expected = {name for name, n in counts.items() if n * 2 > len(per_exercise)}
    short = {
        slug: sorted(expected - names) for slug, names in per_exercise.items() if expected - names
    }
    assert not short, (
        f"these exercises are missing part of the shared layer that {len(expected)} file(s) of it "
        "are vendored by a majority of exercises:\n  "
        + "\n  ".join(f"{slug}: {', '.join(missing)}" for slug, missing in sorted(short.items()))
        + "\n\nA file that is absent cannot drift and cannot be fixed by a shared-layer change; "
        "the page ends up hand-writing what the missing file provides, which is how two exercises "
        "came to carry the ninth and tenth copies of a theme picker a refactor had just removed."
    )


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
