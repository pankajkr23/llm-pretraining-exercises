"""Warn loudly when a local-only file has gone missing from this checkout.

`notebooks/S*.ipynb` and `src/exercises/*/tools/build_notebook.py` are gitignored, which makes them
the only files in the repo with **no second copy**. Losing one is permanent in a way no other
deletion here is.

It has already happened once, and not because anyone deleted anything: after the builders were
untracked, an ordinary `git checkout main && git pull` destroyed all five. `checkout` restored them
as tracked files from the pre-merge `main`, then the fast-forward applied the commit that removed
them from the index, so git deleted the working-tree copies. They were recovered from `db9b288^`
only because that commit was still reachable.

So this is not a test of the code. It is a **tripwire on the working tree**, and it exists because
the danger is routine git operations rather than carelessness.

**On a fresh clone every file here is legitimately absent, so it skips** — it cannot run in CI and
is not meant to. Run it on the checkout that holds the files, especially after a branch switch,
pull, merge, rebase or stash.
"""

import sys
from pathlib import Path

import pytest
from _exercises import exercises_in

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import backup_local_only as backup  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


EXERCISES = exercises_in(REPO_ROOT / "src" / "exercises")

#: One notebook and one builder per exercise, by convention.
EXPECTED_NOTEBOOKS = [
    REPO_ROOT / "notebooks" / f"S{p.name[:2]}-{p.name[3:]}.ipynb" for p in EXERCISES
]
EXPECTED_BUILDERS = [p / "tools" / "build_notebook.py" for p in EXERCISES]

#: The requirements text for each exercise. Gitignored by name everywhere, so a clone has none and a
#: healthy working checkout has one per exercise.
EXPECTED_BRIEFS = [p / "REQUIREMENTS.md" for p in EXERCISES]

#: Programme-level material — the course corpus. **This was the largest exposure and nothing
#: watched it.**
#:
#: The three lists above are the classes `AGENTS.md` names, and they were the only ones guarded.
#: The confidential reference material now lives **outside the repository entirely**, so it can no
#: longer be named by a tracked path, ignored by mistake, or committed by accident. It is still
#: watched here, via the backup tool that knows where it is. `docs/EXPLAINER_*.md` are the two
#: files any explainer is supposed to be built from. All gitignored, none regenerable, none
#: guarded. A tripwire that covers the documented cases and not the biggest one is a tripwire that
#: reads as coverage.
#:
#: Counted rather than enumerated: the corpus grows a file per topic, so a fixed list would go
#: stale and a stale list here fails silently in the safe-looking direction.
NOTES_CORPUS = backup.EXTERNAL_SOURCES["notes"]

#: Where `tools/backup_local_only.py` writes. Read here as a **high-water mark**: a file the store
#: holds and this checkout does not is a loss, and no hand-written floor can notice that.
STORE = REPO_ROOT.parent / f".{REPO_ROOT.name}-local-only"
EXPECTED_PROGRAMME = [
    REPO_ROOT / "docs" / "REQUIREMENTS.md",
    REPO_ROOT / "docs" / "EXPLAINER_PROMPT.md",
    REPO_ROOT / "docs" / "EXPLAINER_PATTERN.md",
]


#: Every class this file watches, flattened. Used to answer "is this a working checkout?" **once**,
#: rather than once per class.
_ALL_WATCHED = EXPECTED_NOTEBOOKS + EXPECTED_BUILDERS + EXPECTED_BRIEFS + EXPECTED_PROGRAMME


def is_a_working_checkout() -> bool:
    """Whether this machine is supposed to have local-only files at all.

    **The hole this closes is the incident the file was written for.** Every guard here decided
    "clone or loss?" from the emptiness of *its own* class — so deleting all six notebooks made the
    notebook guard skip, deleting all six builders made the builder guard skip, and the exact event
    that destroyed all five builders in one `checkout && pull` would have turned the whole file
    green. A guard that reports success on the total loss it exists to catch is worse than absent.

    Liveness is therefore established across every class at once: a real clone has **nothing** from
    any of them, and a working checkout that has lost one class still has the others.

    Returns:
        True when any watched file exists anywhere.
    """
    return any(p.is_file() for p in _ALL_WATCHED) or NOTES_CORPUS.is_dir()


def _partial(paths: list[Path]) -> bool:
    """True when some but not all exist — a clone has none, a healthy checkout has all."""
    present = [p for p in paths if p.is_file()]
    return 0 < len(present) < len(paths)


def test_no_topic_notebook_has_gone_missing() -> None:
    """All of them or none of them. A gap means one was destroyed on this machine."""
    present = [p for p in EXPECTED_NOTEBOOKS if p.is_file()]
    if not present and not is_a_working_checkout():
        pytest.skip("no local-only files anywhere — this is a fresh clone, not a loss")
    assert present, (
        f"all {len(EXPECTED_NOTEBOOKS)} topic notebooks are gone, and this checkout still has "
        f"other local-only files — so it is not a clone. Restore from the backup store: "
        f"`uv run python tools/backup_local_only.py --verify` names what it holds."
    )
    missing = [p.name for p in EXPECTED_NOTEBOOKS if not p.is_file()]
    assert not missing, (
        f"{len(present)} topic notebooks are present but {missing} are gone. These are "
        f"gitignored and have no second copy. Rebuild them from the exercise's "
        f"tools/build_notebook.py before doing anything else, and do not delete anything under "
        f"notebooks/ without PK's explicit permission (see AGENTS.md)."
    )


def test_no_notebook_builder_has_gone_missing() -> None:
    """The builders rebuild a lost notebook, and nothing tracked can restore a lost builder."""
    present = [p for p in EXPECTED_BUILDERS if p.is_file()]
    if not present and not is_a_working_checkout():
        pytest.skip("no local-only files anywhere — this is a fresh clone, not a loss")
    assert present, (
        f"all {len(EXPECTED_BUILDERS)} notebook builders are gone, and this checkout still has "
        f"other local-only files — so it is not a clone. This is the exact incident that has "
        f"happened twice. Restore from the backup store before doing anything else."
    )
    missing = [
        f"{p.parents[1].name}/tools/build_notebook.py" for p in EXPECTED_BUILDERS if not p.is_file()
    ]
    assert not missing, (
        f"{len(present)} notebook builders are present but {missing} are gone. Nothing tracked "
        f"can restore them. If they were removed by a branch switch or pull, recover with:\n"
        f' git checkout "$(git log --all --diff-filter=D --format=%H -1 -- '
        f"'src/exercises/*/tools/build_notebook.py')^\" -- "
        f"'src/exercises/*/tools/build_notebook.py'\n"
        f"and keep a backup outside the repo (see AGENTS.md)."
    )


def test_no_programme_level_document_has_gone_missing() -> None:
    """`docs/REQUIREMENTS.md` and the two explainer specs, which nothing else watched.

    `AGENTS.md` requires both explainer documents to be read before building one, and they exist
    only here. Losing them does not break a build — it silently removes the standard the next
    explainer would have been held to.
    """
    present = [p for p in EXPECTED_PROGRAMME if p.is_file()]
    if not present and not is_a_working_checkout():
        pytest.skip("no local-only files anywhere — this is a fresh clone, not a loss")
    assert present, (
        "every programme-level document is gone, and this checkout still has other local-only "
        "files — so it is not a clone. Restore from the backup store."
    )
    missing = [str(p.relative_to(REPO_ROOT)) for p in EXPECTED_PROGRAMME if not p.is_file()]
    assert not missing, (
        f"{len(present)} programme-level documents are present but {missing} are gone. They are "
        f"gitignored and have no second copy in this repo. Restore from the backup store: "
        f"`uv run python tools/backup_local_only.py --verify` will say whether it has them."
    )


def test_the_reference_corpus_has_not_shrunk() -> None:
    """The course material — records, requirements, notes — is the biggest unguarded exposure.

    **Measured against the backup store, not against a hand-written floor.** The first version
    required at least one topic note per exercise, which tolerated losing two thirds of the
    corpus and ignored the forty-two diagrams entirely: a floor somebody typed is a floor that
    stops meaning anything the moment the corpus grows. The store is a high-water mark that moves
    on its own, so "fewer files than last time" is the question, and it is the right one.
    """
    if not NOTES_CORPUS.is_dir():
        if is_a_working_checkout():
            pytest.fail(
                f"the reference material at {NOTES_CORPUS} is gone entirely and this checkout "
                "still has other local-only files, so it is not a clone."
            )
        pytest.skip("the reference material is not present on this machine")

    here = {p for p in NOTES_CORPUS.rglob("*") if p.is_file() and p.name != ".DS_Store"}

    backed_up = STORE / "notes"
    if backed_up.is_dir():
        was = {
            p.relative_to(backed_up)
            for p in backed_up.rglob("*")
            if p.is_file() and p.name != ".DS_Store"
        }
        gone = sorted(str(r) for r in was - {p.relative_to(NOTES_CORPUS) for p in here})
        assert not gone, (
            f"{len(gone)} files the backup store holds are missing from the reference "
            f"material. Restore them from {backed_up} before doing anything else."
        )
    else:
        # No store on this machine yet, so fall back to the shape check. Weaker on purpose: it is
        # better than nothing and it says so.
        assert len(here) >= len(EXERCISES), (
            f"the reference material holds {len(here)} files for {len(EXERCISES)} exercises, "
            f"and there is no backup store to compare against. Run tools/backup_local_only.py."
        )


def test_no_watched_file_has_been_emptied() -> None:
    """**Presence is not health, and these files are regenerated constantly.**

    `is_file()` returns True for a zero-byte notebook. A builder that crashed half way through
    writing leaves exactly that, and every guard above reports the file present — so the likelier
    loss here, a bad overwrite rather than a deletion, passes cleanly. The floors are deliberately
    crude: they catch empty and truncated, not subtly wrong, which is what a content hash in the
    store is for.
    """
    floors = {".ipynb": 500, ".py": 500, ".md": 100}
    thin = []
    for path in _ALL_WATCHED:
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size < floors.get(path.suffix, 1):
            thin.append(f"{path.relative_to(REPO_ROOT)} ({size} bytes)")

    assert not thin, (
        f"these files exist but are empty or truncated: {thin}. A guard that only checks presence "
        f"would call them healthy. Restore from the backup store."
    )


def test_every_topic_notebook_is_still_valid_json() -> None:
    """A notebook that no longer parses is lost, whatever its size says.

    The builder writes JSON; an interrupted write produces a file that opens, has a plausible
    length, and cannot be read by Jupyter or by anything else.
    """
    import json

    broken = []
    for path in EXPECTED_NOTEBOOKS:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            broken.append(f"{path.name}: {exc}")
            continue
        if not payload.get("cells"):
            broken.append(f"{path.name}: parses but has no cells")

    assert not broken, f"topic notebooks that no longer read as notebooks: {broken}"


def test_the_backup_store_is_named_somewhere_a_reader_will_find_it() -> None:
    """A backup nobody knows about is not a backup.

    The tripwire tells you a file is gone; it has to also tell you where the copy is. This asserts
    the two stay connected, because the recovery instructions above are the only thing standing
    between a loss and a permanent loss.
    """
    tool = REPO_ROOT / "tools" / "backup_local_only.py"
    assert tool.is_file(), "the backup tool is gone; nothing else can restore a local-only file"
    assert "backup_local_only" in (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8"), (
        "AGENTS.md's MANDATORY section does not name the backup tool, so a reader following the "
        "rules would never learn a store exists"
    )


def test_the_tripwire_distinguishes_a_clone_from_a_loss() -> None:
    """The guard must skip on "none present" and fail on "some present".

    Without this the test would be satisfied by an empty checkout, which is the state it most needs
    to tell apart from a deletion.
    """
    assert not _partial([]), "an empty set is a clone, not a loss"
    here, gone = Path(__file__), REPO_ROOT / "does-not-exist-xyz"
    assert _partial([here, gone]), "a mixed set is a loss and must be flagged"
    assert not _partial([here]), "a complete set is healthy"


def test_a_bare_scaffold_directory_is_not_yet_an_exercise(tmp_path: Path) -> None:
    """The regression this filter exists for, pinned.

    An empty `06-build-training-dataset/` appeared before exercise 06 had any content, and both
    guards below reported a *loss* -- 5 notebooks present, 1 "gone" -- when nothing had been lost.
    A guard that cries wolf gets ignored, so the false positive is as much a defect as a miss.
    """
    root = tmp_path / "exercises"
    real = root / "01-real"
    real.mkdir(parents=True)
    (real / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    (root / "06-scaffold-only").mkdir()

    found = [p.name for p in exercises_in(root)]
    assert found == ["01-real"], f"the bare scaffold was counted as an exercise: {found}"


def test_the_filter_still_counts_a_real_exercise(tmp_path: Path) -> None:
    """The twin: narrowing the filter must not narrow it to nothing.

    Without this, deleting the glob entirely would make every guard in this file vacuously pass.
    """
    root = tmp_path / "exercises"
    for name in ("01-a", "02-b", "03-c"):
        d = root / name
        d.mkdir(parents=True)
        (d / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert [p.name for p in exercises_in(root)] == ["01-a", "02-b", "03-c"]


def test_the_filter_ignores_directories_that_do_not_match_the_naming_rule(tmp_path: Path) -> None:
    """`NN-slug` is the convention; `docs/`, `common/` and `9-x` are not exercises."""
    root = tmp_path / "exercises"
    root.mkdir(parents=True)
    for name in ("common", "docs", "9-single-digit", "abc-not-numeric"):
        d = root / name
        d.mkdir()
        (d / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert exercises_in(root) == []


def test_a_file_named_like_an_exercise_is_not_an_exercise(tmp_path: Path) -> None:
    """`is_dir()` matters: a stray `07-notes.md` must not be scanned for a notebook."""
    root = tmp_path / "exercises"
    root.mkdir(parents=True)
    (root / "07-notes.md").write_text("not a directory", encoding="utf-8")

    assert exercises_in(root) == []


def test_no_brief_has_gone_missing() -> None:
    """Requirement files are local-only, so nothing tracked can restore one.

    This is not hypothetical. All four of exercises 01-04's requirement documents were destroyed
    by an ordinary
    branch switch after the commit that untracked them, and were only recoverable because
    `18015b1^` was still reachable. They had existed in git once; a requirements doc written
    *after* the
    untracking convention would not have that safety net at all.
    """
    present = [p for p in EXPECTED_BRIEFS if p.is_file()]
    if not present:
        pytest.skip("no requirement documents here — a fresh clone has none (they are gitignored)")
    missing = [f"{p.parent.name}/REQUIREMENTS.md" for p in EXPECTED_BRIEFS if not p.is_file()]
    assert not missing, (
        f"{len(present)} present, {missing} gone. Nothing tracked can restore "
        f"a requirements file written since the untracking convention. If lost to a branch switch, "
        f"recover them with:\n"
        f" git show 18015b1^:src/exercises/<slug>/REQUIREMENTS.md > <same path>\n"
        f"and keep a backup outside the repo (see AGENTS.md)."
    )
