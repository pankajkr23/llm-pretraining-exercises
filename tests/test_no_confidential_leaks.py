"""No tracked file may reveal what is inside the confidential reference directory.

The reference material is not ours to redistribute. It used to sit inside the working tree,
gitignored, which stopped the files themselves reaching the remote and stopped nothing else.
Every leak this guard exists for got onto a public branch while `.gitignore` worked perfectly:

- a table listing two filenames, their line counts, and a summary of what each contained
- a module docstring citing a source file by name
- a scaffolder that wrote the requirements' path into every new exercise's requirements document
- test fixtures whose invented filenames published the real naming scheme

**Gitignoring a directory protects the bytes. It does nothing about a tracked document that
describes them.** That is the gap here.

Two halves, and they fail in different places on purpose.

`test_no_tracked_file_names_a_confidential_source` is **lexical** — it reads tracked files and looks
for the naming scheme. It needs neither the directory nor a clone to have it, so it runs everywhere,
including CI, which is where a public branch is actually decided.

`test_no_tracked_file_quotes_the_confidential_material` is the stronger check and can only run where
the material is present, so it **skips on a clone and in CI**. Say that plainly: on a fresh clone
this file protects against the naming scheme and nothing else. Quoting is caught on the machine that
has the notes, or not at all.
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import backup_local_only as backup  # noqa: E402

#: The material lives outside the repository; the backup tool is the one place that knows where.
NOTES = backup.EXTERNAL_SOURCES["notes"]

#: The naming scheme the confidential material uses, described rather than instantiated.
#:
#: **The first version listed four real examples here to explain itself, and CI caught it flagging
#: its own docstring.** That is not a false positive: a guard that spells out the scheme publishes
#: it just as surely as the documents it polices, and this file is tracked. It passed locally only
#: because it had not been committed yet, so it was not yet scanning itself.
#:
#: The shape, in words: a lowercase `s`, one or two digits, optionally an underscore and a
#: descriptive tail, then a document extension. Publishing such a name tells a reader what the
#: directory holds even though its bytes never left the machine.
_CONFIDENTIAL_NAME = re.compile(
    # A literal name, OR the same shape with the digits interpolated. An f-string template
    # publishes the scheme just as plainly as a literal does, and the scaffolder carried one for
    # months precisely because the interpolated form slipped past the first version of this pattern.
    #
    # No example is written out here. Twice now this file has been caught leaking through its own
    # explanation of itself — a guard over tracked files is a tracked file, and prose that
    # illustrates the forbidden shape *is* the forbidden shape.
    r"\bs(?:\d{1,2}|\{[^}]{1,40}\})(?:_[a-z0-9_]+)?\.(?:md|html|txt|pdf)\b",
    re.IGNORECASE,
)

#: Files that must name the path for the protection to work at all. `.gitignore` cannot exclude a
#: directory without writing it down, and `PATTERNS` cannot back it up without the same. Both are
#: allowed the *directory*; neither may name a file inside it, which the pattern above still checks.
_MAY_NAME_THE_DIRECTORY = {
    ".gitignore",
    "tools/backup_local_only.py",
    "tests/test_local_only_files_present.py",
    "tests/test_backup_local_only.py",
    "tests/test_no_confidential_leaks.py",
}

#: Files whose overlap is a **functional contract**, not quoted prose — kept with a reason each.
#:
#: The distinction that matters: a paraphrase of prose loses nothing, while a paraphrase of an
#: identifier breaks the thing it identifies. `run.log`'s event names are checked verbatim by the
#: auditor, so rewording them would fail the deliverable rather than protect anything; the pipeline
#: diagram names the same stages in the same order because it describes the same pipeline. Stage
#: names are shared vocabulary, not somebody's wording.
#:
#: **Say what this costs.** The exemption is whole-file, so a genuine quote added to one of these
#: two files would not be caught. It is deliberately two files, each with a reason, and
#: `test_the_functional_ledger_has_no_stale_entries` fails when an entry stops overlapping — so the
#: list cannot quietly grow stale, only quietly grow. Keep it short, and never add to it to make a
#: red test go away.
FUNCTIONAL_OVERLAP: dict[str, str] = {
    "src/exercises/06-build-training-dataset/src/trainingdata/spec.py": (
        "REQUIRED_SEQUENCE holds the literal run.log event names the audit checks for."
    ),
    "src/exercises/06-build-training-dataset/README.md": (
        "The pipeline diagram names the same stages in the same order, as it describes the "
        "same pipeline."
    ),
}


#: How many consecutive words count as a quote. Short enough to catch a lifted sentence, long enough
#: that ordinary shared phrasing ("the model must never") does not trip it.
_SHINGLE = 12


def _tracked_relpaths() -> list[str]:
    """Repo-relative paths of every tracked file, used to tell our own filenames from theirs."""
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"], capture_output=True, text=True, check=False
    ).stdout.split()


def _tracked_text_files() -> list[Path]:
    """Every tracked file git will show us, minus binaries and the frozen standards archive."""
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"], capture_output=True, text=True, check=False
    ).stdout.split()
    keep = []
    for rel in out:
        if rel.startswith("docs/standards-history/"):
            continue  # frozen copies of past releases; never edited, and not ours to rewrite
        path = REPO_ROOT / rel
        if not path.is_file() or path.suffix in {".png", ".jpg", ".ico", ".woff", ".woff2"}:
            continue
        keep.append(path)
    return keep


def _words(text: str) -> list[str]:
    """Lowercased word run, punctuation dropped, so formatting differences do not hide a quote."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _shingles(text: str) -> set[str]:
    """Hashed n-word windows, so this never holds the material itself in memory."""
    words = _words(text)
    return {
        hashlib.blake2b(" ".join(words[i : i + _SHINGLE]).encode(), digest_size=8).hexdigest()
        for i in range(max(0, len(words) - _SHINGLE + 1))
    }


def _source_shingles() -> set[str]:
    """Hashed windows over the reference material's prose. Empty when it is not on this machine."""
    out: set[str] = set()
    if not NOTES.is_dir():
        return out
    for path in NOTES.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        try:
            out |= _shingles(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
    return out


def test_no_tracked_file_names_a_confidential_source() -> None:
    """A filename is itself a disclosure: it says what the directory holds.

    Runs everywhere, including on a clone, because this is the half that has to hold on the branch
    a stranger can read.
    """
    # Exercise 01's own published pages are s1.html .. s4.html, which fit the pattern exactly. A
    # name that belongs to a tracked file in this repo is ours to cite, so the property is not
    # "matches the shape" but "matches the shape AND is not a file we ship".
    ours = {Path(rel).name.lower() for rel in _tracked_relpaths()}

    offenders: list[str] = []
    for path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for match in _CONFIDENTIAL_NAME.finditer(text):
            if match.group(0).lower() in ours:
                continue
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{line}: {match.group(0)}")

    assert not offenders, (
        "tracked files name files inside the confidential reference directory. Gitignoring it "
        "protects the bytes, not a document that describes them — say what was decided, never "
        "which file it came from:\n  " + "\n  ".join(offenders)
    )


def test_the_name_check_catches_a_planted_reference(tmp_path: Path) -> None:
    """The twin. A guard nobody has watched fail is not a guard."""
    planted = tmp_path / "leak.md"
    # Assembled rather than written out, so this file never contains the scheme as a literal.
    name = "s" + "5" + "_" + "source" + ".md"
    planted.write_text(f"As set out in `{name}`, the rule is.\n", encoding="utf-8")
    assert _CONFIDENTIAL_NAME.search(planted.read_text(encoding="utf-8")), (
        "the pattern no longer recognises the confidential naming scheme"
    )
    # ...and does not fire on ordinary prose that merely contains a letter and digits.
    assert not _CONFIDENTIAL_NAME.search("see section s of the 5.md proposal, and README.md")


def test_no_tracked_file_quotes_the_confidential_material() -> None:
    """The stronger half: a sentence lifted verbatim, with no filename anywhere near it.

    Skips where the material is absent, which is a clone and all of CI. A guard that only skips
    protects nothing, so this one is honest about where it runs: on the machine that holds the
    notes, before a push.
    """
    if not NOTES.is_dir():
        pytest.skip("the reference material is not present here — nothing to compare against")

    source = _source_shingles()
    if not source:
        pytest.skip("no readable text in the reference material to compare against")

    offenders: list[str] = []
    for path in _tracked_text_files():
        if path.suffix not in {".md", ".py", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(path.relative_to(REPO_ROOT))
        overlap = _shingles(text) & source
        if overlap and rel not in FUNCTIONAL_OVERLAP:
            offenders.append(f"{rel} ({len(overlap)} run(s) of {_SHINGLE} words)")

    assert not offenders, (
        f"tracked files contain runs of {_SHINGLE}+ words copied from the confidential reference "
        "material. Paraphrase, or state the decision rather than the source's wording:\n  "
        + "\n  ".join(offenders)
    )


def test_the_functional_ledger_has_no_stale_entries() -> None:
    """An exemption for a file that no longer overlaps is an exemption nobody is watching.

    This is the half that keeps the list from rotting into a general-purpose silencer: the moment a
    file stops matching, its entry has to go, so the ledger only ever describes live exceptions.
    """
    source = _source_shingles()
    if not source:
        pytest.skip("no readable reference text to compare against")

    stale = []
    for rel in FUNCTIONAL_OVERLAP:
        path = REPO_ROOT / rel
        if not path.is_file():
            stale.append(f"{rel} (no such file)")
        elif not (_shingles(path.read_text(encoding="utf-8")) & source):
            stale.append(f"{rel} (no longer overlaps)")

    assert not stale, (
        "FUNCTIONAL_OVERLAP lists files that no longer need an exemption — remove them, so the "
        f"ledger keeps describing only live exceptions: {stale}"
    )
