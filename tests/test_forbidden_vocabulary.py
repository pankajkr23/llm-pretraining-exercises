"""Words that must never appear in a tracked file, checked lexically so CI can enforce them.

`tests/test_no_confidential_leaks.py` guards the *material*: its filenames, and sentences copied out
of it. This file guards the **vocabulary** — the handful of ordinary words that, used about our
source, tell a reader what kind of thing it is and how it reaches us. Neither guard subsumes the
other: a document can name no file and copy no sentence and still name the kind of thing
the source is.

**Why this is the half that has to run in CI.** The quoting check compares tracked prose against the
material itself, so it can only run on a machine that holds it — and it skips silently everywhere
else, CI included. This one needs nothing but the repo, so it is the check that actually stands
between a working tree and a public branch.

**Why lexical rather than reviewed.** The words came back three times after being removed by hand.
They are cheap to spell and expensive to notice: a sweep rewrites five hundred of them and leaves
four, and the four are indistinguishable from the rest until someone reads the file.

**A guard over tracked files is itself a tracked file**, and this one has twice caught a sibling
guard leaking through its own explanation of itself. Nothing below writes out a forbidden term to
illustrate a rule; the tests assemble their fixtures at runtime for exactly that reason.
"""

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The forbidden vocabulary, as `word -> why`. The reason is not decoration: it is what tells the
#: next person whether their new usage is the banned sense or an unrelated one.
#:
#: Each is matched whole-word and case-insensitively, including its plural and possessive.
FORBIDDEN: dict[str, str] = {
    # Assembled, never written out. This file is tracked and the gate reads tracked files, so a
    # literal here makes the guard fail on itself — which is exactly what happened on the first
    # commit that tried to add it, and what the docstring above promises not to do. A public repo
    # containing a list of the words it is hiding also tells a reader more than any single slip.
    "sess" + "ion": (
        "names the source's own unit of material. Also a pytest fixture scope and the ordinary "
        "English word for a sitting of a legislature — both are in ALLOWED."
    ),
    "trans" + "cript": (
        "names a kind of document inside the source, so it says what the material is made of. "
        "Unrelated senses — a shell record used as training data, speech-to-text, a run writing "
        "its own record — are cleared by the patterns below."
    ),
    "assign" + "ment": (
        "names the graded task as the source hands it over. Say 'the requirements' — what we were "
        "asked for is publishable; that it arrived as a numbered task from somewhere is not."
    ),
    "lect" + "ure": (
        "names the form the material was delivered in. Public speech corpora that happen to use "
        "the word as a dataset name are cleared below."
    ),
    "br" + "ief": (
        "named the per-exercise requirements document, which is why that file is REQUIREMENTS.md "
        "now. 'briefly' and 'debrief' are ordinary English and are cleared below."
    ),
}

#: Uses that are **not** the forbidden sense, each with the reason it is exempt.
#:
#: These are matched against the whole line, so an exempt phrase clears only the line it appears on
#: rather than the file. That is deliberate: a file-wide exemption is a hole the size of the file,
#: which is the mistake this repo already made once with a whole-file allowlist.
ALLOWED: tuple[tuple[str, str], ...] = (
    (r"scope\s*=\s*[\"']" + "sess" + "ion" + r"[\"']", "a pytest fixture scope — an API value"),
    (r"sync_playwright|playwright\.sync", "playwright's own API surface"),
    (r"shell trans" + "cript", "a training-data format: shell interaction logs"),
    (
        r"asr trans" + r"cri|speech trans" + r"cript|trans" + r"cription",
        "speech-to-text, a data topic",
    ),
    (r"run's own trans" + "cript", "the training run writing its own record"),
    (
        r"lectures \+ trans" + r"cript|NPTEL|SWAYAM",
        "named public speech corpora in the data catalogue — a dataset, not our source",
    ),
    (r"agent conversation|Claude Code", "this repo's own tooling, unrelated to the material"),
    (r"\bbr" + r"iefly\b|\bdebr" + r"ief", "ordinary English adverb and verb, unrelated senses"),
    (r"IIT/IISc|recordings", "a named public speech corpus, described as the dataset it is"),
)

#: Paths this guard does not read, each with the reason.
EXEMPT_PATHS: dict[str, str] = {
    "docs/standards-history/": (
        "frozen copies of past releases. They are records of what shipped, are never edited, and "
        "rewriting one would make it a record of nothing."
    ),
    "src/exercises/02-tokenization/web/tokenizer.json": (
        "the frozen vocabulary. Its entries are BPE tokens learned from real text, not prose, and "
        "its bytes are hashed — every shard manifest in exercise 06 pins that hash. A sweep did "
        "rewrite a token here once; the build caught it, and this exemption is what stops the gate "
        "from ever asking someone to do it deliberately."
    ),
    "src/exercises/02-tokenization/web/data.json": (
        "derived from that vocabulary and carrying the same tokens."
    ),
    "src/exercises/01-introductions/package-lock.json": (
        "npm writes it, including the upstream deprecation notices quoted inside it."
    ),
    "src/exercises/02-tokenization/corpus/": (
        "the tokenizer's training corpus. Its bytes are the measured input the vocabulary is "
        "derived from, so editing a word here changes a result rather than a document."
    ),
}

_BINARY = {".png", ".jpg", ".jpeg", ".ico", ".woff", ".woff2", ".gz", ".parquet"}


def _pattern(word: str) -> re.Pattern[str]:
    """Whole-word, case-insensitive, plural and possessive included."""
    return re.compile(rf"\b{word}(?:s|'s|s')?\b", re.IGNORECASE)


def _allowed(line: str) -> bool:
    """True when this line's use is one of the documented unrelated senses."""
    return any(re.search(p, line, re.IGNORECASE) for p, _ in ALLOWED)


def _tracked_files() -> list[Path]:
    """Every tracked text file this guard is responsible for."""
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"], capture_output=True, text=True, check=False
    ).stdout.split()
    out = []
    for rel in listed:
        if any(rel.startswith(prefix) for prefix in EXEMPT_PATHS):
            continue
        path = REPO_ROOT / rel
        if path.is_file() and path.suffix.lower() not in _BINARY:
            out.append(path)
    return out


def _violations() -> list[str]:
    """Every forbidden term in a tracked file, minus the documented unrelated senses."""
    found: list[str] = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if _allowed(line):
                continue
            for word in FORBIDDEN:
                if _pattern(word).search(line):
                    rel = path.relative_to(REPO_ROOT)
                    found.append(f"{rel}:{number}: {line.strip()[:90]}")
                    break
    return found


def test_no_tracked_file_uses_the_forbidden_vocabulary() -> None:
    """The gate. Runs anywhere, needs nothing but the repo, and blocks a public branch."""
    offenders = _violations()
    assert not offenders, (
        f"{len(offenders)} tracked line(s) use vocabulary that describes the confidential source. "
        "Name the exercise that covers the topic, or call it 'the source' — and if a use is a "
        "genuinely unrelated sense, add it to ALLOWED with the reason rather than rewording good "
        "work:\n  " + "\n  ".join(offenders)
    )


def test_the_gate_catches_a_planted_use(tmp_path: Path) -> None:
    """The twin. A guard nobody has watched fail is not a guard.

    The fixture is assembled at runtime rather than written out, because this file is tracked and
    the gate reads tracked files — a literal here would make the guard fail on itself, which has
    already happened twice to its sibling.
    """
    word = "sess" + "ion"
    planted = tmp_path / "doc.md"
    planted.write_text(f"As the {word} explains, the rule is.\n", encoding="utf-8")
    line = planted.read_text(encoding="utf-8")

    assert _pattern(word).search(line), "the pattern no longer matches the forbidden term"
    assert not _allowed(line), "an ordinary prose use must not be treated as an exempt sense"


def test_plurals_and_possessives_are_caught() -> None:
    """A plural or a possessive is the same disclosure as the singular."""
    word = "trans" + "cript"
    for variant in (word, word + "s", word + "'s"):
        assert _pattern(word).search(f"the {variant} says"), f"{variant} escapes the pattern"


def test_every_allowed_sense_actually_clears_a_line() -> None:
    """An exemption for a sense nothing uses is an exemption nobody is watching.

    Fails in the other direction, so the list can only grow by someone's decision — never quietly,
    and never as the easy way to silence a red gate.
    """
    text = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in _tracked_files()
        if p.suffix != ".lock"
    )
    unused = [p for p, _ in ALLOWED if not re.search(p, text, re.IGNORECASE)]
    assert not unused, (
        "ALLOWED lists senses that appear nowhere in the repo — remove them so the list keeps "
        f"describing live exceptions: {unused}"
    )


def test_every_exempt_path_is_real_or_deliberately_local() -> None:
    """A path exemption outliving its path is an exemption pointing at nothing.

    **"Exists" is the wrong question on a clone**, and asking it turned CI red the first time this
    ran there: one of the exempt paths is gitignored, so it is present on a working checkout and
    absent everywhere else by design. A path is stale only when it is neither present nor something
    git is deliberately keeping out — which is a fact about the repo rather than about the machine.
    """
    stale = []
    for rel in EXEMPT_PATHS:
        if (REPO_ROOT / rel).exists():
            continue
        ignored = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", rel],
            capture_output=True,
            check=False,
        )
        if ignored.returncode != 0:
            stale.append(rel)
    assert not stale, (
        "EXEMPT_PATHS names paths that neither exist nor are gitignored, so they exempt "
        f"nothing: {stale}"
    )
