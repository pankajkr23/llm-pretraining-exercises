"""Every published number here says what produced it, and the corpus can be checked not trusted.

**Three separate failures this file exists to keep closed**, because none of them implies the
others and all three were live:

- `results/harness.json` and `results/sensitivity.json` carried **no provenance block at all**;
- `results/training.json` carried a **16-character prefix** under a field named
  `source_sha256_prefix`, which is not a content hash and which nothing recomputed;
- the corpus was read from the repository's live `AGENTS.md` at run time, so every published loss
  was a function of a file edited on most pull requests. It had already moved — 92,021 bytes when
  the published run read it against 103,347 live — and no test was capable of noticing.

**The last one is why the checks here can exist.** A digest over a moving file can only ever be
recorded; a digest over a frozen file can be **recomputed from a clone**, which is the difference
between a decoration and a gate. It is the same argument `.quote-check-receipt.json` makes for the
quoting check, one directory up.

No `torch` anywhere in this file. It reads JSON and hashes bytes, so it runs in the ordinary CI job
rather than the `train` one — a provenance guard that only runs where the optional dependency is
installed is a provenance guard that mostly does not run.
"""

import hashlib
import json
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"
CORPUS = EXERCISE / "corpus"
MANIFEST = json.loads((CORPUS / "MANIFEST.json").read_text(encoding="utf-8"))

REQUIRED_FIELDS = (
    "config_fingerprint",
    "code_digest",
    "git_sha",
    "corpus_digest",
    "tokenizer_digest",
    "environment",
)
"""Restated here on purpose, rather than imported from the module under test.

Importing the list would make this file agree with `provenance.py` by construction, and the thing
worth checking is that the *published files* carry them — a guard that reads its expectation from
the code it is guarding cannot fail when that code drops a field.
"""

PUBLISHED = ("harness.json", "training.json", "sensitivity.json")
"""Every tracked result a document renders. All three, because two of them had nothing."""


def _result(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256((CORPUS / MANIFEST["file"]).read_bytes()).hexdigest()


def test_the_frozen_corpus_matches_its_manifest() -> None:
    """The bytes on disk are the bytes the manifest claims. Recomputed, never read back."""
    raw = (CORPUS / MANIFEST["file"]).read_bytes()
    assert len(raw) == MANIFEST["bytes"], (
        f"{MANIFEST['file']} is {len(raw):,} bytes and MANIFEST.json says {MANIFEST['bytes']:,}. "
        "These bytes are a measured input: if they changed, a published result changed with them. "
        "Read corpus/README.md before deciding which of the two is wrong."
    )
    assert _corpus_digest() == MANIFEST["sha256_digest"]


def test_the_frozen_corpus_is_the_revision_it_claims_to_be() -> None:
    """The filename names a commit, so the filename is a claim, so it gets checked.

    Skipped rather than failed where git cannot answer — a tarball or a shallow clone missing the
    blob is a fact about the checkout, not a defect in the corpus. The digest check above needs no
    git at all and is the one that runs everywhere.
    """
    import subprocess

    sha = MANIFEST["git_sha"]
    try:
        blob = subprocess.run(
            ["git", "-C", str(EXERCISE), "show", f"{sha}:AGENTS.md"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"{sha[:7]} is not reachable from this checkout")
    assert blob == (CORPUS / MANIFEST["file"]).read_bytes(), (
        f"corpus/{MANIFEST['file']} is not byte-identical to AGENTS.md at {sha[:7]}, so the "
        "filename and the manifest disagree about what was frozen."
    )


@pytest.mark.parametrize("name", PUBLISHED)
def test_every_published_result_says_what_produced_it(name: str) -> None:
    """All six fields, in all three files. Two of the three carried none of them."""
    block = _result(name).get("provenance") or {}
    missing = [field for field in REQUIRED_FIELDS if not block.get(field)]
    assert not missing, (
        f"results/{name} is missing {', '.join(missing)}. Every figure in this exercise's "
        "documents and on its page is read from one of these three files, and a number nobody "
        "can regenerate is not evidence. Re-run the producer rather than hand-editing the file."
    )


@pytest.mark.parametrize("name", PUBLISHED)
def test_every_published_result_was_measured_on_the_corpus_in_this_repository(name: str) -> None:
    """The recorded digest equals the corpus's own, recomputed here from its bytes.

    This is the check the old 16-character prefix could not support and the live `AGENTS.md` could
    not survive. It is also the one that goes red if anyone edits the frozen text.
    """
    data = _result(name)
    recorded = (data.get("corpus") or {}).get("source_digest")
    assert recorded, f"results/{name} records no corpus digest"
    assert recorded == _corpus_digest(), (
        f"results/{name} was measured on a different corpus than the one in corpus/. Either the "
        "frozen text was edited — see corpus/README.md, it never should be — or this result "
        "predates the freeze and the producer needs re-running."
    )


@pytest.mark.parametrize("name", PUBLISHED)
def test_no_published_digest_is_a_prefix(name: str) -> None:
    """`sha256:` followed by 64 hex, or it is not a content hash.

    The field this replaces held sixteen characters and was named `source_sha256_prefix`. It read
    as checked for months. A prefix is not wrong so much as unfalsifiable: nothing recomputes it,
    and at sixteen characters a reader cannot tell whether anything could.
    """
    data = _result(name)
    digests = {
        "corpus.source_digest": (data.get("corpus") or {}).get("source_digest"),
        "provenance.code_digest": (data.get("provenance") or {}).get("code_digest"),
        "provenance.corpus_digest": (data.get("provenance") or {}).get("corpus_digest"),
        "provenance.tokenizer_digest": (data.get("provenance") or {}).get("tokenizer_digest"),
    }
    for field, value in digests.items():
        assert value, f"results/{name} has no {field}"
        assert value.startswith("sha256:") and len(value) == len("sha256:") + 64, (
            f"results/{name}'s {field} is {value!r}, which is not a full-length sha256 digest."
        )


@pytest.mark.parametrize("name", PUBLISHED)
def test_every_published_result_states_its_epoch_count(name: str) -> None:
    """`total_tokens / corpus_tokens`, stated rather than left to be noticed.

    `AGENTS.md` asks for it beside any run, and asks for it **per row** where a sweep varies the
    size of the read. The sweep here varies the step count, so it varies the epochs too — which was
    computed nowhere and shown nowhere, while the page called the step count "the only arbitrary
    thing in the run".
    """
    data = _result(name)
    assert "epochs" in (data.get("corpus") or {}), f"results/{name} states no epoch count"
    for row in data.get("by_steps") or []:
        assert "epochs" in row, (
            f"results/{name} has a sweep row at {row.get('steps')} steps with no epoch count. "
            "Each row reads a different amount of the corpus, so each row has its own."
        )
