"""The attribution NOTICE must name every redistributed file, and its numbers must be derived.

This exercise **redistributes CC BY-SA Wikipedia text** — fourteen tracked files under `corpus/`,
plus a tokenizer built from them that the deployed page offers as a download. Until `NOTICE` was
written, the licence appeared in exactly one place in the exercise: a parenthetical inside
`CLAUDE.md`, a file addressed to coding agents. An obligation recorded only in the instructions to
the machine is not one a reader of the work can discover.

Two properties are checked, and they fail in opposite directions:

- **Nothing redistributed goes unattributed.** A sixth language added to `corpus/v2/` and not to
  `NOTICE` is an attribution gap, and it is exactly the kind that happens by momentum.
- **No number in `NOTICE` is typed.** `AGENTS.md` names prose-that-states-a-number as the failure
  that has cost this repository the most edits: a table under a hand-written sentence looks
  maintained, and only the sentence is wrong. Every character count here is read back from the
  `meta.json` that produced it.
"""

import json
import re
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
NOTICE = EXERCISE / "NOTICE"
CORPUS = EXERCISE / "corpus"


def _metadata() -> dict[str, dict]:
    """Provenance for each faithful corpus file, keyed by language.

    Returns:
        The parsed `meta.json` payloads.
    """
    return {
        path.name.split(".")[0]: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(CORPUS.glob("v2/*.meta.json"))
    }


def test_the_notice_exists_and_names_the_licence() -> None:
    """Without this the whole file is decorative: the licence is the obligation."""
    assert NOTICE.is_file(), (
        "02-tokenization redistributes Wikipedia text and has no NOTICE. See the module docstring."
    )
    text = NOTICE.read_text(encoding="utf-8")
    assert "CC BY-SA" in text, "the NOTICE does not name the licence the corpus is under"
    assert "share-alike" in text.lower(), (
        "CC BY-SA is a share-alike licence and the NOTICE does not say what that means for the "
        "tokenizer derived from the corpus"
    )


@pytest.mark.parametrize("language", sorted(_metadata()))
def test_every_redistributed_language_is_attributed(language: str) -> None:
    """A language shipped in `corpus/` and missing from `NOTICE` is an attribution gap."""
    text = NOTICE.read_text(encoding="utf-8")
    meta = _metadata()[language]
    assert meta["title"] in text, (
        f"corpus/v2 ships the {language!r} article {meta['title']!r} and the NOTICE never names it"
    )


@pytest.mark.parametrize("language", sorted(_metadata()))
def test_every_character_count_in_the_notice_matches_its_metadata(language: str) -> None:
    """The numbers are read back from the files that produced them, never typed."""
    meta = _metadata()[language]
    expected = f"{meta['chars']:,}"
    text = NOTICE.read_text(encoding="utf-8")
    assert expected in text, (
        f"the NOTICE does not carry {language}'s character count {expected} from "
        f"corpus/v2/{language}.meta.json. A number in prose that no longer matches its source is "
        "the failure AGENTS.md calls the most expensive one in this repository."
    )


def test_the_notice_covers_the_derived_artefacts_the_page_hands_out() -> None:
    """The tokenizer is a derivative work of the corpus and the page offers it as a download."""
    text = NOTICE.read_text(encoding="utf-8")
    for artefact in ("web/tokenizer.json", "web/data.json"):
        assert artefact in text, (
            f"the NOTICE does not mention {artefact}, which is derived from CC BY-SA text and "
            "served publicly"
        )


def test_the_checks_can_actually_fail() -> None:
    """The twin: every assertion above is a substring test, so prove one can miss.

    Reads a deliberately incomplete NOTICE rather than mutating the real one — a mutation restored
    on the happy path is one an early return leaves behind.
    """
    planted = "# NOTICE\n\nCC BY-SA, share-alike. India 598,265.\n"
    meta = _metadata()

    missing_titles = [
        language
        for language, entry in meta.items()
        if entry["title"] not in planted and entry["title"] != "India"
    ]
    assert missing_titles, "the planted NOTICE should omit at least one language's article"

    missing_counts = [
        language for language, entry in meta.items() if f"{entry['chars']:,}" not in planted
    ]
    assert missing_counts, "the planted NOTICE should omit at least one character count"
    assert not re.search(r"tokenizer\.json", planted), (
        "the planted NOTICE should omit the artefacts"
    )
