"""The two invariants that decide whether this exercise is safe to publish.

Named `test_publication_invariants` rather than `test_invariants` because exercise 03 already
has a file by that name, and pytest cannot import two test modules sharing a basename without
package markers. Same for `test_page_render.py` beside it.

Everything else in this suite protects a claim. These two protect a person.

    INV-1  no personal information reaches any published artifact
    INV-2  no corpus text is published beyond the declared, bounded window

Both are enforced over **every byte** of `web/` and over the notebook, because a leak does not
announce itself — it arrives as a plausible-looking excerpt in a field nobody thought to check. Both
are paired with twins that inject a leak and confirm the scan names it, since a scanner that reports
"clean" while doing nothing is exactly the failure mode here.

`AGENTS.md` calls this out directly: a guard that cannot fail is worse than no guard, because it
reads as coverage.
"""

import json
from pathlib import Path

import pytest
from datacleaning import pii
from datacleaning.config import EXERCISE_ROOT, Config

CFG = Config()
WEB = CFG.web_dir
DATA_JSON = WEB / "data.json"
NOTEBOOK = EXERCISE_ROOT.parents[2] / "notebooks" / "S04-data-cleaning-dedup.ipynb"

# Text files a reader could actually receive. Binary assets are excluded because they are not a
# route for a pasted email address, and reading them as UTF-8 would fail noisily for no benefit.
PUBLISHED_SUFFIXES = {".json", ".html", ".js", ".css", ".md", ".txt", ""}


def _published_files() -> list[Path]:
    if not WEB.exists():
        return []
    return sorted(p for p in WEB.rglob("*") if p.is_file() and p.suffix in PUBLISHED_SUFFIXES)


def _scan(text: str) -> list[pii.Span]:
    """Find structured identifiers, ignoring the ones we deliberately publish.

    The synthetic demo document ships in the bundle on purpose — it is what the page's PII chapter
    operates on. Its identifiers are RFC-reserved and belong to nobody, so they are excluded by
    value rather than by turning the scan off.
    """
    allowed = {s.matched for s in pii.find_structured(pii.SYNTHETIC_DEMO)}
    return [s for s in pii.find_structured(text) if s.matched not in allowed]


# ---- INV-1 · no personal information is published ----------------------------------------------


@pytest.mark.skipif(not DATA_JSON.exists(), reason="bundle not built")
def test_no_personal_information_anywhere_in_the_published_bundle():
    """Scan every published byte, not merely the fields we expect to carry text."""
    leaks: list[str] = []
    for path in _published_files():
        for span in _scan(path.read_text(encoding="utf-8", errors="replace")):
            leaks.append(f"{path.name}: {span.kind}")
    assert not leaks, f"personal information reached the published bundle: {leaks[:10]}"


def test_the_bundle_scan_can_actually_fail():
    """The twin, and the most important test in this file.

    Without it, the scan above passes on an empty directory, on a bundle it failed to read, and on a
    scanner whose patterns were accidentally emptied.
    """
    planted = 'a field with "contact": "someone@realdomain.com" inside it'
    found = _scan(planted)
    assert [s.kind for s in found] == ["email"], "a planted address was not detected"


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebook missing")
def test_no_personal_information_in_the_notebook():
    """Executing a PII cell and committing the output would bake real addresses into git.

    `test_notebook.py` already requires outputs to be stripped; this checks the source cells too,
    since a hand-pasted example is just as permanent.
    """
    leaks = [s.kind for s in _scan(NOTEBOOK.read_text(encoding="utf-8"))]
    assert not leaks, f"personal information in the notebook source: {leaks[:10]}"


@pytest.mark.skipif(not DATA_JSON.exists(), reason="bundle not built")
def test_the_pii_stage_publishes_counts_and_not_matches():
    """Aggregates are the deliverable. The matched strings are the thing being removed."""
    bundle = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    stage = next((s for s in bundle["stages"] if s["id"] == "pii"), None)
    if stage is None:
        pytest.skip("PII stage not in the bundle yet")

    assert stage["detail"]["by_kind"], "the stage should report what it masked"
    assert "matches" not in stage["detail"], "matched identifiers must never be published"
    assert stage["detail"]["name_layer"]["precision"] is None


# ---- INV-2 · corpus text is published only in a bounded window ----------------------------------


@pytest.mark.skipif(not DATA_JSON.exists(), reason="bundle not built")
def test_corpus_excerpts_stay_inside_the_declared_window():
    """The dedup chapter needs real near-identical documents on screen; nothing else does.

    `DECISIONS.md` §D6 argues why this relaxes exercise 03's absolute rule. The relaxation is
    bounded, and this is the bound.
    """
    bundle = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    excerpts = _collect_excerpts(bundle)

    assert len(excerpts) <= CFG.max_excerpts, (
        f"{len(excerpts)} excerpts published, limit is {CFG.max_excerpts}"
    )
    for text in excerpts:
        assert len(text) <= CFG.max_excerpt_chars, (
            f"an excerpt is {len(text)} characters, limit is {CFG.max_excerpt_chars}"
        )


def test_the_excerpt_bound_can_actually_fail():
    """The twin: a bundle carrying an over-long excerpt must be caught."""
    over = {"dedup": {"example_duplicates": [{"excerpt": "x" * (CFG.max_excerpt_chars + 1)}]}}
    found = _collect_excerpts(over)
    assert found and len(found[0]) > CFG.max_excerpt_chars


@pytest.mark.skipif(not DATA_JSON.exists(), reason="bundle not built")
def test_no_raw_corpus_shard_is_shipped():
    """A parquet or jsonl in `web/` would publish the corpus wholesale."""
    stray = [p.name for p in WEB.rglob("*") if p.suffix in {".parquet", ".jsonl", ".gz", ".arrow"}]
    assert not stray, f"raw corpus files in the published bundle: {stray}"


def _collect_excerpts(payload: object, key_hint: str = "excerpt") -> list[str]:
    """Walk a bundle and return every string stored under an excerpt-ish key."""
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str) and key_hint in key:
                found.append(value)
            else:
                found.extend(_collect_excerpts(value, key_hint))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_collect_excerpts(item, key_hint))
    return found


# ---- the licences that make any of this redistributable ------------------------------------------


def test_a_notice_names_every_corpus_licence():
    """CC-BY and CC-BY-SA both require attribution. Publishing excerpts without it is the breach."""
    notice = EXERCISE_ROOT / "NOTICE"
    if not notice.exists():
        pytest.skip("NOTICE not written yet")

    text = notice.read_text(encoding="utf-8")
    from datacleaning.sources import ALL_SPECS

    for spec in ALL_SPECS:
        assert spec.repo_id in text, f"{spec.repo_id} is not attributed in NOTICE"
        assert spec.licence in text, f"{spec.licence} is not named in NOTICE"
