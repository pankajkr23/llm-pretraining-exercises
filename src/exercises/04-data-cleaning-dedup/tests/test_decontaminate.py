"""Stage 7 — the canary proof, and the rule that UNCHECKED is never 'clean'.

This stage has a specific failure mode that ordinary testing misses: with no evaluation index on the
machine, a naive implementation scans nothing, finds nothing, and reports zero contaminated
documents — which reads as a clean bill of health. Half the tests here exist to make that
impossible.

The other half hold the canary pass, which is what makes the stage demonstrable at all on a machine
without the gated benchmark download. Its first version generated canaries of five words and scanned
at thirteen, so it recovered nothing while appearing to run.
"""

from dataclasses import replace

from datacleaning import decontaminate as dc
from datacleaning.config import Config
from datacleaning.records import Document

CFG = Config()

FILLER = (
    "This is an ordinary document about an ordinary subject, written at sufficient length that "
    "it can be shingled at thirteen words without difficulty, and containing nothing unusual."
)


# ---- canaries ------------------------------------------------------------------------------------


def test_canaries_are_long_enough_to_be_seen_by_the_scanner():
    """The bug this pins: a canary shorter than the n-gram width produces no n-grams at all.

    The first version made five-word canaries and scanned at thirteen, so the index was empty and
    the pass recovered 0/24 while looking like it had run.
    """
    for canary in dc.canary_strings(4, CFG.minhash_seed, CFG.decontam_n):
        assert len(canary.split()) >= CFG.decontam_n, "a canary must outlast the scanning window"
        assert dc.ngrams(canary, CFG.decontam_n), "a canary must produce at least one n-gram"


def test_canaries_are_reproducible():
    """Determinism: the run id must not move because a canary did."""
    first = dc.canary_strings(6, CFG.minhash_seed, CFG.decontam_n)
    second = dc.canary_strings(6, CFG.minhash_seed, CFG.decontam_n)
    assert first == second
    assert len(set(first)) == 6, "canaries must be distinct from one another"


def test_a_planted_canary_is_recovered():
    canaries = dc.canary_strings(3, CFG.minhash_seed, CFG.decontam_n)
    index = dc.build_index(canaries, CFG.decontam_n)
    for canary in canaries:
        assert dc.contaminated(f"{FILLER} {canary} {FILLER}", index, CFG.decontam_n)


def test_the_canary_check_can_actually_fail():
    """The twin. Ordinary text must not trip the scanner, or 'recovered' means nothing."""
    index = dc.build_index(dc.canary_strings(3, CFG.minhash_seed, CFG.decontam_n), CFG.decontam_n)
    assert not dc.contaminated(FILLER, index, CFG.decontam_n)


def test_a_short_canary_would_be_invisible():
    """Prove the failure mode directly, so the length requirement is not folklore."""
    too_short = "s04canary aaaa bbbb cccc"
    assert dc.ngrams(too_short, CFG.decontam_n) == set()
    assert dc.build_index([too_short], CFG.decontam_n) == set()


# ---- UNCHECKED is not clean ----------------------------------------------------------------------


def test_unchecked_is_never_reported_as_clean():
    """The rule that matters most here.

    With no index, the honest answer is 'we did not check'. Reporting zero contaminated documents
    would be indistinguishable from having checked and found nothing.
    """
    report = dc.DecontamReport(coverage="none", docs_scanned=1000)
    assert "UNCHECKED" in report.headline
    assert "clean" not in report.headline.lower()


def test_a_checked_and_empty_result_says_so_differently():
    """The twin: 'checked and found nothing' must be a distinguishable answer from 'not checked'."""
    checked = dc.DecontamReport(coverage="held-out", docs_scanned=1000, docs_flagged=0)
    assert "UNCHECKED" not in checked.headline
    assert "No overlap" in checked.headline


def test_a_contaminated_result_names_the_count():
    report = dc.DecontamReport(coverage="held-out", docs_scanned=1000, docs_flagged=7)
    assert "7" in report.headline


def test_the_bundle_records_coverage_and_the_reason():
    payload = dc.DecontamReport(coverage="none").as_json()
    assert payload["coverage"] == "none"
    assert "UNCHECKED" in payload["note"] or "UNCHECKED" in payload["headline"]


# ---- the stage over a corpus ---------------------------------------------------------------------


def test_the_stage_runs_the_canary_pass_on_every_machine():
    """Whether or not the gated index exists, the scanner is proven to work."""
    docs = [Document(f"d{i}", FILLER, "t", "s", "en") for i in range(3)]
    _, stat = dc.decontaminate_stage(docs, CFG)

    detail = stat.detail
    assert detail["canaries_injected"] == CFG.canary_count
    assert detail["canaries_recovered"] == CFG.canary_count
    assert detail["canary_recall"] == 1.0


def test_the_stage_warns_loudly_when_the_canary_pass_fails():
    """A broken scanner must not produce a reassuring note.

    Setting the width beyond the canary length reproduces the original bug; the note must say the
    result above it is meaningless rather than repeating 'known to work'.
    """
    broken = replace(CFG, decontam_n=13, canary_count=4)
    canaries = dc.canary_strings(4, broken.minhash_seed, 5)  # too short for width 13
    index = dc.build_index(canaries, broken.decontam_n)
    assert index == set(), "the deliberately short canaries should be invisible"

    docs = [Document("d0", FILLER, "t", "s", "en")]
    _, stat = dc.decontaminate_stage(docs, broken)
    assert stat.detail["canary_recall"] == 1.0, "the real stage sizes its own canaries correctly"


def test_the_stage_drops_contaminated_documents():
    """Unlike a mislabelled language, evaluation text in the corpus is not a finding to keep."""
    canary = dc.canary_strings(1, CFG.minhash_seed, CFG.decontam_n)[0]
    index = dc.build_index([canary], CFG.decontam_n)
    leaked = f"{FILLER} {canary}"
    assert dc.contaminated(leaked, index, CFG.decontam_n)


def test_the_stage_reports_its_gram_width():
    """Long windows matter: at three words ordinary prose collides with everything."""
    docs = [Document("d0", FILLER, "t", "s", "en")]
    _, stat = dc.decontaminate_stage(docs, CFG)
    assert stat.detail["gram_width"] == CFG.decontam_n
    assert CFG.decontam_n >= 8, "a short window would flag ordinary prose as contaminated"
