"""Stage 5 — MinHash/LSH, and the guards that stop it deleting the wrong things.

A deduplicator is easy to test badly. "It removed some documents" passes against one that removes
everything; "it kept the distinct ones" passes against one that removes nothing. So every guard here
is paired with the opposite case, and the threshold tests assert both that a known duplicate is
caught **and** that it escapes when the threshold is raised.

The X16 guard is the odd one out: it asserts a fact about Python's regex rather than about our code,
because that fact is the reason we import exercise 03's tokenizer instead of writing our own.
"""

import re
from dataclasses import replace

import pytest
from datacleaning import dedup
from datacleaning.config import Config
from datacleaning.records import Document
from dataframework.shingles import normalise

CFG = Config()

BASE = (
    "The monsoon arrives in Kerala in early June and moves north across the subcontinent over "
    "the following six weeks. Farmers plan the sowing season around its arrival, and a late "
    "monsoon can cost a district its entire kharif crop. Meteorologists track it closely."
)
NEAR = BASE.replace("Meteorologists track it closely.", "Forecasters watch it very closely indeed.")
UNRELATED = (
    "Compiling a kernel module requires the matching kernel headers to be installed. On Debian "
    "systems the package is named linux-headers followed by the release string, which you can "
    "find by running uname -r in a terminal. Without it the build fails immediately."
)

HINDI = "भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं। हर राज्य की अपनी संस्कृति है।"


# ---- the maths ---------------------------------------------------------------------------------


def test_the_lsh_threshold_matches_the_published_formula():
    """FineWeb's preset is 14 bands of 8. The session quotes ~0.75; the formula gives 0.719."""
    assert dedup.lsh_threshold(14, 8) == pytest.approx((1 / 14) ** (1 / 8))
    assert dedup.lsh_threshold(14, 8) == pytest.approx(0.7190, abs=1e-4)


def test_the_s_curve_is_steep_around_the_threshold():
    """Banding works because the curve is steep, not because the threshold is exact."""
    threshold = dedup.lsh_threshold(14, 8)
    assert dedup.p_candidate(threshold - 0.2, 14, 8) < 0.15
    assert dedup.p_candidate(threshold + 0.2, 14, 8) > 0.95


def test_the_threshold_check_can_actually_fail():
    """A different banding must give a different threshold, or the formula is not being used."""
    assert dedup.lsh_threshold(6, 4) != dedup.lsh_threshold(14, 8)
    assert dedup.lsh_threshold(6, 4) == pytest.approx(0.6389, abs=1e-4)


def test_minhash_estimates_the_true_jaccard():
    """The property the whole stage rests on: slot agreement approximates set similarity."""
    a, b = dedup._permutations(CFG)
    sa = dedup.shingles(BASE, CFG.shingle_k)
    sb = dedup.shingles(NEAR, CFG.shingle_k)

    true = dedup.jaccard(sa, sb)
    sig_a, sig_b = dedup.signature(sa, a, b), dedup.signature(sb, a, b)
    estimate = float((sig_a == sig_b).mean())

    assert abs(estimate - true) < 0.20, f"estimate {estimate:.3f} vs true {true:.3f}"


def test_a_broken_permutation_family_loses_the_estimate():
    """The twin. Reuse one coefficient pair for every slot and the signature stops carrying
    information — every document agrees with every other on all 112 slots."""
    import numpy as np

    a = np.full(CFG.minhash_permutations, 7, dtype=np.uint64)
    b = np.full(CFG.minhash_permutations, 11, dtype=np.uint64)
    sig_a = dedup.signature(dedup.shingles(BASE, CFG.shingle_k), a, b)
    sig_b = dedup.signature(dedup.shingles(NEAR, CFG.shingle_k), a, b)

    degenerate = float((sig_a == sig_b).mean())
    assert degenerate in (0.0, 1.0), (
        "a degenerate family has no resolution: every slot holds the same value, so two documents "
        "either agree on all 112 or on none"
    )

    # And the real family does have resolution on the same pair.
    ra, rb = dedup._permutations(CFG)
    real = float(
        (
            dedup.signature(dedup.shingles(BASE, CFG.shingle_k), ra, rb)
            == dedup.signature(dedup.shingles(UNRELATED, CFG.shingle_k), ra, rb)
        ).mean()
    )
    true = dedup.jaccard(
        dedup.shingles(BASE, CFG.shingle_k), dedup.shingles(UNRELATED, CFG.shingle_k)
    )
    assert abs(real - true) < 0.2, "the real family should track the true similarity"


# ---- determinism -------------------------------------------------------------------------------


def test_shingle_hashing_is_stable_across_processes():
    """Python's built-in `hash()` on strings is randomised per interpreter.

    Using it would make bucketing — and therefore which documents get deleted — drift between runs,
    which would quietly void the pipeline's reproducibility claim. The expected value is hard-coded
    so a change of hash function has to be deliberate.
    """
    assert dedup._stable_hash("the monsoon arrives in kerala") == 10778432804417743726


def test_the_same_corpus_gives_the_same_answer_twice():
    docs = [Document(f"d{i}", t, "t", "s", "en") for i, t in enumerate([BASE, NEAR, UNRELATED])]
    first, _ = dedup.dedup_stage(list(docs), CFG)
    second, _ = dedup.dedup_stage(list(docs), CFG)
    assert [d.doc_id for d in first] == [d.doc_id for d in second]


# ---- catching duplicates, and only duplicates ---------------------------------------------------


def test_a_known_near_duplicate_pair_is_caught_at_the_configured_threshold():
    docs = [
        Document("a", BASE, "t", "s", "en"),
        Document("b", NEAR, "t", "s", "en"),
        Document("c", UNRELATED, "t", "s", "en"),
    ]
    kept, stat = dedup.dedup_stage(docs, CFG)
    kept_ids = {d.doc_id for d in kept}

    assert len(kept_ids) == 2, "the near-duplicate pair should collapse to one"
    assert "c" in kept_ids, "the unrelated document must survive"
    assert stat.detail["near"]["docs_removed"] == 1


def test_the_same_pair_escapes_when_the_threshold_is_raised():
    """The twin, and the point of the chapter: the threshold is a decision, not a setting.

    A catcher that catches everything is not a catcher. Raising rows per band raises the similarity
    required, and the same real pair falls out of the candidate set.
    """
    strict = replace(CFG, bands=4, rows_per_band=24)
    assert dedup.lsh_threshold(4, 24) > dedup.lsh_threshold(14, 8)

    docs = [Document("a", BASE, "t", "s", "en"), Document("b", NEAR, "t", "s", "en")]
    kept, stat = dedup.dedup_stage(docs, strict)
    assert len(kept) == 2, "at a stricter threshold this pair is no longer a duplicate"
    assert stat.detail["near"]["docs_removed"] == 0


def test_an_exact_duplicate_is_removed_by_the_cheap_pass_first():
    docs = [
        Document("a", BASE, "t", "s", "en"),
        Document("b", BASE, "t", "s", "en"),
        Document("c", UNRELATED, "t", "s", "en"),
    ]
    kept, stat = dedup.dedup_stage(docs, CFG)
    assert len(kept) == 2
    assert stat.detail["exact"]["docs_removed"] == 1


def test_unrelated_documents_are_never_merged():
    """The twin for every removal test above."""
    docs = [
        Document("a", BASE, "t", "s", "en"),
        Document("b", UNRELATED, "t", "s", "en"),
        Document("c", HINDI * 3, "t", "s", "hi"),
    ]
    kept, stat = dedup.dedup_stage(docs, CFG)
    assert len(kept) == 3
    assert stat.detail["near"]["docs_removed"] == 0


GRADED_TEXT = (
    "the monsoon arrives in kerala in early june and moves north across the subcontinent over "
    "the following six weeks farmers plan the sowing season around its arrival and a late monsoon "
    "can cost a district its entire kharif crop meteorologists track it very closely each year "
    "because the timing of the rains decides whether the harvest succeeds or fails entirely"
)


def _graded_family() -> list[Document]:
    """A base document plus variants of steadily decreasing similarity.

    Tuned so that some pairs land just *below* the 0.719 threshold while still colliding in at
    least one band — which is the only situation where the true-Jaccard check does any work.
    """
    words = GRADED_TEXT.split()
    base = " ".join(words)
    docs = [Document("base", base, "t", "s", "en")]
    for n in range(1, 14):
        docs.append(Document(f"v{n}", " ".join(words[:-n] + ["zulu"] * n), "t", "s", "en"))
    return docs


def test_lsh_proposes_and_the_true_jaccard_rejects_some_of_its_proposals():
    """Banding is a recall device, not a verdict.

    An earlier version of this test used three documents, LSH proposed no false candidates at all,
    and deleting the similarity check entirely left every assertion green — a guard that could not
    fail. This family is built so that LSH genuinely over-proposes and the check genuinely
    rejects.
    """
    _, stat = dedup.dedup_stage(_graded_family(), CFG)
    near = stat.detail["near"]

    assert near["candidate_pairs"] > 0, "LSH should propose candidates for this family"
    assert near["false_candidate_pairs"] > 0, (
        "at least one candidate must fall below the similarity threshold, or the check is untested"
    )
    assert near["verified_pairs"] < near["candidate_pairs"]


def test_a_pair_below_the_threshold_is_reported_as_a_near_miss():
    """The rejected candidates are published, not silently discarded.

    The page shows them beside the confirmed duplicates, because "what did it nearly delete" is as
    informative as "what did it delete".
    """
    _, stat = dedup.dedup_stage(_graded_family(), CFG)
    misses = stat.detail["example_near_misses"]
    assert misses, "near misses should be recorded for the page"
    threshold = dedup.lsh_threshold(CFG.bands, CFG.rows_per_band)
    assert all(m["jaccard"] < threshold for m in misses)


def test_clusters_chain_transitively_and_that_is_reported():
    """A property worth knowing rather than discovering later.

    Deduplication here is single-linkage: if A~B and B~C, all three collapse even when A and C are
    below the threshold. On the graded family that merges the whole chain into one document. This
    is standard and defensible, but it means "documents removed" is not the same as "pairs above
    the threshold", so both numbers are published.
    """
    kept, stat = dedup.dedup_stage(_graded_family(), CFG)
    assert len(kept) < 3, "a similarity chain should collapse transitively"
    assert stat.detail["largest_cluster"] > 2


# ---- correction X16, run rather than read --------------------------------------------------------


def test_pythons_word_regex_shatters_indic_words():
    """Why we import exercise 03's tokenizer instead of writing `re.findall(r'\\w+', text)`.

    A combining mark is not a word character to Python's `\\w`, so every Devanagari word splits at
    every vowel sign. Shingles built from the fragments would compare fragment overlap rather than
    word overlap — silently, and plausibly.
    """
    naive = re.findall(r"\w+", HINDI)
    correct = normalise(HINDI)
    assert len(naive) > len(correct) * 1.5, (
        f"expected the naive regex to over-split: {len(naive)} vs {len(correct)}"
    )


def test_our_shingles_use_the_indic_safe_tokenizer():
    """The twin: prove the fix is actually wired in, not merely available."""
    words = normalise(HINDI)
    expected = max(len(words) - CFG.shingle_k + 1, 1)
    assert len(dedup.shingles(HINDI, CFG.shingle_k)) == expected


def test_a_document_shorter_than_the_window_still_shingles():
    """An empty shingle set would make every short document identical to every other one."""
    assert len(dedup.shingles("three short words", 5)) == 1
    assert dedup.shingles("", 5) == set()
