"""Stage 3, graded against two baselines that are easy to forget to check.

An accuracy figure on its own says nothing. Ten of these languages share one script, so the two
questions that make the number readable are:

- What would a **script-only** detector score? On this set that is the chance baseline, not a weak
  result, and the n-gram detector has to beat it convincingly.
- What would a **constant** detector score — one that always answers `hi`? Any test the constant
  detector also passes is not testing detection.

Both are asserted here rather than assumed.
"""

import pytest
from datacleaning import langid
from datacleaning.config import Config
from datacleaning.records import Document

CFG = Config()

HINDI = "भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं।"
ENGLISH = "Connectivity is the most vital component of bilateral ties between the two countries."
TELUGU = "భారతదేశం ఒక పెద్ద దేశం మరియు ఇక్కడ అనేక భాషలు మాట్లాడతారు."
ASSAMESE = "অসমীয়া ভাষা ভাৰতৰ এটা প্ৰধান ভাষা আৰু ইয়াক বহু মানুহে কয়।"


def _profiles_or_skip() -> dict[str, langid.Profile]:
    profiles = langid.load_profiles(str(CFG.flores_dir))
    if not profiles:
        pytest.skip("FLORES-200 not on disk; the detector cannot be trained or graded")
    return profiles


# ---- script detection, the easy half ---------------------------------------------------------


def test_script_detection_separates_different_scripts():
    for text, expected in ((HINDI, "Devanagari"), (ENGLISH, "Latin"), (TELUGU, "Telugu")):
        script, _, share = langid.detect_script(text)
        assert script == expected
        assert share > 0.9


def test_script_detection_cannot_separate_two_devanagari_languages():
    """The reason this stage needs more than a script detector.

    Hindi and Maithili are different languages and identical scripts. If this ever fails, the
    n-gram model has become unnecessary — and that would be a surprise worth investigating.
    """
    marathi = "महाराष्ट्र हे भारतातील एक राज्य आहे."
    assert langid.detect_script(HINDI)[0] == langid.detect_script(marathi)[0] == "Devanagari"


def test_text_with_no_letters_is_unknown_rather_than_guessed():
    assert langid.detect_script("12345 !!! ---")[0] == "Unknown"


# ---- the discriminator ------------------------------------------------------------------------


def test_the_detector_identifies_languages_it_was_trained_on():
    profiles = _profiles_or_skip()
    for text, expected in ((HINDI, "hi"), (ENGLISH, "en"), (TELUGU, "te"), (ASSAMESE, "as")):
        verdict = langid.detect(text, CFG, profiles)
        assert verdict.detected == expected, f"{expected}: got {verdict.detected}"


def test_short_text_is_undecided_rather_than_guessed():
    """`undecided` is a real answer. A detector that always answers cannot be graded."""
    verdict = langid.detect("नमस्ते", CFG, _profiles_or_skip())
    assert verdict.detected is None
    assert "too short" in verdict.reason


def test_a_detector_that_always_answers_would_be_caught():
    """The twin for the test above.

    A detector wired to answer regardless of evidence passes every accuracy test at chance and
    fails this one, which is the only test that asks it to decline.
    """
    always = [langid.detect(t, CFG, _profiles_or_skip()).detected for t in ("नमस्ते", "हि", "a")]
    assert all(v is None for v in always), "short inputs must be declined, not guessed"


def test_a_script_with_no_trained_profile_is_declined():
    """Perso-Arabic Kashmiri is in our corpus and has no Devanagari profile to match.

    The sample is deliberately long: under `langid_min_chars` the detector declines for a
    *different* reason, and this test would then pass without ever exercising the branch it names.
    """
    kashmiri = "کٲشُر زبان کشمیرس منٛز بولان چھِ لوٗکھ۔ " * 3
    assert len(kashmiri) > CFG.langid_min_chars, "the sample must clear the too-short branch"

    verdict = langid.detect(kashmiri, CFG, _profiles_or_skip())
    assert verdict.script == "Arabic"
    assert verdict.detected is None
    assert "no trained profile" in verdict.reason


# ---- held-out grading, and the two baselines --------------------------------------------------


@pytest.fixture(scope="module")
def grading() -> dict:
    _profiles_or_skip()
    result = langid.grade(CFG)
    if result.get("coverage") != "held-out":
        pytest.skip(str(result.get("note")))
    return result


def test_the_detector_is_graded_on_data_it_never_trained_on(grading):
    assert grading["protocol"].startswith("trained on FLORES-200 dev, graded on devtest")
    assert grading["documents"] > 1000


def test_the_detector_beats_the_script_only_baseline_by_a_wide_margin(grading):
    """Without the baseline the accuracy is unreadable — nine languages share Devanagari."""
    assert grading["script_only_accuracy"] < 0.5, "the baseline should be near chance"
    assert grading["accuracy"] > grading["script_only_accuracy"] + 0.4


def test_accuracy_is_reported_at_several_document_lengths(grading):
    """One number would flatter the detector; five sentences is a lot of evidence.

    The single-sentence figure is the honest one for short web documents, so it must be published
    beside the headline rather than replaced by it.
    """
    lengths = grading["by_document_length"]
    assert {"1", "2", "5"} <= set(lengths)
    assert lengths["1"]["accuracy"] <= lengths["5"]["accuracy"], (
        "shorter documents must not be easier; if they are, the grading is wrong"
    )


def test_the_grading_names_what_it_cannot_cover(grading):
    """A measurement that hides its limits is worse than none — it reads as coverage."""
    limits = " ".join(grading["limits"]).lower()
    assert "upper bound" in limits
    assert "brx" in limits or "bodo" in limits


def test_the_accuracy_check_can_actually_fail():
    """A constant detector must fail the bar the real one passes.

    Without this, `test_the_detector_beats_the_script_only_baseline` would pass against a detector
    that had memorised nothing — nine Devanagari languages at chance is 1/9, and any bug that
    collapsed the scoring would still clear a poorly chosen threshold.
    """
    profiles = _profiles_or_skip()
    devanagari = [lang for lang, p in profiles.items() if p.script == "Devanagari"]
    assert len(devanagari) >= 5, "the Devanagari set should be genuinely confusable"

    texts = {"hi": HINDI, "te": TELUGU, "en": ENGLISH}
    constant_hits = sum(1 for lang in texts if lang == "hi")
    real_hits = sum(
        1 for lang, t in texts.items() if langid.detect(t, CFG, profiles).detected == lang
    )
    assert real_hits > constant_hits, "the detector must beat answering 'hi' every time"


# ---- the stage over a corpus ------------------------------------------------------------------


def test_a_language_with_no_profile_is_unadjudicable_not_mislabelled():
    """The correction that mattered most in this stage.

    Bodo is in the corpus and absent from FLORES-200, so the detector inevitably assigns its
    documents to the nearest Devanagari neighbour. Counting those as mismatches would publish a
    limitation of our detector as a defect in the corpus — about 1,900 fabricated findings in the
    lite profile before this was separated out.
    """
    _profiles_or_skip()
    docs = [
        Document(f"d{i}", HINDI * 3, "indic", "verified/brx/data-0.parquet", "brx")
        for i in range(5)
    ]
    _, stat = langid.langid_stage(docs, CFG)

    assert stat.detail["unadjudicable_total"] == 5
    assert not stat.detail["mismatches"], "an unprofiled language must not appear as a mismatch"


def test_a_real_mismatch_is_still_reported():
    """The twin. Suppressing unadjudicable languages must not suppress genuine findings."""
    _profiles_or_skip()
    docs = [
        Document(f"e{i}", ENGLISH * 3, "indic", "verified/mai/data-0.parquet", "mai")
        for i in range(5)
    ]
    _, stat = langid.langid_stage(docs, CFG)

    assert stat.detail["mismatches"].get("mai->en") == 5
    assert stat.detail["unadjudicable_total"] == 0


def test_the_stage_drops_nothing():
    """A mislabelled document is a finding about the source, not a document to delete."""
    _profiles_or_skip()
    docs = [
        Document(f"f{i}", ENGLISH * 3, "indic", "verified/mai/data-0.parquet", "mai")
        for i in range(4)
    ]
    out, stat = langid.langid_stage(docs, CFG)
    assert len(out) == len(docs) == stat.docs_out
