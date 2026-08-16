"""Stage 4 — nine rules, each proven to fire, and the language bias proven to be real.

Two things need holding down here. First, that each rule actually rejects what it claims to: a
cascade where one rule is wired to the wrong comparator still passes a "documents were dropped"
assertion. Second, that the English-versus-script-aware gap is a genuine property of the rules and
not an artefact of how we count.
"""

import pytest
from datacleaning import quality
from datacleaning.config import Config
from datacleaning.records import Document

CFG = Config()

GOOD_ENGLISH = (
    "The history of the region is long and well documented. Scholars have written about it "
    "for centuries, and the record is unusually complete. Much of that work survives in "
    "libraries that were built with the express purpose of preserving it. Readers who wish to "
    "consult the material will find it catalogued and indexed with considerable care, and the "
    "staff are helpful to those who ask. That combination is rarer than it ought to be."
)

GOOD_HINDI = (
    "भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं। हर राज्य की अपनी संस्कृति है। "
    "लोग अलग-अलग भाषाओं में बात करते हैं और उनके त्योहार भी अलग होते हैं। यह विविधता ही "
    "देश की सबसे बड़ी ताकत है और इसे बनाए रखना सबका काम है। शिक्षा के क्षेत्र में भी बहुत "
    "काम हुआ है और नए विद्यालय खोले गए हैं। इससे बच्चों को पढ़ने का अवसर मिला है।"
)


def _rule(results: list[quality.RuleResult], name: str) -> quality.RuleResult:
    return next(r for r in results if r.rule == name)


# ---- every rule fires on a document built to break it -----------------------------------------

BREAKERS: dict[str, str] = {
    "mean_word_length": "a b c d e f g h i j " * 12,
    "symbol_to_word_ratio": ("# " * 40) + GOOD_ENGLISH,
    "terminal_punctuation": " ".join(["no terminal punctuation anywhere in this document"] * 12),
    "duplicate_lines": "The same sentence again. " * 30,
    "top_bigram": ("spam spam " * 60) + " and a few other words here to pad the document out.",
    "stop_words": " ".join(["zzz qqq xxx yyy www vvv uuu ttt sss rrr"] * 8),
    "word_count": "Too short.",
    "ellipsis_lines": " ".join(["This line trails off..."] * 20),
    "bullet_lines": " ".join([f"- bullet item number {i}." for i in range(30)]),
}


@pytest.mark.parametrize("rule_name", sorted(BREAKERS))
def test_each_rule_rejects_a_document_built_to_break_it(rule_name: str):
    """Nine documents, each crafted to fail exactly the rule it is paired with."""
    results = quality.run_rules(BREAKERS[rule_name], CFG)
    assert not _rule(results, rule_name).passed, f"{rule_name} did not fire on its breaker"


def test_a_well_formed_document_passes_every_rule():
    """The twin for the whole table above.

    Without it, every rule test would pass against a cascade that rejects everything.
    """
    failures = [r.rule for r in quality.run_rules(GOOD_ENGLISH, CFG) if not r.passed]
    assert not failures, f"clean English prose should clear all nine rules, failed: {failures}"


def test_the_rule_table_has_exactly_nine_rules():
    """The session's cascade is nine rules. A tenth appearing silently would change the yield."""
    assert len({r.rule for r in quality.run_rules(GOOD_ENGLISH, CFG)}) == 9


# ---- the language bias -------------------------------------------------------------------------


def test_english_thresholds_reject_well_formed_hindi():
    """The finding: the published rules do not filter Indic text, they delete it.

    Terminal punctuation asks for `.`/`!`/`?`; Devanagari sentences end in the danda. Stop words
    asks for English function words, which Hindi never contains.
    """
    english_rules = quality.run_rules(GOOD_HINDI, CFG, script_aware=False)
    assert not _rule(english_rules, "terminal_punctuation").passed
    assert not _rule(english_rules, "stop_words").passed


def test_script_aware_thresholds_accept_the_same_hindi():
    """The twin. Without it, the test above could pass against text that is simply bad."""
    aware = quality.run_rules(GOOD_HINDI, CFG, script_aware=True)
    assert _rule(aware, "terminal_punctuation").passed, "the danda should close a sentence"
    assert _rule(aware, "stop_words").passed, "Hindi function words should count as stop words"


def test_script_awareness_does_not_change_the_verdict_on_english():
    """A fix that also changed English behaviour would be a different rule, not a fix."""
    english = quality.run_rules(GOOD_ENGLISH, CFG, script_aware=False)
    aware = quality.run_rules(GOOD_ENGLISH, CFG, script_aware=True)
    assert [r.passed for r in english] == [r.passed for r in aware]


def test_devanagari_word_length_counts_combining_marks():
    """The third language-neutrality failure, and the only one invisible in the rule text.

    Python's `\\w` and `str.isalnum` both skip Devanagari matras (category `Mn`), so measuring word
    length with them makes every Devanagari word look shorter than it is. Hindi prose scored 2.24
    against a floor of 3.0 and failed a rule it should clear comfortably.
    """
    import re

    word = "भारत"
    naive = len(re.findall(r"[^\W\d_]", word))
    correct = quality._word_length(word)
    assert correct > naive, f"matras must count toward word length: {correct} vs {naive}"

    result = _rule(quality.run_rules(GOOD_HINDI, CFG), "mean_word_length")
    assert result.passed, f"well-formed Hindi scored {result.observed}, outside {result.threshold}"
    assert result.observed >= 3.0


def test_the_word_length_fix_does_not_change_english():
    """The twin: a fix that also moved English would be a different rule, not a fix."""
    for word in ("history", "documented", "libraries"):
        assert quality._word_length(word) == len(word)


def test_only_the_two_language_sensitive_rules_are_marked_as_such():
    """The page renders this flag, so it must match which rules actually consult the script."""
    sensitive = {r.rule for r in quality.run_rules(GOOD_ENGLISH, CFG) if r.language_sensitive}
    assert sensitive == {"terminal_punctuation", "stop_words"}


# ---- the stage over a corpus --------------------------------------------------------------------


def _corpus() -> list[Document]:
    return [
        Document("en-1", GOOD_ENGLISH, "qa", "s", "en"),
        Document("en-2", GOOD_ENGLISH + " Extra sentence to differ.", "qa", "s", "en"),
        Document("hi-1", GOOD_HINDI, "indic", "s", "hi"),
        Document("hi-2", GOOD_HINDI + " एक और वाक्य यहाँ जोड़ा गया है।", "indic", "s", "hi"),
        Document("junk", "a b c d e f g h i j " * 12, "indic", "s", "hi"),
    ]


def test_the_stage_keeps_hindi_that_english_rules_would_delete():
    docs = _corpus()
    kept, stat = quality.quality_stage(docs, CFG)
    kept_ids = {d.doc_id for d in kept}

    assert {"hi-1", "hi-2"} <= kept_ids, "well-formed Hindi must survive the script-aware cascade"
    assert "junk" not in kept_ids
    assert stat.detail["dropped_english_thresholds"] > stat.detail["dropped_script_aware"]


def test_the_bias_measurement_can_actually_fail():
    """An all-English corpus must show no gap between the two rule sets.

    Without this, `extra_dropped_by_english_rules` could be any positive number produced by a
    counting bug and would still look like the finding.
    """
    docs = [Document(f"en-{i}", GOOD_ENGLISH, "qa", "s", "en") for i in range(4)]
    _, stat = quality.quality_stage(docs, CFG)
    assert stat.detail["extra_dropped_by_english_rules"] == 0


def test_the_classifier_gate_is_off_by_default_and_never_measured():
    """No FineWeb-Edu model exists here; publishing its yield would manufacture a measurement."""
    _, stat = quality.quality_stage(_corpus(), CFG)
    gate = stat.detail["classifier_gate"]
    assert gate["enabled"] is False
    assert gate["dropped"] == 0
    assert gate["provenance"] == "illustrative"


def test_the_classifier_gate_runs_when_asked():
    """The twin: the flag must actually do something, or 'off by default' is meaningless."""
    from dataclasses import replace

    cfg = replace(CFG, run_classifier_gate=True, classifier_threshold=5.0)
    kept, stat = quality.quality_stage(_corpus(), cfg)
    assert stat.detail["classifier_gate"]["enabled"] is True
    assert len(kept) <= 5


def test_the_residual_bias_is_reported_rather_than_implied():
    """Script-awareness narrows the gap; claiming it closes the gap would be the sin."""
    _, stat = quality.quality_stage(_corpus(), CFG)
    bias = stat.detail["residual_bias"]
    assert "survival_by_corpus" in bias
    assert "under-serves" in bias["note"] or "tuned on English" in bias["note"]
