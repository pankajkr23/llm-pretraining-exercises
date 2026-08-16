"""Our tokenizer, and the gate that stops an unusable count being published.

The central claim of `tokens.py` is that a token count is only a measurement when the tokenizer
could actually read the script. These tests hold that claim down from both sides — the in-vocabulary
languages must pass, the out-of-vocabulary ones must fail, and each check is paired with a twin
proving the check can fail at all.

The twins matter more than usual here. `test_our_tokenizer_reads_devanagari` would pass against a
tokenizer that returned `[UNK]` for nothing whatsoever, and `test_the_gate_rejects...` would pass
against a gate wired to reject everything. Neither test means anything alone.
"""

import pytest
from datacleaning import tokens
from datacleaning.config import Config
from datacleaning.records import Figure

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

HINDI = "मैथिली भाषा भारतक एक प्रमुख भाषा थिक।"
ASSAMESE = "অসমীয়া ভাষা ভাৰতৰ এটা প্ৰধান ভাষা।"
ENGLISH = "Connectivity is the most vital component of bilateral ties."


def _flores_or_skip() -> dict[str, tokens.TokenCount]:
    graded = tokens.flores_fertility()
    if not graded:
        pytest.skip("FLORES-200 dev not on disk; run exercise 03's fetch first")
    return graded


# ---- the tokenizer loads at all ------------------------------------------------------------


def test_our_tokenizer_loads_and_is_the_submitted_vocabulary():
    tok = tokens.load_tokenizer()
    assert tok.get_vocab_size() == 10_000
    assert tokens.tokenizer_name() == "ours/s02-bpe-10000"


def test_a_missing_tokenizer_is_named_rather_than_crashing_obscurely():
    with pytest.raises(FileNotFoundError, match="exercise 02"):
        tokens.load_tokenizer("/nonexistent/tokenizer.json")


# ---- what the vocabulary can and cannot read -------------------------------------------------


def test_our_tokenizer_reads_english_and_devanagari():
    for text in (ENGLISH, HINDI):
        count = tokens.count(text)
        assert count.tokens > 0
        assert count.unk == 0, f"unexpected [UNK] in {text!r}"
        assert count.usable


def test_our_tokenizer_cannot_read_bengali_script():
    """The finding that decided the corpus: S2 never saw Bengali script.

    If this ever passes, the tokenizer was retrained and `sources.py` should be revisited — the
    out-of-vocabulary probe would no longer be out of vocabulary.
    """
    count = tokens.count(ASSAMESE)
    assert count.unk_share > 0.5, f"expected mostly [UNK], got {count.unk_share:.1%}"
    assert not count.usable


def test_the_unk_check_can_actually_fail():
    """Without this, the two tests above would both pass against a tokenizer with no `[UNK]` at all.

    Pure ASCII must come back clean, so a detector that flagged everything as unreadable is caught.
    """
    count = tokens.count("plain ascii text with nothing exotic in it at all")
    assert count.unk == 0
    assert count.usable


# ---- the publication gate ---------------------------------------------------------------------


def test_an_unusable_count_is_published_as_unknown_not_as_a_number():
    """The gate's whole purpose: a count that is 84% `[UNK]` must not reach the page as a number."""
    figure = tokens.count(ASSAMESE).as_figure()
    assert isinstance(figure, Figure)
    assert figure.value is None
    assert figure.provenance == "unknown"
    assert "cannot read this script" in figure.source


def test_a_usable_count_is_published_as_measured():
    figure = tokens.count(ENGLISH).as_figure()
    assert figure.value and figure.value > 0
    assert figure.provenance == "measured"
    assert "ours/s02-bpe" in figure.source


def test_the_publication_gate_can_actually_fail():
    """A gate that let everything through would pass the test above.

    Hand-build counts either side of the threshold and assert the verdict flips.
    """
    just_under = tokens.TokenCount(tokens=1000, words=400, unk=40, tokenizer="t")  # 4%
    just_over = tokens.TokenCount(tokens=1000, words=400, unk=60, tokenizer="t")  # 6%
    assert just_under.usable and just_under.as_figure().value == 1000
    assert not just_over.usable and just_over.as_figure().value is None


# ---- the memo must not change any answer ------------------------------------------------------


def test_the_memo_returns_the_same_answer_as_a_cold_count():
    """The memo is a 13x speedup; it must be invisible in the results."""
    texts = [ENGLISH, HINDI, ASSAMESE, ENGLISH]
    tokens.clear_memo()
    cold = tokens.count_many(texts)
    warm = tokens.count_many(texts)
    assert (cold.tokens, cold.words, cold.unk) == (warm.tokens, warm.words, warm.unk)


def test_changed_text_is_recounted_rather_than_served_from_the_memo():
    """The memo keys on the text's hash, so editing a document must invalidate it.

    A memo keyed on a document id would silently serve a stale count after every cleaning stage —
    which is precisely where this pipeline edits text.
    """
    tokens.clear_memo()
    short = tokens.count_many(["one two three"])
    longer = tokens.count_many(["one two three four five six seven eight"])
    assert longer.tokens > short.tokens


def test_counting_many_agrees_with_counting_one_at_a_time():
    texts = [ENGLISH, HINDI]
    tokens.clear_memo()
    batch = tokens.count_many(texts)
    singly = [tokens.count(t) for t in texts]
    assert batch.tokens == sum(c.tokens for c in singly)
    assert batch.unk == sum(c.unk for c in singly)


# ---- the spread table, which is a published finding -------------------------------------------


def test_the_tokenizer_spread_is_large_enough_to_be_the_point():
    """ "90M tokens" is a fact about a corpus *and a tokenizer*. The spread is the evidence."""
    table = tokens.spread_table()
    if not table["rows"].get("mni", {}).get(tokens.REFERENCE_TOKENIZERS[0]):
        pytest.skip("exercise 03's fertility record not on disk")
    assert table["spread"]["mni"] > 5, "Manipuri should swing several-fold across tokenizers"
    assert table["spread"]["en"] < 2, "English should be stable across tokenizers"


def test_flores_grading_separates_readable_scripts_from_unreadable_ones():
    graded = _flores_or_skip()
    for lang in ("en", "hi", "mai", "te"):
        assert graded[lang].usable, f"{lang} should be readable by our vocabulary"
    for lang in ("as", "bn", "mni"):
        assert not graded[lang].usable, f"{lang} should be out of vocabulary"


def test_unreadable_languages_are_reported_worst_first():
    _flores_or_skip()
    bad = tokens.unreadable_languages()
    assert set(bad) == {"as", "bn", "mni"}
    assert list(bad.values()) == sorted(bad.values(), reverse=True)


def test_the_readability_split_can_actually_fail():
    """Both tests above would pass against a `usable` property hard-wired to a language list.

    Drive the property from numbers instead and assert it responds to them.
    """
    readable = tokens.TokenCount(tokens=100, words=50, unk=0, tokenizer="t")
    unreadable = tokens.TokenCount(tokens=100, words=50, unk=84, tokenizer="t")
    assert readable.usable != unreadable.usable


def test_a_config_with_defaults_names_our_tokenizer():
    cfg = Config()
    assert cfg.tokenizer_path.name == "tokenizer.json"
    assert cfg.tokenizer_path.parent.parent.name == "02-tokenization"
