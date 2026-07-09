"""Tests for the hand-written BPE — fast unit checks plus one offline train/round-trip."""

import pytest
from tokenization.bpe_scratch import UNK_TOKEN, WORD_PREFIX, ScratchBPE


def _tiny_english() -> dict[str, str]:
    return {"en": "low lower lowest newer newest wider widest " * 30}


def test_pre_tokenize_marks_word_starts_and_normalizes():
    tok = ScratchBPE(normalization="NFKC")
    # NFKC folds the compatibility ligature ﬁ -> fi; whitespace splitting yields two word units.
    tok.train({"en": "ﬁre wood " * 20}, vocab_size=60, weights={"en": 1})
    tokens = tok.encode("ﬁre wood").tokens
    assert "".join(tokens).replace(WORD_PREFIX, " ").strip() == "fire wood"
    assert tokens[0].startswith(WORD_PREFIX)  # first token carries the word-start marker


def test_frequent_words_compress_below_char_count():
    tok = ScratchBPE(normalization=None)
    tok.train(_tiny_english(), vocab_size=300, weights={"en": 1})
    assert tok.merges  # merges were learned
    # A frequent training word collapses to far fewer tokens than its characters (incl. the ▁).
    assert len(tok.encode("lower").ids) < len("▁lower")
    # Every learned merge produces a multi-character token in the vocabulary.
    vocab = tok.get_vocab()
    assert any(len(tok_str) > 1 for tok_str in vocab)


def test_vocab_size_is_respected_and_ids_are_contiguous():
    tok = ScratchBPE(normalization=None)
    tok.train(_tiny_english(), vocab_size=120, weights={"en": 1})
    assert tok.get_vocab_size() <= 120
    ids = sorted(tok.get_vocab().values())
    assert ids == list(range(len(ids)))  # 0..N-1, no gaps
    assert tok.get_vocab()[UNK_TOKEN] == 0


def test_training_is_deterministic():
    a, b = ScratchBPE(normalization=None), ScratchBPE(normalization=None)
    a.train(_tiny_english(), vocab_size=200, weights={"en": 1})
    b.train(_tiny_english(), vocab_size=200, weights={"en": 1})
    assert a.merges == b.merges
    assert a.get_vocab() == b.get_vocab()


def test_unknown_character_maps_to_unk():
    tok = ScratchBPE(normalization=None)
    tok.train(_tiny_english(), vocab_size=200, weights={"en": 1})
    unk_id = tok.get_vocab()[UNK_TOKEN]
    # "本" never appears in the training corpus, so it cannot be represented by any known symbol.
    assert unk_id in tok.encode("本").ids


def test_weighting_shifts_merges_toward_upsampled_language():
    text = {"en": "aaaa " * 50, "hi": "कककक " * 50}
    heavy_hi = ScratchBPE(normalization=None)
    heavy_hi.train(text, vocab_size=60, weights={"en": 1, "hi": 10})
    # Upsampling Hindi 10x makes the Devanagari pair win merges its flat counterpart may not.
    assert ("क", "क") in heavy_hi.merges


def test_save_load_round_trips(tmp_path):
    tok = ScratchBPE(normalization="NFKC")
    tok.train(_tiny_english(), vocab_size=200, weights={"en": 1})
    path = tmp_path / "scratch.json"
    tok.save(path)
    reloaded = ScratchBPE.load(path)
    assert reloaded.merges == tok.merges
    assert reloaded.encode("lowest newer").ids == tok.encode("lowest newer").ids


@pytest.mark.integration
def test_multilingual_train_encodes_and_round_trips():
    corpora = {
        "en": "india is a country in south asia " * 40,
        "hi": "भारत एक देश है " * 40,
    }
    tok = ScratchBPE(normalization="NFKC")
    tok.train(corpora, vocab_size=400, weights={"en": 1, "hi": 2})
    assert tok.get_vocab_size() <= 400
    for text in corpora.values():
        assert len(tok.encode(text).ids) > 0
    # Char-level covers every training character, so encode -> decode reconstructs the words.
    assert tok.decode(tok.encode("india है").ids) == "india है"
