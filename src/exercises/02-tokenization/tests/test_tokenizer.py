"""Integration: the byte-level path still trains and round-trips, and save/load survives it."""

import pytest
from tokenization.ablate import Spec, train_spec
from tokenization.tokenizer import count_tokens, save
from tokenizers import Tokenizer

_CORPORA = {
    "en": "india is a country in south asia " * 40,
    "hi": "भारत एक देश है " * 40,
}


@pytest.mark.integration
def test_byte_level_bpe_trains_encodes_and_round_trips():
    spec = Spec(algo="bpe", level="byte", normalization=None, vocab_size=300, weighting="flat")
    tok = train_spec(spec, _CORPORA)
    assert tok.get_vocab_size() <= 300
    assert count_tokens(tok, "india") > 0
    # Byte-level covers every possible input, so it decodes losslessly.
    assert tok.decode(tok.encode("india").ids) == "india"


@pytest.mark.integration
def test_saved_tokenizer_reloads_to_identical_ids(tmp_path):
    spec = Spec(algo="bpe", level="char", normalization="NFKC", vocab_size=300, weighting="flat")
    tok = train_spec(spec, _CORPORA)
    path = tmp_path / "tokenizer.json"
    save(tok, path)
    assert Tokenizer.from_file(str(path)).encode("भारत").ids == tok.encode("भारत").ids
