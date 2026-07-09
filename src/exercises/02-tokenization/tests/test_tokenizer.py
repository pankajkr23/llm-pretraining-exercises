"""Integration: train a tiny shared BPE offline and check it round-trips."""

import pytest
from tokenization.tokenizer import count_tokens, train_bpe


@pytest.mark.integration
def test_tiny_bpe_trains_encodes_and_round_trips():
    corpora = {
        "en": "india is a country in south asia " * 40,
        "hi": "भारत एक देश है " * 40,
    }
    tok = train_bpe(corpora, vocab_size=300, weights={"en": 1, "hi": 2})
    assert tok.get_vocab_size() <= 300
    assert count_tokens(tok, "india") > 0
    # byte-level BPE decodes losslessly
    assert tok.decode(tok.encode("india").ids) == "india"
