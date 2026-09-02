"""The frozen 10k vocabulary, loaded once per topic.

Every count in this exercise -- 407 collisions, a 121-byte maximum, the recovery rate -- is a
property of a SPECIFIC vocabulary, so the tests use the one the repo froze rather than a fresh one.
"""

import numpy as np
import pytest
from datacleaning.config import OUR_TOKENIZER
from datacleaning.tokens import load_tokenizer


@pytest.fixture(scope="session")
def vocabulary() -> list[bytes]:
    """Every token of the frozen tokenizer, as UTF-8 bytes, in id order."""
    tok = load_tokenizer(str(OUR_TOKENIZER))
    return [tok.id_to_token(i).encode() for i in range(tok.get_vocab_size())]


@pytest.fixture(scope="session")
def sample(vocabulary: list[bytes]) -> list[bytes]:
    """A fixed 600-token sample. Small enough for a fast suite, real enough to be evidence."""
    idx = np.random.default_rng(0).choice(len(vocabulary), 600, replace=False)
    return [vocabulary[i] for i in sorted(idx)]


@pytest.fixture(scope="session")
def projection() -> np.ndarray:
    """A `(D, 384)` sensing matrix with unit-norm atoms, seeded."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((256 * 32, 384))
    return w / np.linalg.norm(w, axis=1, keepdims=True)
