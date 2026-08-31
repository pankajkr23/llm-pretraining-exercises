"""What v1's truncation actually costs on the repo's own vocabulary."""

from embeddings.collisions import colliding_tokens, collisions_by_code, cosine, truncation_groups
from embeddings.config import KroneckerConfig

ONEHOT = KroneckerConfig(d_p=32, positions="onehot")
WRAP = KroneckerConfig(d_p=32, positions="wrap")


def test_the_frozen_vocabulary_has_the_measured_number_of_collisions(vocabulary):
    """407 tokens in 75 groups, 4.07% of the vocabulary. Derived from the bytes, no codec involved.

    Pinned rather than merely asserted non-zero: if the tokenizer is ever re-frozen this must be a
    visible, deliberate change, because every claim in the README about the cost of truncation is
    this number.
    """
    groups = truncation_groups(vocabulary, 32)
    assert colliding_tokens(vocabulary, 32) == 407
    assert len(groups) == 75
    assert max(len(g) for g in groups) == 83


def test_the_byte_count_and_the_code_count_agree(vocabulary):
    """Two independent routes to the same number. If they disagree, one of them is wrong and the
    published figure is unsupported."""
    subset = vocabulary[:2000]
    assert collisions_by_code(subset, ONEHOT) == colliding_tokens(subset, 32)


def test_wrap_removes_every_collision(vocabulary):
    """The point of wrapping. onehot conflates 407 tokens; wrap must conflate none."""
    subset = vocabulary[:2000]
    assert collisions_by_code(subset, ONEHOT) > 0
    assert collisions_by_code(subset, WRAP) == 0


def test_the_worked_example_is_a_real_collision():
    """Two Devanagari tokens differing at byte 44 -- inside v1's discarded region. The cosine is
    exactly 1.0: the codec cannot tell them apart at all, and nothing in training would say so."""
    a, b = "अंतर्राष्ट्रीयकरण".encode(), "अंतर्राष्ट्रीयता".encode()
    assert a[:32] == b[:32] and a != b
    assert cosine(a, b, ONEHOT) > 1 - 1e-9
    assert cosine(a, b, WRAP) < 0.99
