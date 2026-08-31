"""The parameter arithmetic, including where this architecture stops paying for itself."""

from embeddings.budget import budget, crossover


def test_the_v2_head_has_no_term_in_the_vocabulary():
    """The headline. Hold everything but V fixed and the head's size must not move at all."""
    sizes = {budget(v, 768, d_p=32, n_buckets=8192).v2_total for v in (1_000, 10_000, 10_000_000)}
    assert len(sizes) == 1


def test_v1_is_larger_than_the_tied_baseline_it_replaces():
    """The reason v1 "could not be used fully": on GPT-2 124M its 91% input-side saving is entirely
    eaten by the head it forces you to untie."""
    b = budget(50_257, 768, d_p=32)
    assert b.dense_tied == 38_597_376
    assert b.v1 == 44_888_832
    assert b.v1 > b.dense_tied


def test_the_crossover_is_exactly_the_code_width():
    """Below `V = 256 * d_p` this architecture COSTS parameters. Verified at the boundary, not
    asserted: an off-by-one here flips the recommendation for small vocabularies."""
    for d_p in (32, 128):
        v = crossover(d_p)
        assert v == 256 * d_p
        bare = budget(v, 768, d_p=d_p).v2 - 768 * 768
        assert bare == budget(v, 768, d_p=d_p).dense_tied + 1


def test_the_saving_at_a_million_tokens():
    """The assignment's "vocab of 1M without any issues", as arithmetic."""
    b = budget(1_000_000, 768, d_p=32)
    assert b.dense_tied == 768_000_000
    assert b.saving > 100


def test_a_small_vocabulary_is_the_losing_case(vocabulary):
    """The twin. Every test above is about the regime where this wins; below the crossover it must
    honestly report a LOSS, or the guard is only ever pointed at good news."""
    b = budget(1_000, 768, d_p=32)
    assert b.saving < 1.0
