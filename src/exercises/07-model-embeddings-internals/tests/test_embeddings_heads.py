"""The tied head: the scale fix, the additivity lock, and the term that breaks it.

Module-level `importorskip`, so this file collects NOTHING without torch. That is deliberate and it
is why the path is registered in `tests/test_ci_shards_cover_everything.py`'s
`OPTIONAL_DEPENDENCY_GATES` and in the `train` job of `ci.yml`: a file that collects zero tests is
indistinguishable from a file with nothing in it, and this repo has already lost 46 tests that way.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embeddings.config import KroneckerConfig  # noqa: E402
from embeddings.heads import KroneckerEmbedding, LockBreaker, TiedHead  # noqa: E402

CFG = KroneckerConfig(d_p=32, d_model=256, positions="wrap", n_buckets=8192)

#: The four real vocabulary tokens that form a rectangle -- equal length, differing at two
#: positions, with the (position, byte) multisets cancelling. The tie CANNOT assign them
#: independent logits.
RECTANGLE = (b'"\n', b'".', b")\n", b").")


@pytest.fixture(scope="module")
def head_pair(vocabulary):
    torch.manual_seed(0)
    plain = TiedHead(vocabulary, CFG, lock_breaker=None)
    torch.manual_seed(0)
    ngram = TiedHead(vocabulary, CFG, lock_breaker="ngram")
    return plain, ngram


def _lock(model, ids, h):
    """The rectangle residual RELATIVE to the logit scale.

    Absolute thresholds silently encode an init scale: the MLP's break reads 5.7e-03 at init 0.05
    and 4.67 at init 0.5, on logit scales of 2.50 and 70.5. Both are the same fact. Dividing by the
    logit scale asks the question that actually matters -- is the constraint binding? -- and cannot
    be passed or failed by turning a knob.
    """
    a, b, c, d = ids
    lg = model(h)
    residual = (lg[:, a] - lg[:, b] - lg[:, c] + lg[:, d]).abs().max()
    return (residual / lg.abs().max()).item()


@pytest.fixture(scope="module")
def rect_ids(vocabulary):
    ids = [vocabulary.index(t) for t in RECTANGLE]
    assert len({vocabulary[i] for i in ids}) == 4
    return ids


def test_the_induced_embedding_matches_the_literal_code(vocabulary):
    """`E = K W` computed sparsely must equal building the code and multiplying. The whole point is
    that no `V x D` matrix is ever formed, so the shortcut has to be provably the same map."""
    from embeddings.codec import code

    torch.manual_seed(0)
    emb = KroneckerEmbedding(vocabulary[:200], CFG)
    literal = np.stack([code(bs, CFG) for bs in vocabulary[:200]]) @ emb.w.detach().numpy()
    assert np.abs(emb.induced().detach().numpy() - literal).max() < 1e-4


def test_the_output_scale_keeps_the_initial_loss_near_ln_v(vocabulary):
    """A naive tie starts at loss ~94 against ln(V)=9.21, because z-norm makes the induced rows 49x
    too large. One learned scalar is the entire fix, and this is what it is for."""
    torch.manual_seed(0)
    head = TiedHead(vocabulary, CFG, lock_breaker=None)
    h = torch.randn(64, CFG.d_model) * 0.5
    target = torch.randint(0, len(vocabulary), (64,))
    loss = torch.nn.functional.cross_entropy(head(h), target).item()
    assert loss < 3 * np.log(len(vocabulary))


def test_the_plain_tie_is_locked_on_a_real_rectangle(head_pair, rect_ids):
    """`logit_A - logit_B - logit_C + logit_D = 0` for EVERY hidden state. Not a training
    difficulty -- a property of the function class, and the reason the plain tie loses to v1."""
    plain, _ = head_pair
    h = torch.randn(8, CFG.d_model)
    assert _lock(plain, rect_ids, h) < 1e-6, "the plain tie must be exactly locked"


def test_a_d_by_d_transform_cannot_break_the_lock(vocabulary, rect_ids):
    """`<h, A E> = <A^T h, E>` reparameterises `h`, so it cannot change what is expressible. This is
    the test that corrects the rationale the transform was first added with."""
    torch.manual_seed(0)
    head = TiedHead(vocabulary, CFG, lock_breaker=None, transform=True)
    with torch.no_grad():
        head.transform.copy_(torch.randn(CFG.d_model, CFG.d_model))
    h = torch.randn(8, CFG.d_model)
    assert _lock(head, rect_ids, h) < 1e-6, "a transform on h cannot change the class"


def test_the_lock_breakers_start_as_exact_no_ops(head_pair):
    """Zero-initialised, so step 0 is bit-identical to the plain tie and any later gap is
    attributable to the term rather than to a different starting point."""
    plain, ngram = head_pair
    h = torch.randn(8, CFG.d_model)
    assert (ngram(h) - plain(h)).abs().max().item() == 0.0


@pytest.mark.parametrize("mode", ["ngram", "mlp"])
def test_a_nonzero_lock_breaker_breaks_the_lock(vocabulary, rect_ids, mode):
    """Both modes CAN express a non-zero lock. Only one of them helps in training (-0.412 nats
    against -0.002), which is the finding: expressivity is necessary and not sufficient."""
    torch.manual_seed(0)
    head = TiedHead(vocabulary, CFG, lock_breaker=mode)
    with torch.no_grad():
        for p in head.breaker.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    h = torch.randn(8, CFG.d_model)
    assert _lock(head, rect_ids, h) > 1e-3


def test_the_head_holds_no_vocabulary_sized_parameter(vocabulary):
    """Two vocabularies of very different size must give byte-identical parameter counts."""

    def n_params(vocab):
        torch.manual_seed(0)
        return sum(p.numel() for p in TiedHead(vocab, CFG).parameters() if p.requires_grad)

    assert n_params(vocabulary[:500]) == n_params(vocabulary)


def test_the_ngram_hash_is_reproducible_across_processes(vocabulary):
    """`hash()` is randomised per process for bytes; `zlib.crc32` is not. Two constructions must
    bucket identically or nothing in this exercise reproduces."""
    a = LockBreaker(vocabulary[:300], CFG.d_model, "ngram", 1024)
    b = LockBreaker(vocabulary[:300], CFG.d_model, "ngram", 1024)
    assert torch.equal(a.grams.indices(), b.grams.indices())


def test_an_unknown_lock_breaker_is_rejected(vocabulary):
    """The twin: a typo must fail rather than silently giving the plain tie back."""
    with pytest.raises(ValueError, match="unknown lock-breaker"):
        LockBreaker(vocabulary[:50], CFG.d_model, "attention")


def test_the_byte_head_scores_only_real_vocabulary_entries(vocabulary):
    """The comparability guard, and the first thing a reviewer attacks.

    A factorised byte head can spend probability on strings that are not words, so its raw summed
    cross-entropy is not comparable with a dense softmax's. This head scores ONLY real vocabulary
    rows, so `cross_entropy` renormalises over V and the reported NLL is like-for-like.
    """
    from embeddings.heads import ByteHead

    torch.manual_seed(0)
    vocab = vocabulary[:400]
    head = ByteHead(vocab, CFG)
    scores = head(torch.randn(6, CFG.d_model))
    assert scores.shape == (6, len(vocab))

    probs = torch.softmax(scores, dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(6), atol=1e-5)

    # No parameter mentions V: only the d_p x 257 projection is learned.
    assert sum(p.numel() for p in head.parameters() if p.requires_grad) == (
        CFG.d_model * CFG.d_p * 257
    )


def test_without_a_stop_symbol_a_prefix_always_outscores_its_extensions(vocabulary):
    """The exact defect, and it is an ordering no training can overturn.

    `score = sum_{p<L} log P_p(b_p)` adds a negative term per byte, so extending a token can only
    lower its score. A token that is a strict prefix of another therefore wins ALWAYS -- for every
    weight setting, not on average. `the` can never lose to `there`.

    Checked against random weights rather than a convenient initialisation, and then checked that
    the stop symbol makes the ordering breakable, which is the whole reason it is there.
    """
    from embeddings.heads import END_OF_TOKEN, ByteHead

    pairs = []
    seen = set(vocabulary)
    for bs in vocabulary[:4000]:
        if 2 <= len(bs) <= 8:
            longer = next((o for o in seen if o != bs and o.startswith(bs) and len(o) <= 12), None)
            if longer is not None:
                pairs.append((bs, longer))
        if len(pairs) >= 20:
            break
    assert pairs, "no prefix pairs found in the vocabulary — the test cannot run"

    vocab = sorted({b for pair in pairs for b in pair})
    index = {b: i for i, b in enumerate(vocab)}
    head = ByteHead(vocab, CFG)

    sel = head.selector.to_dense()
    eot_cols = [p * head.n_symbols + END_OF_TOKEN for p in range(CFG.d_p)]
    sel_no_eot = sel.clone()
    sel_no_eot[:, eot_cols] = 0.0

    torch.manual_seed(0)
    beaten_without = 0
    for _ in range(8):
        per_slot = torch.randn(CFG.d_p, head.n_symbols)
        logp = torch.nn.functional.log_softmax(per_slot, dim=-1).reshape(-1)
        s_without = sel_no_eot @ logp
        for short, long_ in pairs:
            beaten_without += int(s_without[index[long_]] > s_without[index[short]])
    total = 8 * len(pairs)
    assert beaten_without == 0, (
        f"without a stop symbol a prefix must NEVER lose; it lost {beaten_without}/{total}"
    )

    # With the stop symbol the ordering becomes EXPRESSIBLE. Random weights do not flip it -- each
    # extra byte costs about ln(257) = 5.5 nats and a random logit gap is worth a fraction of that
    # -- so the honest test is whether the head CAN represent "the longer token wins", not whether
    # it stumbles into it. Build the weights that say so: every byte of the longer token near
    # certain, and stopping at the prefix's length near impossible.
    flipped = 0
    for short, long_ in pairs:
        per_slot = torch.full((CFG.d_p, head.n_symbols), -8.0)
        for p, byte in enumerate(long_[: CFG.d_p]):
            per_slot[p, byte] = 8.0
        if len(long_) < CFG.d_p:
            per_slot[len(long_), END_OF_TOKEN] = 8.0
        per_slot[len(short), END_OF_TOKEN] = -8.0  # "do not stop here"
        logp = torch.nn.functional.log_softmax(per_slot, dim=-1).reshape(-1)
        s_with = sel @ logp
        flipped += int(s_with[index[long_]] > s_with[index[short]])

    assert flipped == len(pairs), (
        f"the stop symbol must make the ordering expressible, but only {flipped}/{len(pairs)} "
        f"prefix pairs could be flipped"
    )
