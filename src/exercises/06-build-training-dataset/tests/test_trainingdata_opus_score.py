"""The OPUS criterion, tested for the things that are silently wrong when they are wrong.

Three of these matter more than the rest:

*The preconditioner is read from live optimizer state, not invented.* If it were identity, this
would be ordinary gradient-space scoring — the thing OPUS exists to improve on — and every number
downstream would look exactly the same.

*Scoring must not pollute the training gradient.* `.backward()` accumulates into `param.grad`, so
scoring sixty-four candidates would quietly add sixty-four gradients to whatever the step was
accumulating. Nothing crashes; the run just trains on a gradient nobody asked for.

*Sequential greedy is the algorithm.* Score-once-take-top-k is the ablation the paper measures as
worse, and the two are indistinguishable from the outside.

torch is an optional extra, so this file skips without it. That gate is recorded in
`tests/test_ci_shards_cover_everything.py`, because a file that collects nothing looks exactly like
a file with nothing in it — which is how 46 tests here once ran nowhere while CI stayed green.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is the `train` extra, not a base dependency")

from trainingdata import masks, model, opus_score, spec  # noqa: E402

#: Small enough for a unit test, real vocabulary so the sentinels stay inside the embedding table.
_CONFIG = {
    "vocab_size": spec.MODEL_VOCAB_SIZE,
    "n_layer": 2,
    "n_head": 4,
    "d_model": 64,
    "d_ff": 128,
}


def _batch(rng: np.random.Generator, rows: int = 2, length: int = 64) -> dict:
    """One candidate in the shape the scorer takes.

    Args:
        rng: Draw source.
        rows: Sequences in the microbatch.
        length: Tokens each.

    Returns:
        The candidate.
    """
    tokens = rng.integers(0, 10_000, size=(rows, length)).astype(np.int64)
    segments = np.zeros((rows, length), dtype=np.int32)
    positions = np.tile(np.arange(length, dtype=np.int64), (rows, 1))
    loss = np.ones((rows, length), dtype=bool)
    loss[:, -1] = False
    additive = np.stack([masks.additive_mask(segments[r]) for r in range(rows)])[:, None, :, :]
    return {"tokens": tokens, "additive": additive, "positions": positions, "loss": loss}


def _trained(steps: int = 3, lr: float = 3e-4):
    """A model whose optimizer has real AdamW state.

    Args:
        steps: How many optimizer steps to take first.
        lr: Learning rate.

    Returns:
        `(net, optimizer, rng)`.
    """
    torch.manual_seed(0)
    net = model.TinyGPT(model.ModelConfig(**_CONFIG))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    rng = np.random.default_rng(0)

    for _ in range(steps):
        batch = _batch(rng)
        logits = net(
            torch.from_numpy(batch["tokens"]),
            torch.from_numpy(batch["additive"]),
            torch.from_numpy(batch["positions"]),
        )
        summed, graded = model.cross_entropy(
            logits, torch.from_numpy(batch["tokens"]), torch.from_numpy(batch["loss"])
        )
        (summed / graded).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return net, optimizer, rng


# --- the preconditioner ------------------------------------------------------------------------


def test_before_the_first_step_the_preconditioner_is_identity_and_says_so() -> None:
    """**The distinction that must never be silent.**

    With no AdamW state there is nothing to precondition with, so this is plain gradient-space
    scoring — exactly what OPUS is an improvement on. Reporting it as OPUS would be a claim about an
    algorithm that did not run.
    """
    torch.manual_seed(0)
    net = model.TinyGPT(model.ModelConfig(**_CONFIG))
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)

    factors, found = opus_score.preconditioner(optimizer)
    assert not found
    assert all(bool((f == 1).all()) for f in factors)


def test_after_stepping_the_preconditioner_is_read_from_the_optimizer() -> None:
    """It must be the optimizer's own second moment, and it must actually vary per weight.

    A preconditioner that came back near-constant would score the same as no preconditioner at all,
    and the whole premise of the method would be untested here.
    """
    _, optimizer, _ = _trained()
    factors, found = opus_score.preconditioner(optimizer)
    assert found

    flat = torch.cat([f.reshape(-1) for f in factors])
    assert torch.isfinite(flat).all()
    assert flat.min() > 0
    assert flat.max() / flat.min() > 100, (
        "the preconditioner barely varies across weights, so scoring in optimizer space is "
        "indistinguishable from scoring in gradient space and this suite proves nothing"
    )


def test_the_preconditioner_is_bias_corrected() -> None:
    """`v` starts at zero, so early in a run an uncorrected `√v` is far too small.

    Without the correction an early-run score is not comparable with a late-run one, and the
    selector's behaviour changes over the run for a reason that has nothing to do with the data.
    """
    _, optimizer, _ = _trained(steps=1)
    corrected = torch.cat([f.reshape(-1) for f in opus_score.preconditioner(optimizer)[0]])

    group = optimizer.param_groups[0]
    raw = []
    for param in group["params"]:
        second = optimizer.state[param]["exp_avg_sq"]
        raw.append((1.0 / (second.sqrt() + group["eps"])).reshape(-1))
    uncorrected = torch.cat(raw)

    assert not torch.allclose(corrected, uncorrected)
    # One step of β₂=0.999 leaves v at a thousandth of its corrected value, so the corrected
    # denominator is ~√1000 larger and the factor correspondingly smaller.
    assert corrected.mean() < uncorrected.mean()


# --- the thing that would silently corrupt training ----------------------------------------------


def test_scoring_leaves_no_gradient_behind() -> None:
    """**The pollution test.**

    `torch.autograd.grad` is used instead of `.backward()` for exactly this. If it regressed, the
    training step would add every candidate's gradient to its own and the loss curve would look
    entirely normal.
    """
    net, optimizer, rng = _trained()
    assert all(p.grad is None for p in net.parameters())

    proxy = opus_score.proxy_direction(net, optimizer, [_batch(rng)], score_len=32)
    opus_score.score_buffer(
        net, optimizer, [_batch(rng) for _ in range(4)], proxy, keep=2, score_len=32
    )

    polluted = [name for name, p in net.named_parameters() if p.grad is not None]
    assert not polluted, f"scoring wrote gradients into {polluted}"


def test_scoring_restores_training_mode() -> None:
    """Scoring runs under `eval()`; leaving the model there would silently change training."""
    net, optimizer, rng = _trained()
    net.train()
    proxy = opus_score.proxy_direction(net, optimizer, [_batch(rng)], score_len=32)
    opus_score.score_buffer(net, optimizer, [_batch(rng)], proxy, keep=1, score_len=32)
    assert net.training


def test_scoring_does_not_move_the_weights() -> None:
    """A scorer that stepped the optimizer would train on data it was only considering."""
    net, optimizer, rng = _trained()
    before = [p.detach().clone() for p in net.parameters()]

    proxy = opus_score.proxy_direction(net, optimizer, [_batch(rng)], score_len=32)
    opus_score.score_buffer(
        net, optimizer, [_batch(rng) for _ in range(3)], proxy, keep=2, score_len=32
    )

    for was, now in zip(before, net.parameters(), strict=True):
        assert torch.equal(was, now)


# --- the algorithm -----------------------------------------------------------------------------


def test_selection_is_sequential_greedy_not_score_once() -> None:
    """**The ablation the paper measures as worse, and it is invisible from the outside.**

    After the first pick, `G` is non-empty, so every remaining candidate's score must have moved.
    If `redundancy` stayed all-zero, this would be a plain top-k.
    """
    net, optimizer, rng = _trained()
    candidates = [_batch(rng) for _ in range(6)]
    proxy = opus_score.proxy_direction(net, optimizer, [_batch(rng)], score_len=32)

    scoring = opus_score.score_buffer(net, optimizer, candidates, proxy, keep=4, score_len=32)

    assert len(scoring.picked) == 4
    assert len(set(scoring.picked)) == 4, "greedy selection picked the same candidate twice"
    rescored = [i for i in range(6) if i not in scoring.picked[:1]]
    assert any(scoring.redundancy[i] != 0.0 for i in rescored), (
        "no candidate was rescored against G, so this is score-once-take-top-k"
    )


def test_the_first_pick_carries_no_redundancy_penalty() -> None:
    """`G` is empty when the first candidate is chosen; a penalty there would be against nothing."""
    net, optimizer, rng = _trained()
    proxy = opus_score.proxy_direction(net, optimizer, [_batch(rng)], score_len=32)
    scoring = opus_score.score_buffer(
        net, optimizer, [_batch(rng) for _ in range(5)], proxy, keep=3, score_len=32
    )
    assert scoring.redundancy[scoring.picked[0]] == 0.0


def test_the_redundancy_share_is_reported_and_is_tiny_at_our_learning_rate() -> None:
    """**The measured finding, pinned so it cannot quietly stop being true.**

    The two terms carry different powers of `η`. At `3e-4` that is a structural 3,333× gap, and the
    penalty ends up contributing well under a percent — so the shipped configuration is very nearly
    a greedy top-k, and every document that describes it must say so.
    """
    net, optimizer, rng = _trained(lr=3e-4)
    proxy = opus_score.proxy_direction(net, optimizer, [_batch(rng)], score_len=32)
    scoring = opus_score.score_buffer(
        net, optimizer, [_batch(rng) for _ in range(8)], proxy, keep=4, score_len=32
    )

    assert 0.0 < scoring.redundancy_share < 0.01, (
        f"the redundancy share is {scoring.redundancy_share:.2e}; the documents claim it is "
        f"negligible at this learning rate"
    )
    assert scoring.learning_rate == pytest.approx(3e-4)
    assert scoring.redundancy_weight == 1.0


def test_a_larger_learning_rate_revives_the_penalty() -> None:
    """**The twin.** If the share were tiny whatever `η` did, the test above would be measuring
    nothing about `η`."""
    shares = {}
    for lr in (3e-4, 1e-1):
        net, optimizer, rng = _trained(lr=lr)
        proxy = opus_score.proxy_direction(net, optimizer, [_batch(rng)], score_len=32)
        shares[lr] = opus_score.score_buffer(
            net, optimizer, [_batch(rng) for _ in range(8)], proxy, keep=4, score_len=32
        ).redundancy_share

    assert shares[1e-1] > shares[3e-4] * 100, shares


def test_the_redundancy_weight_scales_the_penalty_and_is_recorded() -> None:
    """Deviating from Eq. 23 must be a choice someone made and a number the record carries."""
    net, optimizer, rng = _trained()
    candidates = [_batch(rng) for _ in range(6)]
    proxy = opus_score.proxy_direction(net, optimizer, [_batch(rng)], score_len=32)

    faithful = opus_score.score_buffer(
        net, optimizer, candidates, proxy, keep=3, score_len=32, redundancy_weight=1.0
    )
    weighted = opus_score.score_buffer(
        net, optimizer, candidates, proxy, keep=3, score_len=32, redundancy_weight=1000.0
    )

    assert weighted.redundancy_share > faithful.redundancy_share * 100
    assert weighted.redundancy_weight == 1000.0
    assert np.allclose(faithful.alignment, weighted.alignment), (
        "the weight must scale the penalty only; the alignment term is not λ's to touch"
    )


# --- the proxy ---------------------------------------------------------------------------------


def test_an_empty_proxy_set_is_refused() -> None:
    """A zero direction scores every candidate at exactly zero.

    Selection would then be a tie broken by index order — which looks like a working selector,
    produces a full batch, and is not one.
    """
    net, optimizer, _ = _trained()
    with pytest.raises(ValueError, match="proxy set is empty"):
        opus_score.proxy_direction(net, optimizer, [])


def test_the_proxy_direction_is_finite_and_spans_every_parameter() -> None:
    """A short vector would misalign against the candidates and score nonsense."""
    net, optimizer, rng = _trained()
    proxy = opus_score.proxy_direction(net, optimizer, [_batch(rng), _batch(rng)], score_len=32)

    assert proxy.numel() == sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert torch.isfinite(proxy).all()
    assert float(proxy.norm()) > 0


def test_score_len_truncates_the_candidate() -> None:
    """The only cost lever we have: our context IS the score length, so the paper's 12× discount
    does not apply and this had better actually shorten the work."""
    net, optimizer, rng = _trained()
    candidates = [_batch(rng, length=64) for _ in range(4)]
    proxy = opus_score.proxy_direction(net, optimizer, [_batch(rng)], score_len=16)

    short = opus_score.score_buffer(net, optimizer, candidates, proxy, keep=2, score_len=16)
    full = opus_score.score_buffer(net, optimizer, candidates, proxy, keep=2, score_len=64)

    assert not np.allclose(short.scores, full.scores), (
        "scoring 16 tokens and scoring 64 gave the same answer, so score_len is not being applied"
    )
    assert short.backward_passes == full.backward_passes == 4


def test_the_overhead_report_names_what_it_measured() -> None:
    """Publishing the paper's 4.7% as ours would quote a configuration we do not run."""
    net, optimizer, rng = _trained()
    proxy = opus_score.proxy_direction(net, optimizer, [_batch(rng)], score_len=32)
    scoring = opus_score.score_buffer(
        net, optimizer, [_batch(rng) for _ in range(4)], proxy, keep=2, score_len=32
    )

    report = opus_score.overhead(scoring, seconds=2.0, train_seconds=8.0)
    assert report["overhead_fraction"] == 0.25
    assert report["backward_passes"] == 4
    assert report["preconditioned"] is True
    assert report["learning_rate"] == pytest.approx(3e-4)
