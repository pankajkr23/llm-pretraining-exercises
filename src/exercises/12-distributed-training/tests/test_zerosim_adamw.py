"""The flat-shard AdamW agrees with torch's, and does not care how its buffer is sliced.

Two claims, and the second is the one ZeRO depends on. Agreement with `torch.optim.AdamW` is to a
stated tolerance, because this implementation deliberately avoids torch's fused kernels (see
`adamw.py`). Slicing invariance is **exact**: updating a buffer whole and updating it as N shards
must give the same bits, or the stages could not agree with each other bit for bit.
"""

import pytest

torch = pytest.importorskip("torch")

from zerosim.adamw import Hyper, adamw_step  # noqa: E402

HYPER = Hyper(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)


def _state(numel: int, seed: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(numel, generator=generator)
    grads = [torch.randn(numel, generator=generator) * 10 ** (-k) for k in range(6)]
    return weight, grads


def test_it_tracks_torch_adamw_over_several_steps() -> None:
    """Relative tolerance 1e-6 of the weight scale: the two differ only in fused vs separate ops."""
    weight, grads = _state(5_000, 0)
    ours = weight.clone()
    m, v = torch.zeros_like(ours), torch.zeros_like(ours)

    theirs = torch.nn.Parameter(weight.clone())
    optimiser = torch.optim.AdamW(
        [theirs],
        lr=HYPER.lr,
        betas=(HYPER.beta1, HYPER.beta2),
        eps=HYPER.eps,
        weight_decay=HYPER.weight_decay,
        foreach=False,
    )
    for step, grad in enumerate(grads, start=1):
        adamw_step(ours, grad, m, v, step, HYPER)
        theirs.grad = grad.clone()
        optimiser.step()
    gap = float((ours - theirs.detach()).abs().max())
    assert gap <= 1e-6 * float(weight.abs().max()), gap
    assert not torch.equal(ours, weight), "the update did nothing"


@pytest.mark.parametrize("n", [2, 3, 8, 32])
def test_updating_n_shards_is_bit_identical_to_updating_the_whole(n: int) -> None:
    """The property ZeRO-1 rests on — exact, not approximate."""
    numel = 37 * n + 5  # odd sizes, so shard boundaries land mid-vector
    numel -= numel % n
    weight, grads = _state(numel, 1)
    whole = weight.clone()
    m, v = torch.zeros(numel), torch.zeros(numel)
    pieces = list(weight.clone().view(n, -1))
    ms = list(torch.zeros(numel).view(n, -1))
    vs = list(torch.zeros(numel).view(n, -1))
    for step, grad in enumerate(grads, start=1):
        adamw_step(whole, grad, m, v, step, HYPER)
        for r, g in enumerate(grad.view(n, -1)):
            adamw_step(pieces[r], g, ms[r], vs[r], step, HYPER)
    assert torch.equal(torch.cat(pieces), whole)


def test_a_bf16_gradient_is_upcast_and_the_update_is_fp32() -> None:
    weight, grads = _state(100, 2)
    m, v = torch.zeros(100), torch.zeros(100)
    adamw_step(weight, grads[0].to(torch.bfloat16), m, v, 1, HYPER)
    assert weight.dtype == m.dtype == v.dtype == torch.float32


def test_a_padding_zero_with_a_zero_gradient_stays_exactly_zero() -> None:
    """Why padding can never leak into the model: decay of 0 is 0, and 0 / (0 + eps) is 0."""
    weight = torch.zeros(16)
    m, v = torch.zeros(16), torch.zeros(16)
    for step in range(1, 6):
        adamw_step(weight, torch.zeros(16), m, v, step, HYPER)
    assert torch.equal(weight, torch.zeros(16))


def test_the_first_step_moves_each_weight_by_about_the_learning_rate() -> None:
    """Bias correction on: at step 1, m̂/√v̂ = g/|g|, so every weight moves by ≈ lr (plus decay)."""
    weight = torch.zeros(4)
    m, v = torch.zeros(4), torch.zeros(4)
    adamw_step(weight, torch.tensor([3.0, -2.0, 0.5, -7.0]), m, v, 1, HYPER)
    assert torch.allclose(weight, torch.tensor([-1e-3, 1e-3, -1e-3, 1e-3]), rtol=1e-4)


def test_steps_are_counted_from_one() -> None:
    weight, grads = _state(4, 3)
    with pytest.raises(ValueError, match="counted from 1"):
        adamw_step(weight, grads[0], torch.zeros(4), torch.zeros(4), 0, HYPER)
