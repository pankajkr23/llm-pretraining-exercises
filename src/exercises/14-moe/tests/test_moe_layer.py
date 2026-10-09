"""The upcycled layer starts as the dense layer, routes as described, and balances itself."""

import pytest

torch = pytest.importorskip("torch", reason="exercise 14 needs the train extra")

from moe.layer import (  # noqa: E402
    MoE,
    MoEConfig,
    balance_stats,
    count_parameters,
    moe_layers,
    upcycle,
)
from optimizers.model import GPT, MLP, ModelConfig  # noqa: E402

CFG = ModelConfig(vocab_size=50, width=64, depth=3, seq_len=10)


def _x(seed: int = 0) -> torch.Tensor:
    return torch.randn(3, 7, 64, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("router", ["softmax", "sigmoid"])
@pytest.mark.parametrize(("n", "k"), [(8, 2), (4, 1), (4, 4)])
def test_an_upcycled_layer_computes_exactly_what_the_dense_layer_did(
    router: str, n: int, k: int
) -> None:
    dense = MLP(64).double()
    moe = MoE(dense, MoEConfig(n_experts=n, top_k=k, router=router)).double().eval()
    x = _x().double()
    with torch.no_grad():
        torch.testing.assert_close(moe(x), dense(x), rtol=1e-12, atol=1e-12)


def test_an_upcycled_model_has_the_same_loss_as_the_dense_one() -> None:
    dense = GPT(CFG, seed=0).double()
    converted = upcycle(GPT(CFG, seed=0).double(), MoEConfig(n_experts=4, top_k=2))
    batch = torch.randint(0, 50, (2, 11), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert dense.loss(batch).item() == pytest.approx(converted.loss(batch).item(), abs=1e-12)
    assert len(moe_layers(converted)) == CFG.depth


def test_the_twin_a_perturbed_expert_breaks_continuity() -> None:
    """If continuity held for any experts, the test above would prove nothing."""
    dense = MLP(64).double()
    moe = MoE(dense, MoEConfig(n_experts=4, top_k=2)).double().eval()
    with torch.no_grad():
        for expert in moe.experts:
            expert.up.weight.add_(0.1)
        assert not torch.allclose(moe(_x().double()), dense(_x().double()))


def test_each_token_reaches_exactly_top_k_experts_and_loads_sum_to_k() -> None:
    moe = MoE(MLP(64), MoEConfig(n_experts=8, top_k=2)).train()
    moe(_x())
    assert moe.load.sum().item() == pytest.approx(2.0)
    assert (moe.load >= 0).all()


def test_the_router_runs_in_float32_even_for_a_half_precision_input() -> None:
    moe = MoE(MLP(64), MoEConfig())
    scores = moe.scores(_x().to(torch.bfloat16))
    assert scores.dtype == torch.float32


def test_the_router_starts_small() -> None:
    moe = MoE(MLP(64), MoEConfig(router_init_scale=0.1))
    assert moe.router.weight.std().item() == pytest.approx(0.1 / 8, rel=0.15)


def test_the_balancing_bias_moves_toward_the_underloaded_expert() -> None:
    moe = MoE(MLP(64), MoEConfig(n_experts=4, top_k=1, bias_rate=0.01))
    moe.load.copy_(torch.tensor([0.7, 0.1, 0.1, 0.1]))
    moe.rebalance()
    assert moe.bias[0] < 0 < moe.bias[1]
    assert moe.bias[1] == moe.bias[2] == moe.bias[3]


def test_the_bias_changes_which_experts_are_chosen_but_not_their_weights() -> None:
    torch.manual_seed(0)
    moe = MoE(MLP(64), MoEConfig(n_experts=4, top_k=1)).train()
    x = _x()
    moe(x)
    favourite = int(moe.load.argmax())
    moe.bias[favourite] = -10.0
    moe(x)
    assert moe.load[favourite] == 0, "a large negative bias must stop the expert being chosen"


@pytest.mark.parametrize("router", ["softmax", "sigmoid"])
def test_the_bias_never_enters_the_weights(router: str) -> None:
    """With differing experts and a non-zero bias, the output uses the unbiased scores.

    Identical experts cannot test this — any weights summing to one give the same output — so the
    experts are perturbed apart first. Adding the bias to the gathered weights turns this red.
    """
    torch.manual_seed(0)
    moe = MoE(MLP(64).double(), MoEConfig(n_experts=4, top_k=2, router=router)).double().eval()
    with torch.no_grad():
        for i, expert in enumerate(moe.experts):
            expert.up.weight.add_(0.1 * (i + 1) * torch.randn_like(expert.up.weight))
        moe.bias.copy_(torch.tensor([0.3, -0.2, 0.1, -0.4]))
        flat = _x().double().reshape(-1, 64)
        scores = moe.scores(flat).double()
        chosen = (scores + moe.bias).topk(2, dim=-1).indices
        weights = scores.gather(-1, chosen)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        expected = torch.stack(
            [
                sum(weights[t, j] * moe.experts[int(chosen[t, j])](flat[t]) for j in range(2))
                for t in range(flat.shape[0])
            ]
        )
        torch.testing.assert_close(moe(flat), expected, rtol=1e-12, atol=1e-12)


def test_the_router_receives_a_gradient_when_two_experts_are_chosen() -> None:
    """With `top_k = 2` the renormalised weights still depend on the scores, so the router learns.

    (With `top_k = 1` the single weight is exactly 1 and the router gets no gradient through the
    output — a property of renormalised top-1 routing, not a defect here; FULL uses two.)
    A `.detach()` on the weights turns this red; the overfit test cannot see it, because the
    experts alone can bring the loss down.
    """
    torch.manual_seed(0)
    moe = MoE(MLP(64), MoEConfig(n_experts=4, top_k=2)).train()
    with torch.no_grad():
        for i, expert in enumerate(moe.experts):
            expert.up.weight.add_(0.1 * (i + 1) * torch.randn_like(expert.up.weight))
    moe(_x()).pow(2).sum().backward()
    assert moe.router.weight.grad is not None
    assert moe.router.weight.grad.abs().sum() > 0


def test_balance_stats_reports_violation_and_dead_experts() -> None:
    moe = MoE(MLP(64), MoEConfig(n_experts=4, top_k=1))
    moe.load.copy_(torch.tensor([0.5, 0.5, 0.0, 0.0]))
    stats = balance_stats([moe])
    assert stats["max_violation"] == pytest.approx(1.0)
    assert stats["dead"] == 2


def test_active_parameters_exclude_the_unchosen_experts() -> None:
    model = upcycle(GPT(CFG), MoEConfig(n_experts=8, top_k=2))
    per_expert = sum(p.numel() for p in model.blocks[0].mlp.experts[0].parameters())
    total = count_parameters(model, False, 2, 8)
    active = count_parameters(model, True, 2, 8)
    assert total - active == CFG.depth * 6 * per_expert


@pytest.mark.parametrize("bad", [{"top_k": 0}, {"top_k": 9}, {"router": "argmax"}])
def test_invalid_configs_are_refused(bad) -> None:
    with pytest.raises(ValueError):
        MoEConfig(**bad)


def test_a_one_batch_overfit_drives_the_converted_models_loss_down() -> None:
    torch.manual_seed(0)
    model = upcycle(GPT(CFG, seed=0), MoEConfig(n_experts=4, top_k=2))
    batch = torch.randint(0, 50, (4, 11), generator=torch.Generator().manual_seed(1))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(100):
        loss = model.loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        for layer in moe_layers(model):
            layer.rebalance()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.3 * first


def test_the_balancing_buffers_live_on_the_experts_device() -> None:
    """Built on any device, the bias and load sit with the experts and the router.

    They were once created on the CPU whatever the dense layer's device, so routing on a GPU added
    a CPU bias to GPU scores and crashed — invisible to every CPU test. The `meta` device makes
    the check device-independent.
    """
    moe = MoE(MLP(64).to("meta"), MoEConfig(n_experts=4, top_k=2))
    assert moe.bias.device.type == moe.load.device.type == "meta"
    assert moe.router.weight.device.type == moe.experts[0].up.weight.device.type == "meta"
