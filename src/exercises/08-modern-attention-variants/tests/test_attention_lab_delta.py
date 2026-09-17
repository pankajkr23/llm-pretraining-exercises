"""The delta family's own identities.

- the delta rule with β = 1 and a key orthogonal to every earlier key writes exactly its value and
  leaves every earlier association intact; its parallel form equals its recurrence;
- DeltaNet's chunkwise form equals its recurrence, for chunk sizes that do and do not divide T;
- Gated DeltaNet with its decay forced to 1 is DeltaNet with the same weights, and its chunkwise
  form (with decay) equals its recurrence;
- KDA with a channel-constant decay is Gated DeltaNet's rule (the state transposed), and its
  chunkwise form equals its recurrence;
- Gated DeltaNet-2 with erase = write = β·1 is KDA's rule, and its chunkwise form equals its
  recurrence.
"""

import math

import pytest

torch = pytest.importorskip("torch", reason="the attention lab needs the train extra")

from attention.lab import registry  # noqa: E402
from attention.lab.delta import (  # noqa: E402
    K3_OVERRIDES,
    delta_rule_chunkwise,
    delta_rule_recurrent,
    gdn2_chunkwise,
    gdn2_recurrent,
    kda_chunkwise,
    kda_recurrent,
)

B, H, T, DK, DV = 2, 3, 13, 5, 4
CHUNKS = [1, 3, 4, 5, 13, 64]


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _qkv(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = _gen(seed)
    q = torch.nn.functional.normalize(torch.randn(B, H, T, DK, generator=g).double(), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, DK, generator=g).double(), dim=-1)
    v = torch.randn(B, H, T, DV, generator=g).double()
    return q, k, v


def _beta(seed: int = 1) -> torch.Tensor:
    return torch.rand(B, H, T, generator=_gen(seed)).double()


def _log_alpha(*shape: int, seed: int = 2) -> torch.Tensor:
    return -torch.rand(*shape, generator=_gen(seed)).double()


def _x(width: int = 32, seed: int = 0) -> torch.Tensor:
    return torch.randn(2, T, width, generator=_gen(seed)).double()


def _build(name: str, **overrides):
    torch.manual_seed(0)
    return registry.get(name).build("lab", **overrides).double().eval()


# --- the delta rule -------------------------------------------------------------------------------


def test_beta_one_and_an_orthogonal_key_write_exactly_the_value() -> None:
    """After writing each (k_i, v_i) with β = 1 and orthonormal keys, S k_i = v_i for all i."""
    dk, n = 6, 5
    keys = torch.linalg.qr(torch.randn(dk, dk, generator=_gen(3)).double())[0][:, :n].T
    values = torch.randn(n, DV, generator=_gen(4)).double()
    k = keys.view(1, 1, n, dk)
    v = values.view(1, 1, n, DV)
    s0 = torch.randn(1, 1, DV, dk, generator=_gen(5)).double()
    _, s = delta_rule_recurrent(k, k, v, torch.ones(1, 1, n).double(), s0)
    torch.testing.assert_close(s[0, 0] @ keys.T, values.T)
    # A direction no key touched still reads what the initial state held there.
    free = torch.linalg.qr(torch.randn(dk, dk, generator=_gen(3)).double())[0][:, n]
    torch.testing.assert_close(s[0, 0] @ free, s0[0, 0] @ free)


@pytest.mark.parametrize("feature_map", ["elu", "dpfp"])
@pytest.mark.parametrize("chunk", CHUNKS)
def test_delta_rule_parallel_form_equals_its_recurrence(feature_map: str, chunk: int) -> None:
    parallel = _build("delta_rule", mode="parallel", feature_map=feature_map, chunk_size=chunk)
    recurrent = _build("delta_rule", mode="recurrent", feature_map=feature_map, chunk_size=chunk)
    recurrent.load_state_dict(parallel.state_dict())
    x = _x()
    with torch.no_grad():
        a, sa = parallel(x)
        b, sb = recurrent(x)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(sa["W"], sb["W"])


def test_sum_normalised_features_sum_to_one() -> None:
    mixer = _build("delta_rule", feature_map="dpfp", nu=2)
    phi = mixer.phi(torch.randn(3, 8, generator=_gen(6)).double())
    assert phi.shape[-1] == 2 * 8 * 2
    torch.testing.assert_close(phi.sum(-1), torch.ones(3).double())


# --- DeltaNet -------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", CHUNKS)
def test_deltanet_chunkwise_kernel_equals_the_recurrence(chunk: int) -> None:
    q, k, v = _qkv()
    s0 = torch.randn(B, H, DV, DK, generator=_gen(7)).double()
    a, sa = delta_rule_chunkwise(q, k, v, _beta(), s0, chunk)
    b, sb = delta_rule_recurrent(q, k, v, _beta(), s0)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(sa, sb)


@pytest.mark.parametrize("chunk", [1, 4, 5, 64])
def test_deltanet_module_chunkwise_equals_recurrent(chunk: int) -> None:
    chunkwise = _build("deltanet_parallel", mode="chunkwise", chunk_size=chunk)
    recurrent = _build("deltanet_parallel", mode="recurrent", chunk_size=chunk)
    recurrent.load_state_dict(chunkwise.state_dict())
    with torch.no_grad():
        a, _ = chunkwise(_x())
        b, _ = recurrent(_x())
    torch.testing.assert_close(a, b)


# --- Gated DeltaNet -------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", CHUNKS)
def test_gated_chunkwise_kernel_equals_the_recurrence(chunk: int) -> None:
    q, k, v = _qkv()
    s0 = torch.randn(B, H, DV, DK, generator=_gen(7)).double()
    log_alpha = _log_alpha(B, H, T)
    a, sa = delta_rule_chunkwise(q, k, v, _beta(), s0, chunk, log_alpha)
    b, sb = delta_rule_recurrent(q, k, v, _beta(), s0, log_alpha)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(sa, sb)


@pytest.mark.parametrize("mode", ["chunkwise", "recurrent"])
def test_gated_deltanet_with_its_decay_forced_to_one_is_deltanet(mode: str) -> None:
    deltanet = _build("deltanet_parallel", mode=mode)
    gated = _build("gated_deltanet", mode=mode, output_gate=False)
    missing, unexpected = gated.load_state_dict(deltanet.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {"alpha_proj.weight", "A_log", "dt_bias"}
    with torch.no_grad():
        gated.A_log.fill_(-math.inf)  # exp(A_log) = 0, so log α = 0 and α = 1 exactly
        a, sa = deltanet(_x())
        b, sb = gated(_x())
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(sa["S"], sb["S"])


def test_gated_deltanet_decay_is_in_the_open_unit_interval() -> None:
    gated = _build("gated_deltanet")
    with torch.no_grad():
        alpha = torch.exp(gated._log_alpha(_x()))
    assert ((alpha > 0) & (alpha < 1)).all()


# --- KDA ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", CHUNKS)
def test_kda_with_a_channel_constant_decay_is_the_gated_delta_rule(chunk: int) -> None:
    q, k, v = _qkv()
    s0 = torch.randn(B, H, DV, DK, generator=_gen(7)).double()  # Gated DeltaNet orientation
    scalar = _log_alpha(B, H, T)
    channel = scalar[..., None].expand(B, H, T, DK)
    want, s_want = delta_rule_recurrent(q, k, v, _beta(), s0, scalar)
    for got, s_got in (
        kda_recurrent(q, k, v, _beta(), s0.transpose(-2, -1), channel),
        kda_chunkwise(q, k, v, _beta(), s0.transpose(-2, -1), channel, chunk),
    ):
        torch.testing.assert_close(got, want)
        torch.testing.assert_close(s_got.transpose(-2, -1), s_want)


@pytest.mark.parametrize("chunk", CHUNKS)
def test_kda_chunkwise_kernel_equals_the_recurrence(chunk: int) -> None:
    q, k, v = _qkv()
    s0 = torch.randn(B, H, DK, DV, generator=_gen(8)).double()
    log_alpha = _log_alpha(B, H, T, DK)
    a, sa = kda_chunkwise(q, k, v, _beta(), s0, log_alpha, chunk)
    b, sb = kda_recurrent(q, k, v, _beta(), s0, log_alpha)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(sa, sb)


def test_kda_k3_options_bound_the_decay_and_keep_decoding_exact() -> None:
    assert K3_OVERRIDES == {"decay_floor": -5.0, "output_gate": "full-rank"}
    mixer = _build("kda", **K3_OVERRIDES)
    assert isinstance(mixer.gate, torch.nn.Linear)
    x = _x() * 100  # push the decay logits far out; the bound must still hold
    with torch.no_grad():
        log_alpha = mixer.log_alpha(x)
        full, _ = mixer(_x())
        state = mixer.init_state(2, dtype=torch.float64)
        steps = []
        for t in range(T):
            y, state = mixer(_x()[:, t : t + 1], state)
            steps.append(y)
    # The bound is open in the paper; float64 saturates the sigmoid, so it is reached, not passed.
    assert (log_alpha >= -5.0).all() and (log_alpha <= 0).all()
    assert log_alpha.min() < -4.9  # the logits really were pushed to the floor
    torch.testing.assert_close(torch.cat(steps, dim=1), full)


# --- Gated DeltaNet-2 -----------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", CHUNKS)
def test_gdn2_with_tied_gates_is_kda(chunk: int) -> None:
    q, k, v = _qkv()
    s0 = torch.randn(B, H, DK, DV, generator=_gen(8)).double()
    log_alpha = _log_alpha(B, H, T, DK)
    beta = _beta()
    erase = beta[..., None].expand(B, H, T, DK)
    write = beta[..., None].expand(B, H, T, DV)
    want, s_want = kda_recurrent(q, k, v, beta, s0, log_alpha)
    for got, s_got in (
        gdn2_recurrent(q, k, v, erase, write, s0, log_alpha),
        gdn2_chunkwise(q, k, v, erase, write, s0, log_alpha, chunk),
    ):
        torch.testing.assert_close(got, want)
        torch.testing.assert_close(s_got, s_want)


@pytest.mark.parametrize("chunk", CHUNKS)
def test_gdn2_chunkwise_kernel_equals_the_recurrence(chunk: int) -> None:
    q, k, v = _qkv()
    g = _gen(9)
    erase = torch.rand(B, H, T, DK, generator=g).double()
    write = torch.rand(B, H, T, DV, generator=g).double()
    s0 = torch.randn(B, H, DK, DV, generator=g).double()
    log_alpha = _log_alpha(B, H, T, DK)
    a, sa = gdn2_chunkwise(q, k, v, erase, write, s0, log_alpha, chunk)
    b, sb = gdn2_recurrent(q, k, v, erase, write, s0, log_alpha)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(sa, sb)


def test_gdn2_rejects_a_state_size_other_than_its_head_dim() -> None:
    with pytest.raises(ValueError, match="state_size"):
        registry.get("gated_deltanet2").factory(
            d_model=32, heads=4, head_dim=8, chunk_size=5, state_size=16
        )
