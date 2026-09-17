"""The linear family's own identities: each form of a variant computes the same function.

- linear attention: the RNN form (Eq 16-20) equals the cumulative form (Eq 10-12), and both equal
  the quadratic masked form of Eq 9 written out directly;
- lightning attention: the tiled pass (Algorithm 1) equals the recurrence (Eq 5) for tile sizes
  that do and do not divide the length; with the decay switched off it equals plain cumulative
  linear attention followed by the same norm and gate; with the decay on it does not.
"""

import pytest

torch = pytest.importorskip("torch", reason="the attention lab needs the train extra")

from attention.lab import registry  # noqa: E402
from attention.lab.linear import cumulative_linear_attention, elu_plus_one  # noqa: E402
from attention.lab.ops import merge_heads, split_heads  # noqa: E402

TOKENS = 13


def _x(width: int, tokens: int = TOKENS, seed: int = 0) -> torch.Tensor:
    return torch.randn(2, tokens, width, generator=torch.Generator().manual_seed(seed)).double()


def _build(name: str, **overrides):
    torch.manual_seed(0)
    return registry.get(name).build("lab", **overrides).double().eval()


def test_linear_attention_recurrent_form_equals_parallel_form() -> None:
    parallel = _build("linear_attention", mode="parallel")
    recurrent = _build("linear_attention", mode="recurrent")
    recurrent.load_state_dict(parallel.state_dict())
    x = _x(32)
    with torch.no_grad():
        a, state_a = parallel(x)
        b, state_b = recurrent(x)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(state_a["s"], state_b["s"])
    torch.testing.assert_close(state_a["z"], state_b["z"])


def test_linear_attention_equals_the_masked_quadratic_form_of_eq_9() -> None:
    mixer = _build("linear_attention")
    x = _x(32)
    with torch.no_grad():
        got, _ = mixer(x)
        q = elu_plus_one(split_heads(mixer.q_proj(x), mixer.heads))
        k = elu_plus_one(split_heads(mixer.k_proj(x), mixer.heads))
        v = split_heads(mixer.v_proj(x), mixer.heads)
        sim = (q @ k.transpose(-2, -1)).tril()
        want = mixer.out_proj(merge_heads((sim @ v) / sim.sum(-1, keepdim=True)))
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("block_size", [1, 3, 5, 13, 64])
def test_lightning_tiled_pass_equals_the_recurrence(block_size: int) -> None:
    tiled = _build("lightning_attention", mode="tiled", block_size=block_size)
    recurrent = _build("lightning_attention", mode="recurrent", block_size=block_size)
    recurrent.load_state_dict(tiled.state_dict())
    x = _x(32)
    with torch.no_grad():
        a, state_a = tiled(x)
        b, state_b = recurrent(x)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(state_a["kv"], state_b["kv"])


def _plain_linear_with_lightning_head(mixer, x: torch.Tensor) -> torch.Tensor:
    """Cumulative linear attention (no denominator, no decay) with lightning's norm and gate."""
    parts = torch.nn.functional.silu(mixer.qkv_proj(x)).chunk(3, dim=-1)
    q, k, v = (split_heads(p, mixer.heads) for p in parts)
    out, _ = cumulative_linear_attention(q, k, v)
    out = torch.sigmoid(mixer.output_gate(x)) * mixer.norm(merge_heads(out))
    return mixer.out_proj(out)


def test_lightning_without_decay_is_plain_linear_attention_with_its_norm_and_gate() -> None:
    mixer = _build("lightning_attention", decay=False)
    x = _x(32)
    with torch.no_grad():
        got, _ = mixer(x)
        want = _plain_linear_with_lightning_head(mixer, x)
    torch.testing.assert_close(got, want)


def test_lightning_decay_changes_the_output() -> None:
    """The twin of the identity above: with the decay on, the plain form must not match."""
    mixer = _build("lightning_attention", decay=True)
    assert (mixer.ratio < 1).all()
    x = _x(32)
    with torch.no_grad():
        got, _ = mixer(x)
        plain = _plain_linear_with_lightning_head(mixer, x)
    assert (got - plain).abs().max() > 1e-3


def test_lightning_slopes_follow_the_released_geometric_rule() -> None:
    """Four heads: start = 2^(-2^(-(2-3))) = 1/4, so the slopes are 1/4, 1/16, 1/64, 1/256."""
    from attention.lab.linear import alibi_style_slopes

    assert alibi_style_slopes(4) == [0.25, 0.0625, 0.015625, 0.00390625]
    assert len(alibi_style_slopes(6)) == 6
