"""MLA keeps only its latent and one shared rotated key per token, and decodes from them exactly.

The generic suite already checks stepwise == full for every variant. What is specific to MLA is
*what* the cache holds: DeepSeek-V2 (arXiv:2405.04434v5 §2.1.3) says the layer needs
`(d_c + d_h^R)` elements per token per layer, independent of the number of heads. These tests
count that from the tensors the layer really keeps, and check that a decode fed nothing but those
tensors reproduces the full pass.
"""

import pytest

torch = pytest.importorskip("torch", reason="the attention lab needs the train extra")

from attention.lab import registry  # noqa: E402
from attention.lab.base import as_kwargs  # noqa: E402

SPEC = registry.get("mla")
LAB = as_kwargs(SPEC.lab)


def _mixer(**overrides):
    torch.manual_seed(0)
    return SPEC.build("lab", **overrides).double().eval()


def _x(tokens: int, batch: int = 1, width: int = LAB["d_model"], seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, tokens, width, generator=generator, dtype=torch.float64)


def test_the_cache_holds_only_the_latent_and_the_shared_rotated_key() -> None:
    mixer = _mixer()
    with torch.no_grad():
        _, state = mixer(_x(7))
    assert set(state) == {"c_kv", "k_rope", "pos"}
    assert state["c_kv"].shape == (1, 7, LAB["kv_rank"])
    assert state["k_rope"].shape == (1, 7, LAB["rope_dim"])


@pytest.mark.parametrize("tokens", [5, 9])
def test_cache_bytes_per_token_are_latent_width_plus_rope_width(tokens) -> None:
    """Eq 9 and 15: `(d_c + d_h^R)` elements per token, counted from the real tensors."""
    mixer = _mixer()
    with torch.no_grad():
        _, state = mixer(_x(tokens))
    per_token = mixer.state_bytes(state) / tokens
    assert per_token == (LAB["kv_rank"] + LAB["rope_dim"]) * torch.float64.itemsize


def test_cache_bytes_do_not_depend_on_the_number_of_heads() -> None:
    """The point of MLA: more heads, same cache."""
    sizes = []
    for heads in (2, 8):
        mixer = _mixer(heads=heads, head_dim=LAB["d_model"] // heads)
        with torch.no_grad():
            _, state = mixer(_x(6))
        sizes.append(mixer.state_bytes(state))
    assert sizes[0] == sizes[1]


def test_paper_scale_cache_is_576_elements_per_token() -> None:
    """At the stated sizes (d_c = 512, d_h^R = 64) the cache is 576 elements per token.

    Built on the meta device, so the shapes are real and no memory is allocated.
    """
    kw = as_kwargs(SPEC.paper)
    with torch.device("meta"):
        mixer = SPEC.build("paper")
        x = torch.empty(1, 3, kw["d_model"])
        _, state = mixer(x)
    elements = sum(t.numel() for k, t in state.items() if k != "pos")
    assert elements / 3 == kw["kv_rank"] + kw["rope_dim"] == 576


def test_decoding_from_the_cache_alone_reproduces_the_full_pass() -> None:
    """A prefill, then single tokens, threading only the cached latent and rotated key."""
    mixer = _mixer()
    x = _x(10, batch=2)
    with torch.no_grad():
        full, _ = mixer(x)
        pieces, state = [], None
        y, state = mixer(x[:, :4], state)
        pieces.append(y)
        for t in range(4, 10):
            cached = {"c_kv": state["c_kv"], "k_rope": state["k_rope"], "pos": state["pos"]}
            y, state = mixer(x[:, t : t + 1], cached)
            pieces.append(y)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full)


def test_shifting_every_position_leaves_the_output_unchanged() -> None:
    """Eq 14–15 rotate queries and the shared key at the same positions, so only offsets count."""
    mixer = _mixer()
    x = _x(4)
    with torch.no_grad():
        at_zero, _ = mixer(x)
        state = mixer.init_state(1, dtype=torch.float64)
        state["pos"] = 5  # pretend five tokens came first, with nothing cached
        shifted, _ = mixer(x, state)
    # Relative positions are unchanged, so RoPE gives the same attention scores.
    torch.testing.assert_close(at_zero, shifted)
