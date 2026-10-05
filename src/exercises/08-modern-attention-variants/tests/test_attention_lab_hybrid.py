"""A hybrid is its layers, in the declared order, and nothing else.

The generic suite already checks that each hybrid is causal, decodes stepwise, grows its state and
can memorise a batch. What is specific to a hybrid is that stacking adds nothing of its own: the
layout is the one declared, a one-layer stack is exactly that layer behind its norm, the memory is
the sum of the layers' memories, and the full-attention layers the sources describe (NoPE MLA,
Kimi K3's Gated MLA) are the parent MLA with one thing changed.
"""

import pytest

torch = pytest.importorskip("torch", reason="the attention lab needs the train extra")

from attention.lab import registry  # noqa: E402
from attention.lab.base import as_kwargs, tensor_bytes  # noqa: E402
from attention.lab.core import GroupedAttention  # noqa: E402
from attention.lab.delta import K3_OVERRIDES, KimiDeltaAttention  # noqa: E402
from attention.lab.hybrid import (  # noqa: E402
    GatedNoPEMultiHeadLatentAttention,
    NoPEMultiHeadLatentAttention,
    PartialRoPEGroupedAttention,
    layout,
)
from attention.lab.linear import LightningAttention  # noqa: E402
from attention.lab.mla import MultiHeadLatentAttention  # noqa: E402

KDA_HYBRID = registry.get("kda_hybrid")
K3 = registry.get("kimi_k3_stack")
LIGHTNING = registry.get("lightning_hybrid")

LETTER_CLASS = {
    "K": KimiDeltaAttention,
    "M": NoPEMultiHeadLatentAttention,
    "G": GatedNoPEMultiHeadLatentAttention,
    "L": LightningAttention,
    "A": PartialRoPEGroupedAttention,
}


def _build(spec, **overrides):
    torch.manual_seed(0)
    return spec.build("lab", **overrides).double().eval()


def _x(tokens: int, batch: int = 2, width: int = 32, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, tokens, width, generator=generator, dtype=torch.float64)


def _mla_kwargs(spec) -> dict:
    kw = as_kwargs(spec.lab)
    return {
        "d_model": kw["d_model"],
        "heads": kw["mla_heads"],
        "head_dim": kw["mla_head_dim"],
        "kv_rank": kw["mla_kv_rank"],
        "q_rank": kw["mla_q_rank"],
        "rope_dim": kw["mla_rope_dim"],
        "rope_base": kw["mla_rope_base"],
    }


# --- the layout -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (KDA_HYBRID, "KKKMKKKM"),
        (K3, "KKKGKKKGG"),
        (LIGHTNING, "LLLLLLLA"),
    ],
    ids=lambda v: getattr(v, "name", v),
)
def test_the_layers_built_are_the_declared_layout(spec, expected) -> None:
    """Two 3:1 blocks, K3's extra final Gated MLA, one 7:1 block — letter by letter and by class."""
    mixer = _build(spec)
    assert mixer.letters == expected
    built = [type(layer) for layer in mixer.layers]
    assert built == [LETTER_CLASS[c] for c in expected]


@pytest.mark.parametrize(
    ("spec", "full_letter", "full_layers"),
    [
        # Kimi Linear's released config: full attention at 1-based layers 4, 8, ..., 24 and 27.
        (KDA_HYBRID, "M", [*range(4, 25, 4), 27]),
        # Kimi K3's released config: 4, 8, ..., 92 and then 93.
        (K3, "G", [*range(4, 93, 4), 93]),
        # MiniMax-Text-01's released config: softmax at every eighth of 80 layers.
        (LIGHTNING, "A", list(range(8, 81, 8))),
    ],
    ids=lambda v: getattr(v, "name", None),
)
def test_the_paper_scale_layout_is_the_released_one(spec, full_letter, full_layers) -> None:
    kw = as_kwargs(spec.paper)
    linear = "L" if spec is LIGHTNING else "K"
    letters = layout(kw["ratio"], kw["pattern"], kw["final"], kw["num_layers"], linear, full_letter)
    assert len(letters) == kw["num_layers"]
    assert [i + 1 for i, c in enumerate(letters) if c == full_letter] == full_layers


def test_a_layout_that_cannot_fill_the_depth_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be"):
        layout(3, "", "", 10, "K", "M")


def test_a_learner_can_change_the_ratio_and_the_pattern() -> None:
    assert _build(KDA_HYBRID, ratio=1).letters == "KMKMKMKM"
    assert _build(KDA_HYBRID, pattern="MK", num_layers=4).letters == "MKMK"


def test_k3_kda_layers_carry_the_k3_changes_and_lightning_layers_know_their_depth() -> None:
    for layer in _build(K3).layers:
        if isinstance(layer, KimiDeltaAttention):
            assert layer.decay_floor == K3_OVERRIDES["decay_floor"] == -5.0
            assert isinstance(layer.gate, torch.nn.Linear), "K3's output gate is full rank"
    stack = _build(LIGHTNING)
    ratios = [layer.ratio for layer in stack.layers if isinstance(layer, LightningAttention)]
    for index, ratio in enumerate(ratios):
        solo = LightningAttention(32, 4, 8, 5, layer_index=index, num_layers=len(stack.layers))
        torch.testing.assert_close(ratio, solo.ratio)
    assert not torch.equal(ratios[0], ratios[-1]), "the decay must change with depth"


# --- one layer is that layer behind its norm ------------------------------------------------------


def _single(spec, letter: str):
    stack = _build(spec, pattern=letter, num_layers=1, final="")
    assert stack.letters == letter
    return stack


@pytest.mark.parametrize(
    ("spec", "letter"),
    [(KDA_HYBRID, "M"), (KDA_HYBRID, "K"), (K3, "G"), (LIGHTNING, "A")],
    ids=["nope_mla", "kda", "gated_mla", "partial_rope_gqa"],
)
def test_a_one_layer_stack_is_that_layer_wrapped(spec, letter) -> None:
    """`stack(x) == layer(norm(x))`, with a layer built on its own and given the same weights."""
    stack = _single(spec, letter)
    kw = as_kwargs(spec.lab)
    if letter in "MG":
        solo = LETTER_CLASS[letter](position=kw["mla_position"], **_mla_kwargs(spec))
    elif letter == "K":
        solo = KimiDeltaAttention(
            32, kw["kda_heads"], kw["kda_head_dim"], kw["kda_chunk_size"], kw["kda_conv_kernel"]
        )
    else:
        solo = PartialRoPEGroupedAttention(
            32,
            kw["attn_heads"],
            kw["attn_kv_heads"],
            kw["attn_head_dim"],
            kw["attn_rotary_dim"],
            kw["attn_rope_base"],
        )
    solo = solo.double().eval()
    solo.load_state_dict(stack.layers[0].state_dict())
    x = _x(9)
    with torch.no_grad():
        got, state = stack(x)
        want, solo_state = solo(stack.norms[0](x))
    torch.testing.assert_close(got, want)
    assert stack.state_bytes(state) == solo.state_bytes(solo_state)


# --- memory ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [KDA_HYBRID, K3], ids=lambda s: s.name)
def test_state_bytes_are_the_layers_bytes_and_only_the_mla_part_grows(spec) -> None:
    stack = _build(spec)
    kw = as_kwargs(spec.lab)
    full = "G" if spec is K3 else "M"
    parts = []
    for tokens in (16, 40):
        with torch.no_grad():
            _, state = stack(_x(tokens))
        per_layer = [tensor_bytes(layer_state) for layer_state in state["layers"]]
        assert len(per_layer) == len(stack.layers)
        assert stack.state_bytes(state) == sum(per_layer) == tensor_bytes(state)
        parts.append(stack.bytes_by_letter(state))
    assert parts[0]["K"] == parts[1]["K"] > 0, "the KDA part must not depend on the length"
    grown = parts[1][full] - parts[0][full]
    per_token = (kw["mla_kv_rank"] + kw["mla_rope_dim"]) * torch.float64.itemsize
    assert grown == stack.letters.count(full) * per_token * 2 * (40 - 16)


# --- the full-attention layers --------------------------------------------------------------------


def _with_gap(layer, gap: int):
    """Prefill four tokens, then decode three more as if `gap` positions had passed in between.

    A global offset would not do: RoPE depends only on the distance between query and key, so
    shifting every position by the same amount leaves the rotated parent unchanged too. A gap
    between the cache and the new tokens changes that distance, which only a NoPE layer ignores.
    """
    x = _x(7)
    with torch.no_grad():
        _, state = layer(x[:, :4])
        state = {**state, "pos": state["pos"] + gap}
        out, _ = layer(x[:, 4:], state)
    return out


def test_nope_mla_does_not_see_how_far_back_its_cache_is() -> None:
    torch.manual_seed(0)
    layer = NoPEMultiHeadLatentAttention(**_mla_kwargs(KDA_HYBRID)).double().eval()
    torch.testing.assert_close(_with_gap(layer, 0), _with_gap(layer, 100))


def test_rope_mla_does_see_it_so_the_gap_check_can_fail() -> None:
    """The same check against the rotated parent must find a difference."""
    torch.manual_seed(0)
    layer = MultiHeadLatentAttention(**_mla_kwargs(KDA_HYBRID)).double().eval()
    assert not torch.allclose(_with_gap(layer, 0), _with_gap(layer, 100))


def test_the_nope_subclass_with_rope_switched_on_is_the_parent_mla() -> None:
    """The subclass re-states the parent's forward; with rotation on, the two must agree exactly."""
    kw = _mla_kwargs(KDA_HYBRID)
    torch.manual_seed(0)
    parent = MultiHeadLatentAttention(**kw).double().eval()
    child = NoPEMultiHeadLatentAttention(position="RoPE", **kw).double().eval()
    child.load_state_dict(parent.state_dict())
    x = _x(6)
    with torch.no_grad():
        prefill_p, state_p = parent(x[:, :4])
        prefill_c, state_c = child(x[:, :4])
        step_p, _ = parent(x[:, 4:], state_p)
        step_c, _ = child(x[:, 4:], state_c)
    torch.testing.assert_close(prefill_c, prefill_p)
    torch.testing.assert_close(step_c, step_p)


def test_gated_mla_with_its_gate_held_open_is_the_ungated_nope_mla() -> None:
    """Eq 7 with `σ(W_g x) = 1` everywhere is `W_o õ`, the NoPE MLA output."""
    kw = _mla_kwargs(K3)
    torch.manual_seed(0)
    gated = GatedNoPEMultiHeadLatentAttention(**kw).double().eval()
    plain = NoPEMultiHeadLatentAttention(**kw).double().eval()
    plain.load_state_dict(gated.state_dict(), strict=False)
    gated.gate = lambda x: torch.ones(*x.shape[:-1], kw["heads"] * kw["head_dim"], dtype=x.dtype)
    x = _x(8)
    with torch.no_grad():
        torch.testing.assert_close(gated(x)[0], plain(x)[0])


def test_the_gate_is_applied_channel_by_channel_before_the_output_projection() -> None:
    """Closing one head channel removes exactly that channel's column of `W_o` times `õ`.

    A uniform gate cannot tell "before `W_o`" from "after `W_o`", because `W_o` is linear; a gate
    that closes a single channel can.
    """
    kw = _mla_kwargs(K3)
    torch.manual_seed(0)
    gated = GatedNoPEMultiHeadLatentAttention(**kw).double().eval()
    width = kw["heads"] * kw["head_dim"]
    mask = torch.ones(width, dtype=torch.float64)
    mask[3] = 0.0
    gated.gate = lambda x: mask.expand(*x.shape[:-1], width)
    x = _x(5)
    with torch.no_grad():
        o, _ = gated.head_outputs(x, None)
        expected = gated.w_o(o) - o[..., 3:4] * gated.w_o.weight[:, 3]
        torch.testing.assert_close(gated(x)[0], expected)


def test_an_ungated_k3_stack_builds_plain_nope_mla() -> None:
    stack = _build(K3, mla_output_gate="none")
    assert stack.letters == "KKKGKKKGG"
    assert [type(m) for m in stack.layers][3] is NoPEMultiHeadLatentAttention


def test_partial_rope_leaves_the_unrotated_half_of_each_head_alone() -> None:
    """RoPE on the leading `rotary_dim` dimensions only: the rest is position-free."""
    torch.manual_seed(0)
    layer = PartialRoPEGroupedAttention(32, 4, 2, 8, 4, 10000.0).double()
    q = torch.randn(1, 4, 3, 8, dtype=torch.float64)
    turned = layer._rotate(q, 5)
    torch.testing.assert_close(turned[..., 4:], q[..., 4:])
    assert not torch.allclose(turned[..., :4], q[..., :4])

    none = PartialRoPEGroupedAttention(32, 4, 2, 8, 0, 10000.0).double()
    plain = GroupedAttention(32, 4, 2, 8).double()
    none.load_state_dict(layer.state_dict())
    plain.load_state_dict(layer.state_dict())
    x = _x(6)
    with torch.no_grad():
        torch.testing.assert_close(none(x)[0], plain(x)[0], msg="rotary_dim=0 must be plain GQA")
