"""The positions family keeps the identities its sources state, and the ones the lab computed.

Each identity below was watched failing against a deliberately broken implementation before it was
trusted (the mutations are listed in the report of the change that added this file). The generic
contract — causal, stepwise, memory growth, overfit — is `tests/test_attention_lab.py`.
"""

import math

import pytest

torch = pytest.importorskip("torch", reason="the attention lab needs the train extra")

from attention.lab import ops  # noqa: E402
from attention.lab.positions import (  # noqa: E402
    AlibiAttention,
    DropeAttention,
    HdRopeAttention,
    LearnedAbsoluteAttention,
    NtkAwareAttention,
    RopeAttention,
    YarnAttention,
    alibi_slopes,
    apply_hd_rope,
    ntk_base,
    sinusoidal_table,
    yarn_attention_factor,
    yarn_frequencies,
)

D_MODEL, HEADS, BASE = 32, 2, 10000.0
DTYPE = torch.float64


def _x(tokens: int = 10, seed: int = 0, width: int = D_MODEL) -> torch.Tensor:
    return torch.randn(2, tokens, width, generator=torch.Generator().manual_seed(seed), dtype=DTYPE)


def _pair(first: torch.nn.Module, second: torch.nn.Module) -> tuple:
    """Give `second` exactly `first`'s weights, so any difference is the position scheme."""
    second.load_state_dict(first.state_dict())
    return first.double().eval(), second.double().eval()


def _run(mixer: torch.nn.Module, x: torch.Tensor, start: int = 0) -> torch.Tensor:
    state = mixer.init_state(x.shape[0], dtype=x.dtype, start=start)
    with torch.no_grad():
        y, _ = mixer(x, state)
    return y


# --- sinusoidal -----------------------------------------------------------------------------------


def test_a_sinusoidal_row_at_p_plus_k_is_a_fixed_linear_map_of_the_row_at_p() -> None:
    """§3.5's stated reason for the table: `PE(p + k)` is a linear function of `PE(p)`.

    `M_k` is fitted by least squares on one set of positions and must reproduce another set it
    never saw, so a table that is merely *approximately* linear over the fitted range fails.
    """
    width, offset = 16, 7
    generator = torch.Generator().manual_seed(0)
    fit = torch.randint(0, 5000, (400,), generator=generator)
    held_out = torch.randint(5000, 20000, (200,), generator=generator)
    source = sinusoidal_table(fit, width, BASE)
    target = sinusoidal_table(fit + offset, width, BASE)
    m_k = torch.linalg.lstsq(source, target).solution
    predicted = sinusoidal_table(held_out, width, BASE) @ m_k
    torch.testing.assert_close(predicted, sinusoidal_table(held_out + offset, width, BASE))


def test_the_sinusoidal_table_matches_the_printed_formula_at_one_cell() -> None:
    pos, i, width = 37, 3, 16
    table = sinusoidal_table(torch.tensor([pos]), width, BASE)
    assert table[0, 2 * i].item() == pytest.approx(math.sin(pos / BASE ** (2 * i / width)))
    assert table[0, 2 * i + 1].item() == pytest.approx(math.cos(pos / BASE ** (2 * i / width)))


# --- learned absolute -----------------------------------------------------------------------------


def test_a_learned_table_has_no_row_beyond_its_size() -> None:
    mixer = LearnedAbsoluteAttention(D_MODEL, HEADS, max_positions=8, init_std=0.1).double()
    _run(mixer, _x(8))
    with pytest.raises(ValueError, match="beyond"):
        _run(mixer, _x(1), start=8)


# --- RoPE -----------------------------------------------------------------------------------------


def _rotated_score(mixer: RopeAttention, q, k, m: int, n: int) -> torch.Tensor:
    rq = mixer.rotate(q, torch.tensor([m]))
    rk = mixer.rotate(k, torch.tensor([n]))
    return (rq * rk).sum()


def test_a_rope_score_depends_only_on_the_offset() -> None:
    """Eq 16: shifting both positions by the same amount leaves `q·k` unchanged."""
    mixer = RopeAttention(D_MODEL, HEADS, BASE)
    generator = torch.Generator().manual_seed(1)
    q = torch.randn(1, 1, 1, mixer.head_dim, generator=generator, dtype=DTYPE)
    k = torch.randn(1, 1, 1, mixer.head_dim, generator=generator, dtype=DTYPE)
    base_score = _rotated_score(mixer, q, k, 3, 11)
    for shift in (1, 50, 997):
        torch.testing.assert_close(_rotated_score(mixer, q, k, 3 + shift, 11 + shift), base_score)
    assert not torch.isclose(_rotated_score(mixer, q, k, 3, 12), base_score)


def test_a_whole_rope_layer_is_unchanged_when_every_position_shifts() -> None:
    torch.manual_seed(0)
    mixer = RopeAttention(D_MODEL, HEADS, BASE).double().eval()
    x = _x()
    torch.testing.assert_close(_run(mixer, x, start=500), _run(mixer, x))


def test_rope_at_angle_zero_is_no_rotation() -> None:
    x = _x().view(2, 1, 10, D_MODEL)
    torch.testing.assert_close(ops.apply_rope(x, torch.zeros(10, D_MODEL // 2)), x)


def test_rope_preserves_the_length_of_every_vector() -> None:
    x = _x().view(2, 1, 10, D_MODEL)
    angles = ops.rope_angles(torch.arange(10) * 13, D_MODEL, BASE)
    torch.testing.assert_close(ops.apply_rope(x, angles).norm(dim=-1), x.norm(dim=-1))


# --- ALiBi ----------------------------------------------------------------------------------------


def test_eight_heads_get_the_slopes_the_paper_lists() -> None:
    """§3: "1/2^1, 1/2^2, ..., 1/2^8" for eight heads."""
    assert alibi_slopes(8) == pytest.approx([2.0**-i for i in range(1, 9)])


def test_sixteen_heads_get_the_interpolated_slopes_the_paper_describes() -> None:
    """§3: sixteen heads start at 1/2^0.5 with ratio 1/2^0.5, ending at 1/2^8."""
    assert alibi_slopes(16) == pytest.approx([2.0 ** (-0.5 * i) for i in range(1, 17)])


def test_a_non_power_of_two_head_count_follows_the_official_code() -> None:
    """Six heads: the four slopes for four heads, then every other slope for eight, twice."""
    assert alibi_slopes(6) == pytest.approx([2.0**-2, 2.0**-4, 2.0**-6, 2.0**-8, 2.0**-1, 2.0**-3])


def test_the_alibi_bias_is_zero_on_the_diagonal_and_minus_m_times_distance_below() -> None:
    mixer = AlibiAttention(D_MODEL, 8)
    positions = torch.arange(6)
    bias = mixer.score_bias(positions, positions)
    slopes = torch.tensor(alibi_slopes(8), dtype=DTYPE)
    for i in range(6):
        for j in range(i + 1):
            torch.testing.assert_close(bias[:, i, j], -slopes * (i - j))
    torch.testing.assert_close(torch.diagonal(bias, dim1=1, dim2=2), torch.zeros(8, 6, dtype=DTYPE))


# --- NTK-aware ------------------------------------------------------------------------------------


def test_ntk_aware_at_scale_one_is_rope() -> None:
    torch.manual_seed(0)
    rope, ntk = _pair(
        RopeAttention(D_MODEL, HEADS, BASE), NtkAwareAttention(D_MODEL, HEADS, BASE, 1)
    )
    torch.testing.assert_close(_run(ntk, _x()), _run(rope, _x()))


def test_ntk_aware_divides_the_slowest_frequency_by_s_and_keeps_the_fastest() -> None:
    """App. A.1: the base change is chosen so exactly this holds."""
    dim, scale = 16, 8.0
    rope = ops.rope_angles(torch.tensor([1]), dim, BASE)[0]
    ntk = ops.rope_angles(torch.tensor([1]), dim, ntk_base(BASE, scale, dim))[0]
    torch.testing.assert_close(ntk[0], rope[0])
    torch.testing.assert_close(ntk[-1], rope[-1] / scale)


# --- YaRN -----------------------------------------------------------------------------------------


def _yarn(scale: float, length: int = 16) -> YarnAttention:
    return YarnAttention(D_MODEL, HEADS, BASE, scale, length, int(scale * length), 1, 32)


def test_yarn_at_scale_one_is_rope() -> None:
    """With s = 1 every h(θ_d) is θ_d and sqrt(1/t) = 1, so nothing changes."""
    torch.manual_seed(0)
    rope, yarn = _pair(RopeAttention(D_MODEL, HEADS, BASE), _yarn(1))
    assert yarn.attention_factor == 1.0
    torch.testing.assert_close(_run(yarn, _x()), _run(rope, _x()))


def test_yarn_leaves_fast_frequencies_alone_and_divides_slow_ones_by_s() -> None:
    """Eq 18–20: r > β keeps θ_d; r < α gives θ_d / s; between, a linear blend."""
    dim, scale, length, alpha, beta = 64, 16.0, 4096, 1.0, 32.0
    theta = BASE ** (-2.0 * torch.arange(dim // 2, dtype=DTYPE) / dim)
    ratio = length / (2 * math.pi / theta)
    h = yarn_frequencies(dim, BASE, scale, length, alpha, beta)
    fast, slow = ratio > beta, ratio < alpha
    between = ~fast & ~slow
    assert fast.any() and slow.any() and between.any()
    torch.testing.assert_close(h[fast], theta[fast])
    torch.testing.assert_close(h[slow], theta[slow] / scale)
    gamma = (ratio[between] - alpha) / (beta - alpha)
    torch.testing.assert_close(
        h[between], (1 - gamma) * theta[between] / scale + gamma * theta[between]
    )


def test_yarn_scales_q_and_k_by_the_printed_temperature() -> None:
    """Eq 22: sqrt(1/t) = 0.1 ln(s) + 1, applied to both q and k."""
    assert yarn_attention_factor(16) == pytest.approx(0.1 * math.log(16) + 1)
    mixer = _yarn(16).double()
    q = _x(3, width=mixer.head_dim).view(2, 1, 3, mixer.head_dim)
    positions = torch.arange(3)
    unscaled = ops.apply_rope(q, mixer.angles(positions))
    torch.testing.assert_close(mixer.rotate(q, positions), unscaled * yarn_attention_factor(16))


def test_yarn_refuses_a_length_that_contradicts_its_scale() -> None:
    with pytest.raises(ValueError, match="Eq 11"):
        YarnAttention(D_MODEL, HEADS, BASE, 4, 16, 60, 1, 32)


# --- DroPE ----------------------------------------------------------------------------------------


def _drope(rope_on: bool, extension: float = 2) -> DropeAttention:
    return DropeAttention(D_MODEL, HEADS, BASE, rope_on, extension, 0.103)


def test_drope_with_the_rotation_on_is_rope() -> None:
    torch.manual_seed(0)
    rope, drope = _pair(RopeAttention(D_MODEL, HEADS, BASE), _drope(True))
    torch.testing.assert_close(_run(drope, _x()), _run(rope, _x()))


def test_drope_without_the_rotation_ignores_a_global_position_offset() -> None:
    torch.manual_seed(0)
    mixer = _drope(False).double().eval()
    x = _x()
    torch.testing.assert_close(_run(mixer, x, start=1000), _run(mixer, x))


def test_drope_without_the_rotation_cannot_see_the_order_of_earlier_tokens() -> None:
    """With no position signal the last token's output is a function of a *set*; RoPE's is not."""
    torch.manual_seed(0)
    on, off = _pair(_drope(True), _drope(False))
    x = _x(8)
    shuffled = x.clone()
    shuffled[:, :-1] = x[:, torch.tensor([3, 0, 6, 1, 5, 2, 4])]
    torch.testing.assert_close(_run(off, shuffled)[:, -1], _run(off, x)[:, -1])
    assert not torch.allclose(_run(on, shuffled)[:, -1], _run(on, x)[:, -1])


def test_drope_multiplies_the_scores_by_one_plus_c_log_s() -> None:
    """App. C.2: β = 1 + c ln(s); at s = 1 there is no change to the usual scale."""
    assert _drope(False, 1).score_scale() == pytest.approx(1 / math.sqrt(D_MODEL // HEADS))
    beta = 1 + 0.103 * math.log(2)
    assert _drope(False, 2).score_scale() == pytest.approx(beta / math.sqrt(D_MODEL // HEADS))
    assert _drope(True, 2).score_scale() is None


# --- HD-RoPE --------------------------------------------------------------------------------------


def _hd_matrix(angle: float, width: int = 4) -> torch.Tensor:
    """The linear map Algorithm 2 applies at one angle, one column per basis vector."""
    basis = torch.eye(width, dtype=DTYPE)
    angles = torch.full((width, width // 4), angle, dtype=DTYPE)
    return apply_hd_rope(basis, angles).T


@pytest.mark.parametrize("angle", [0.3, -2.1, 7.9])
def test_the_printed_hd_rope_transform_is_a_rotation(angle: float) -> None:
    """Our computation, not a claim of the paper: orthogonal with determinant 1 at every angle."""
    r = _hd_matrix(angle)
    torch.testing.assert_close(r.T @ r, torch.eye(4, dtype=DTYPE))
    assert torch.linalg.det(r).item() == pytest.approx(1.0)


def test_an_hd_rope_score_depends_only_on_the_offset() -> None:
    """The relative-position property (Eq 14), checked on a whole 16-wide head."""
    mixer = HdRopeAttention(D_MODEL, HEADS, BASE)
    generator = torch.Generator().manual_seed(2)
    q = torch.randn(1, 1, 1, mixer.head_dim, generator=generator, dtype=DTYPE)
    k = torch.randn(1, 1, 1, mixer.head_dim, generator=generator, dtype=DTYPE)
    base_score = _rotated_score(mixer, q, k, 5, 19)
    for shift in (1, 64, 1001):
        torch.testing.assert_close(_rotated_score(mixer, q, k, 5 + shift, 19 + shift), base_score)
    assert not torch.isclose(_rotated_score(mixer, q, k, 5, 20), base_score)


def test_the_printed_algorithm_is_q4_r_q4_transposed_not_the_eq_13_order() -> None:
    """What we computed about Algorithm 2 against Eq 13/19, recorded so it cannot drift silently.

    With one θ per 4D block (Eq 16), `Q4ᵀ R Q4` is again a rotation in two fixed planes. The
    printed algorithm is `Q4 R Q4ᵀ`, and `det(Q4) = −1`, so `Q4` is not in SO(4).
    """
    c4 = torch.tensor([[0, 1, 1, 1], [1, 0, -1, 1], [1, 1, 0, -1], [1, -1, 1, 0]], dtype=DTYPE)
    q4 = c4 / math.sqrt(3)
    angle = 0.9
    c, s = math.cos(angle), math.sin(angle)
    r = torch.tensor([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, c, -s], [0, 0, s, c]], dtype=DTYPE)
    printed = _hd_matrix(angle)
    torch.testing.assert_close(q4 @ r @ q4.T, printed)
    assert not torch.allclose(q4.T @ r @ q4, printed)
    torch.testing.assert_close(c4 @ c4.T, 3 * torch.eye(4, dtype=DTYPE))
    assert torch.linalg.det(q4).item() == pytest.approx(-1.0)


def test_hd_rope_refuses_a_head_width_not_divisible_by_four() -> None:
    with pytest.raises(ValueError, match="divisible by 4"):
        HdRopeAttention(12, 2, BASE)
