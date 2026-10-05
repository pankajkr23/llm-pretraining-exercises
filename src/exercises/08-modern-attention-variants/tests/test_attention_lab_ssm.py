"""The recurrent family's own identities: Mamba's two paths and rules, Mamba-3's three reductions.

Each identity is checked against a calculation written here independently of `attention.lab.ssm`
— a matrix exponential for Eq 4, a plain Mamba-2 loop for λ = 1, a plain Eq 5 loop for a
non-rotating state — so a shared mistake cannot make both sides agree.
"""

import math

import pytest

torch = pytest.importorskip("torch", reason="the attention lab needs the train extra")

from attention.lab import registry, ssm  # noqa: E402
from attention.lab.base import tensor_bytes  # noqa: E402

F64 = torch.float64


def _gen(seed: int = 0):
    return torch.Generator().manual_seed(seed)


def _randn(*shape, seed: int = 0):
    return torch.randn(*shape, generator=_gen(seed), dtype=F64)


def _build(name: str, **overrides):
    torch.manual_seed(0)
    return registry.get(name).build("lab", **overrides).double().eval()


# --- mamba ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("discretization", ssm.DISCRETIZATIONS)
def test_mamba_parallel_form_equals_the_scan(discretization) -> None:
    """The cumulative-log-decay form and the recurrence are the same function, from any state."""
    scan = _build("mamba", mode="scan", discretization=discretization)
    parallel = _build("mamba", mode="parallel", discretization=discretization)
    x = _randn(2, 11, 32)
    with torch.no_grad():
        _, warm = scan(_randn(2, 5, 32, seed=1))  # a non-zero starting state and conv buffer
        y_scan, s_scan = scan(x, warm)
        y_par, s_par = parallel(x, warm)
    torch.testing.assert_close(y_par, y_scan)
    torch.testing.assert_close(s_par["h"], s_scan["h"])
    torch.testing.assert_close(s_par["conv"], s_scan["conv"])


def test_zoh_is_equation_4_computed_with_a_matrix_exponential() -> None:
    """One step from a zero state with x = 1 leaves exactly B̄ in the state; B̄ must be Eq 4.

    The reference builds `(ΔA)^{-1}(exp(ΔA) - I) ΔB` from full matrices with `matrix_exp` and a
    linear solve, not from the elementwise shortcut the implementation uses.
    """
    channels, n = 3, 5
    a = -(torch.arange(n, dtype=F64) + 1).repeat(channels, 1) * (1 + _randn(channels, n).abs())
    delta = _randn(1, 1, channels, seed=1).abs() + 0.2
    b_in = _randn(1, 1, n, seed=2)
    c_out = torch.zeros(1, 1, n, dtype=F64)
    _, h = ssm.selective_scan(
        torch.ones(1, 1, channels, dtype=F64),
        delta,
        a,
        b_in,
        c_out,
        torch.zeros(channels, dtype=F64),
        torch.zeros(1, channels, n, dtype=F64),
        "zoh",
    )
    for d in range(channels):
        step = delta[0, 0, d]
        delta_a = torch.diag(step * a[d])
        expected = torch.linalg.solve(
            delta_a, torch.linalg.matrix_exp(delta_a) - torch.eye(n, dtype=F64)
        ) @ (step * b_in[0, 0])
        torch.testing.assert_close(h[0, d], expected)


def test_zoh_minus_euler_vanishes_at_second_order_with_the_predicted_constant() -> None:
    """As Δ = εΔ₀ → 0, `(y_zoh - y_euler) / ε²` tends to `½ Σ_{s≤t} C_t·(Δ₀,s² A ⊙ B_s) x_s`.

    Why: Ā is identical under both rules, and B̄_zoh = εΔ₀B·(exp(z)-1)/z with z = εΔ₀A, which is
    B̄_euler·(1 + z/2 + O(z²)). So the two states differ by `Σ_s Ā(s→t)·ε²Δ₀²AB_s x_s / 2 + O(ε³)`,
    and `Ā(s→t) = 1 + O(ε)`. The remainder is first order in ε, so the relative error at ε = 1e-6
    must be tiny while at ε = 1e-2 it must not be — the relation is a limit, not an identity.
    """
    batch, steps, channels, n = 2, 10, 3, 4
    x = _randn(batch, steps, channels)
    delta0 = torch.rand(batch, steps, channels, generator=_gen(1), dtype=F64) + 0.5
    a = -(torch.arange(n, dtype=F64) + 1).repeat(channels, 1)
    b_in, c_out = _randn(batch, steps, n, seed=2), _randn(batch, steps, n, seed=3)
    skip = _randn(channels, seed=4)
    h = torch.zeros(batch, channels, n, dtype=F64)
    at_or_before = torch.tril(torch.ones(steps, steps, dtype=F64))  # [t, s], 1 where s <= t
    limit = 0.5 * torch.einsum(
        "btn,bscn,bsc,ts->btc",
        c_out,
        (delta0**2).unsqueeze(-1) * a * b_in.unsqueeze(2),
        x,
        at_or_before,
    )

    def relative_error(eps: float) -> float:
        zoh, _ = ssm.selective_scan(x, eps * delta0, a, b_in, c_out, skip, h, "zoh")
        euler, _ = ssm.selective_scan(x, eps * delta0, a, b_in, c_out, skip, h, "euler")
        gap = (zoh - euler) / eps**2
        return ((gap - limit).abs().max() / limit.abs().max()).item()

    assert relative_error(1e-6) < 1e-4
    assert relative_error(1e-2) > 1e-2


def test_mamba_starts_from_the_papers_choices() -> None:
    """S4D-Real `A_n = -(n+1)`, ZOH by default, and Δ inside the stated initial range."""
    mixer = _build("mamba")
    n = mixer.state_size
    expected = -(torch.arange(n, dtype=F64) + 1).expand(mixer.inner, n)
    torch.testing.assert_close(mixer.a.detach(), expected)
    assert mixer.discretization == "zoh"
    dt = torch.nn.functional.softplus(mixer.dt_proj.bias.detach())
    assert dt.min() >= 0.001 - 1e-9 and dt.max() <= 0.1 + 1e-9


@pytest.mark.parametrize("name", ["mamba", "mamba3"])
def test_the_state_is_the_same_size_after_one_token_and_after_many(name) -> None:
    """Constant state, including the convolution buffer and Mamba-3's previous-token memory."""
    overrides = {"use_conv": True} if name == "mamba3" else {}
    mixer = _build(name, **overrides)
    sizes = []
    with torch.no_grad():
        for steps in (1, 3, 40):
            _, state = mixer(_randn(2, steps, 32))
            sizes.append(
                (
                    tensor_bytes(state),
                    {k: tuple(v.shape) for k, v in state.items() if torch.is_tensor(v)},
                )
            )
    assert sizes[0] == sizes[1] == sizes[2], sizes


# --- mamba3 --------------------------------------------------------------------------------------


def _mamba3_inputs(seed: int = 0, batch: int = 2, heads: int = 3, steps: int = 9):
    p, n = 4, 6
    return {
        "x": _randn(batch, heads, steps, p, seed=seed),
        "b_in": _randn(batch, heads, steps, n, seed=seed + 1),
        "c_out": _randn(batch, heads, steps, n, seed=seed + 2),
        "log_alpha": -_randn(batch, heads, steps, seed=seed + 3).abs(),
        "delta": _randn(batch, heads, steps, seed=seed + 4).abs() + 0.1,
        "lam": torch.rand(batch, heads, steps, generator=_gen(seed + 5), dtype=F64),
        "angles": _randn(batch, heads, steps, n // 2, seed=seed + 6),
        "skip": _randn(heads, seed=seed + 7),
        "state": {
            "h": _randn(batch, heads, p, n, seed=seed + 8),
            "b_prev": _randn(batch, heads, n, seed=seed + 9),
            "x_prev": _randn(batch, heads, p, seed=seed + 10),
        },
    }


def _plain_eq5(x, b_in, c_out, log_alpha, delta, lam, skip, state):
    """Eq 5 with a real state, written out: no rotation anywhere."""
    h, b_prev, x_prev = state["h"], state["b_prev"], state["x_prev"]
    ys = []
    for t in range(x.shape[2]):
        alpha = log_alpha[:, :, t].exp()
        beta = (1 - lam[:, :, t]) * delta[:, :, t] * alpha
        gamma = lam[:, :, t] * delta[:, :, t]
        h = (
            alpha[..., None, None] * h
            + beta[..., None, None] * torch.einsum("bhp,bhn->bhpn", x_prev, b_prev)
            + gamma[..., None, None] * torch.einsum("bhp,bhn->bhpn", x[:, :, t], b_in[:, :, t])
        )
        ys.append(torch.einsum("bhpn,bhn->bhp", h, c_out[:, :, t]) + skip[:, None] * x[:, :, t])
        b_prev, x_prev = b_in[:, :, t], x[:, :, t]
    return torch.stack(ys, dim=2), h


def _plain_mamba2(x, b_in, c_out, log_alpha, delta, skip, h):
    """Mamba-2's exponential-Euler update, Eq 1: `h_t = α_t h_{t-1} + Δ_t B_t x_t^T`."""
    ys = []
    for t in range(x.shape[2]):
        h = log_alpha[:, :, t].exp()[..., None, None] * h + delta[:, :, t][
            ..., None, None
        ] * torch.einsum("bhp,bhn->bhpn", x[:, :, t], b_in[:, :, t])
        ys.append(torch.einsum("bhpn,bhn->bhp", h, c_out[:, :, t]) + skip[:, None] * x[:, :, t])
    return torch.stack(ys, dim=2), h


@pytest.mark.parametrize("run", ["trapezoidal_scan", "trapezoidal_parallel"])
def test_lambda_one_is_mamba2s_exponential_euler_update(run) -> None:
    """With λ_t = 1 the previous token's term vanishes and Eq 5 is Eq 1 (Proposition 1, (b))."""
    inputs = _mamba3_inputs()
    inputs["lam"] = torch.ones_like(inputs["lam"])
    inputs["angles"] = torch.zeros_like(inputs["angles"])
    y, state = getattr(ssm, run)(**inputs)
    y_ref, h_ref = _plain_mamba2(
        inputs["x"],
        inputs["b_in"],
        inputs["c_out"],
        inputs["log_alpha"],
        inputs["delta"],
        inputs["skip"],
        inputs["state"]["h"],
    )
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(state["h"], h_ref)
    # And the identity is not vacuous: with a learned λ the two differ.
    inputs["lam"] = torch.full_like(inputs["lam"], 0.5)
    y_half, _ = getattr(ssm, run)(**inputs)
    assert (y_half - y_ref).abs().max() > 1e-3


@pytest.mark.parametrize("run", ["trapezoidal_scan", "trapezoidal_parallel"])
def test_zero_rotation_is_the_real_valued_trapezoidal_update(run) -> None:
    """With every angle Δ_t θ_t = 0, R_t = I and the complex-state layer is plain Eq 5."""
    inputs = _mamba3_inputs(seed=3)
    inputs["angles"] = torch.zeros_like(inputs["angles"])
    y, state = getattr(ssm, run)(**inputs)
    y_ref, h_ref = _plain_eq5(**{k: v for k, v in inputs.items() if k != "angles"})
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(state["h"], h_ref)


def test_a_layer_whose_angle_projection_is_zero_equals_the_non_rotating_layer() -> None:
    """The same statement one level up: θ_t = 0 in the module reproduces `rotate=False`."""
    rotating = _build("mamba3")
    real = _build("mamba3", rotate=False)
    real.load_state_dict(rotating.state_dict())
    with torch.no_grad():
        rotating.in_proj.weight[-rotating.sizes[-1] :] = 0.0
        real.in_proj.weight[-real.sizes[-1] :] = 0.0
        x = _randn(2, 10, 32)
        torch.testing.assert_close(rotating(x)[0], real(x)[0])
        # and the rotation does something once θ is non-zero
        rotating.in_proj.weight[-rotating.sizes[-1] :] = 1.0
        assert (rotating(x)[0] - real(x)[0]).abs().max() > 1e-6


def test_the_rotation_is_orthogonal_and_so_preserves_the_state_norm() -> None:
    """R(θ) of Proposition 2 is a rotation: `RᵀR = I`, `det R = 1`, and ‖R h‖ = ‖h‖."""
    n = 6
    angles = _randn(n // 2, seed=5) * 3
    basis = torch.eye(n, dtype=F64)
    matrix = ssm.rotate_pairs(basis, angles).T  # column j is R e_j
    torch.testing.assert_close(matrix.T @ matrix, torch.eye(n, dtype=F64))
    torch.testing.assert_close(torch.linalg.det(matrix), torch.tensor(1.0, dtype=F64))
    first = matrix[:2, :2]
    c, s = math.cos(angles[0].item()), math.sin(angles[0].item())
    torch.testing.assert_close(first, torch.tensor([[c, -s], [s, c]], dtype=F64))
    h = _randn(4, 3, 5, n, seed=6)
    many = _randn(4, 3, 1, n // 2, seed=7)
    torch.testing.assert_close(
        torch.linalg.vector_norm(ssm.rotate_pairs(h, many), dim=-1),
        torch.linalg.vector_norm(h, dim=-1),
    )


def test_the_rope_trick_equals_rotating_the_state() -> None:
    """Proposition 4: rotating B and C by the cumulative angle gives the direct rotation's outputs.

    Checked with non-zero angles, a learned λ and a non-zero starting state, and including the
    state handed back, so a following call continues correctly from either path.
    """
    inputs = _mamba3_inputs(seed=11)
    y_scan, s_scan = ssm.trapezoidal_scan(**inputs)
    y_par, s_par = ssm.trapezoidal_parallel(**inputs)
    torch.testing.assert_close(y_par, y_scan)
    for key in ("h", "b_prev", "x_prev"):
        torch.testing.assert_close(s_par[key], s_scan[key])


def test_mamba3_layer_paths_agree_with_and_without_the_short_convolution() -> None:
    """The module's parallel and scan modes agree, the optional conv included, across a split."""
    for use_conv in (False, True):
        scan = _build("mamba3", mode="scan", use_conv=use_conv)
        parallel = _build("mamba3", mode="parallel", use_conv=use_conv)
        x = _randn(2, 12, 32)
        with torch.no_grad():
            y_scan, _ = scan(x)
            head, state = parallel(x[:, :5])
            tail, _ = parallel(x[:, 5:], state)
        torch.testing.assert_close(torch.cat([head, tail], dim=1), y_scan)


def test_mamba3_starts_from_the_papers_choices() -> None:
    """No short conv by default, biases start at one, and λ is a sigmoid (so inside [0, 1])."""
    mixer = _build("mamba3")
    assert mixer.conv is None
    assert mixer.trapezoid and mixer.rotate
    torch.testing.assert_close(mixer.b_bias.detach(), torch.ones_like(mixer.b_bias))
    torch.testing.assert_close(mixer.c_bias.detach(), torch.ones_like(mixer.c_bias))
    assert mixer.state_size % 2 == 0


def test_both_variants_build_at_paper_scale_with_the_stated_shapes() -> None:
    """Paper-scale specs build: N = 16, E = 2 for Mamba; N = 128, P = 64, E = 2 for Mamba-3."""
    mamba = registry.get("mamba").build("paper")
    assert (mamba.state_size, mamba.inner // mamba.d_model) == (16, 2)
    mamba3 = registry.get("mamba3").build("paper")
    assert (mamba3.state_size, mamba3.head_dim, mamba3.inner // mamba3.d_model) == (128, 64, 2)
